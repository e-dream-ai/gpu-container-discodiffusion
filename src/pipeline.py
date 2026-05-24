from __future__ import annotations

import math
import os
import pathlib
import random
import re
import shutil
import subprocess
from glob import glob
from types import SimpleNamespace
from typing import Optional

import numpy as np
import PIL
import torch

from . import disco, warp
from .config import DiscoConfig

OUTPUT_DIR = os.environ.get("DISCO_OUTPUT_DIR", "/workspace/output")
CONSISTENCY_CHECKER_BIN = os.environ.get(
    "DISCO_CONSISTENCY_CHECKER",
    "/disco/neural-style-tf/video_input/consistencyChecker/consistencyChecker",
)

_CACHED_MODELS: Optional[dict] = None
_CACHED_CFG_KEY: Optional[tuple] = None


def _cache_key(cfg: DiscoConfig) -> tuple:
    return (
        cfg.diffusion_model,
        cfg.steps,
        cfg.use_secondary_model,
        cfg.use_checkpoint,
        cfg.use_fp16,
        cfg.clip_vit_b32, cfg.clip_vit_b16, cfg.clip_vit_l14, cfg.clip_vit_l14_336,
        cfg.clip_rn50, cfg.clip_rn101, cfg.clip_rn50x4, cfg.clip_rn50x16, cfg.clip_rn50x64,
    )


def load_models(cfg: DiscoConfig) -> dict:
    global _CACHED_MODELS, _CACHED_CFG_KEY
    key = _cache_key(cfg)
    if _CACHED_MODELS is not None and _CACHED_CFG_KEY == key:
        print("Reusing cached models")
        return _CACHED_MODELS

    print("Loading models...")
    model, diffusion_obj = disco.load_diffusion_model(cfg)
    clip_models = disco.load_clip_models(cfg)
    secondary_model = disco.load_secondary_model() if cfg.use_secondary_model else None
    lpips_model = disco.load_lpips()

    _CACHED_MODELS = {
        "model": model,
        "diffusion": diffusion_obj,
        "clip_models": clip_models,
        "secondary_model": secondary_model,
        "lpips_model": lpips_model,
        "normalize": disco.CLIP_NORMALIZE,
    }
    _CACHED_CFG_KEY = key
    return _CACHED_MODELS


_TERM_RE = re.compile(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*\]\s*\*\s*(\d+)")


def parse_cut_schedule(expr) -> list:
    if isinstance(expr, list):
        return expr
    expr = str(expr).strip()
    if not expr:
        return []
    parts = [p.strip() for p in expr.split("+")]
    out: list = []
    for part in parts:
        match = _TERM_RE.fullmatch(part)
        if not match:
            raise ValueError(f"Cannot parse cut-schedule term: {part!r}")
        value_str, count_str = match.group(1), match.group(2)
        value = float(value_str) if "." in value_str else int(value_str)
        out.extend([value] * int(count_str))
    return out


def extract_video_frames(video_path: str, frames_folder: str, extract_nth_frame: int = 1) -> int:
    os.makedirs(frames_folder, exist_ok=True)
    for f in pathlib.Path(frames_folder).glob("*.jpg"):
        f.unlink()
    vf = f"select=not(mod(n\\,{extract_nth_frame}))"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-vsync", "vfr",
         "-q:v", "2", "-loglevel", "error", "-stats",
         os.path.join(frames_folder, "%04d.jpg")],
        check=True,
    )
    return len(sorted(glob(os.path.join(frames_folder, "*.jpg"))))


def _resolve_seed(cfg: DiscoConfig) -> int:
    if cfg.seed is None or cfg.seed < 0:
        random.seed()
        return random.randint(0, 2 ** 32 - 1)
    return int(cfg.seed)


def build_args(cfg: DiscoConfig, *, max_frames: int, batch_num: int, start_frame: int,
               batch_name: str, text_prompts: dict, image_prompts: dict) -> SimpleNamespace:
    side_x = (cfg.width // 64) * 64
    side_y = (cfg.height // 64) * 64

    skip_step_ratio = int(cfg.frames_skip_steps.rstrip("%")) / 100
    calc_frames_skip_steps = math.floor(cfg.steps * skip_step_ratio)
    if cfg.steps <= calc_frames_skip_steps:
        raise ValueError("steps must be greater than frames_skip_steps")

    if isinstance(cfg.intermediate_saves, list):
        steps_per_checkpoint = None
        intermediate_saves = cfg.intermediate_saves
    elif cfg.intermediate_saves:
        steps_per_checkpoint = max(
            1, math.floor((cfg.steps - cfg.skip_steps - 1) // (cfg.intermediate_saves + 1)),
        )
        intermediate_saves = []
    else:
        steps_per_checkpoint = cfg.steps + 10
        intermediate_saves = []

    series = {}
    if cfg.key_frames and cfg.animation_mode in ("2D", "3D"):
        for field in ("angle", "zoom", "translation_x", "translation_y", "translation_z",
                      "rotation_3d_x", "rotation_3d_y", "rotation_3d_z"):
            series[f"{field}_series"] = disco.safe_keyframe_series(
                getattr(cfg, field), max_frames, cfg.interp_spline,
            )

    prompts_series = (
        disco.split_prompts(text_prompts, max_frames) if text_prompts else None
    )
    image_prompts_series = (
        disco.split_prompts(image_prompts, max_frames) if image_prompts else None
    )

    args = SimpleNamespace(
        batchNum=batch_num,
        batch_name=batch_name,
        seed=_resolve_seed(cfg),
        display_rate=cfg.display_rate,
        n_batches=cfg.n_batches if cfg.animation_mode == "None" else 1,
        steps=cfg.steps,
        diffusion_sampling_mode=cfg.diffusion_sampling_mode,
        width_height=[cfg.width, cfg.height],
        side_x=side_x,
        side_y=side_y,
        cutn_batches=cfg.cutn_batches,
        init_image=cfg.init_image,
        init_scale=cfg.init_scale,
        skip_steps=cfg.skip_steps,
        animation_mode=cfg.animation_mode,
        key_frames=cfg.key_frames,
        max_frames=max_frames,
        interp_spline=cfg.interp_spline,
        start_frame=start_frame,
        midas_depth_model=cfg.midas_depth_model,
        midas_weight=cfg.midas_weight,
        near_plane=cfg.near_plane,
        far_plane=cfg.far_plane,
        fov=cfg.fov,
        padding_mode=cfg.padding_mode,
        sampling_mode=cfg.sampling_mode,
        frames_scale=cfg.frames_scale,
        calc_frames_skip_steps=calc_frames_skip_steps,
        skip_step_ratio=skip_step_ratio,
        prompts_series=prompts_series,
        image_prompts_series=image_prompts_series,
        text_prompts=text_prompts,
        image_prompts=image_prompts,
        cut_overview=parse_cut_schedule(cfg.cut_overview),
        cut_innercut=parse_cut_schedule(cfg.cut_innercut),
        cut_ic_pow=cfg.cut_ic_pow,
        cut_icgray_p=parse_cut_schedule(cfg.cut_icgray_p),
        intermediate_saves=intermediate_saves,
        intermediates_in_subfolder=cfg.intermediates_in_subfolder,
        steps_per_checkpoint=steps_per_checkpoint,
        clamp_grad=cfg.clamp_grad,
        clamp_max=cfg.clamp_max,
        fuzzy_prompt=cfg.fuzzy_prompt,
        rand_mag=cfg.rand_mag,
    )
    args.__dict__.update(series)
    return args


def _ffmpeg_encode(image_pattern: str, output_path: str, fps: int, last_frame: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-vcodec", "png", "-r", str(fps), "-start_number", "0",
        "-i", image_pattern, "-frames:v", str(last_frame),
        "-c:v", "libx264", "-vf", f"fps={fps}", "-pix_fmt", "yuv420p",
        "-crf", "17", "-preset", "medium", output_path,
    ]
    subprocess.run(cmd, check=True)


def assemble_video(cfg: DiscoConfig, batch_folder: str, batch_name: str,
                   batch_num: int, flo_folder: str = "") -> str:
    folder = batch_name
    run = batch_num
    pattern = os.path.join(batch_folder, f"{folder}({run})_%04d.png")
    out_path = os.path.join(batch_folder, f"{folder}({run}).mp4")

    blend_mode = cfg.blend_mode
    blend = cfg.output_blend

    if blend_mode == "optical flow" and cfg.animation_mode == "Video Input" and flo_folder:
        flow_dir = os.path.join(batch_folder, "flow")
        os.makedirs(flow_dir, exist_ok=True)
        frames_in = sorted(glob(os.path.join(batch_folder, f"{folder}({run})_*.png")))
        if frames_in:
            shutil.copy(frames_in[0], flow_dir)
        for i, (a, b) in enumerate(zip(frames_in[:-1], frames_in[1:]), start=1):
            frame1 = PIL.Image.open(a)
            frame2 = PIL.Image.open(b)
            stem = os.path.basename(a)
            num = int(stem.split("_")[-1][:-4]) + 1
            flo_path = os.path.join(flo_folder, f"{num:04}.jpg.npy")
            weights_path = (os.path.join(flo_folder, f"{num:04}.jpg-21.txt")
                            if cfg.check_consistency else None)
            if not os.path.exists(flo_path):
                continue
            warp.warp(frame1, frame2, flo_path, blend=blend, weights_path=weights_path).save(
                os.path.join(flow_dir, f"{folder}({run})_{i:04}.png")
            )
        pattern = os.path.join(flow_dir, f"{folder}({run})_%04d.png")
        out_path = os.path.join(batch_folder, f"{folder}({run})_flow.mp4")
        last_frame = len(glob(os.path.join(flow_dir, f"{folder}({run})_*.png")))
    elif blend_mode == "linear":
        blend_dir = os.path.join(batch_folder, "blend")
        os.makedirs(blend_dir, exist_ok=True)
        frames_in = sorted(glob(os.path.join(batch_folder, f"{folder}({run})_*.png")))
        if frames_in:
            shutil.copy(frames_in[0], blend_dir)
        for i, (a, b) in enumerate(zip(frames_in[:-1], frames_in[1:]), start=1):
            f1 = np.array(PIL.Image.open(a))
            f2 = np.array(PIL.Image.open(b))
            mixed = (f1 * (1 - blend) + f2 * blend).astype("uint8")
            PIL.Image.fromarray(mixed).save(
                os.path.join(blend_dir, f"{folder}({run})_{i:04}.png")
            )
        pattern = os.path.join(blend_dir, f"{folder}({run})_%04d.png")
        out_path = os.path.join(batch_folder, f"{folder}({run})_blend.mp4")
        last_frame = len(glob(os.path.join(blend_dir, f"{folder}({run})_*.png")))
    else:
        last_frame = len(glob(os.path.join(batch_folder, f"{folder}({run})_*.png")))

    _ffmpeg_encode(pattern, out_path, cfg.fps, last_frame)
    return out_path

def run_job(cfg: DiscoConfig, *, output_root: Optional[str] = None,
            progress_cb=None) -> dict:
    output_root = output_root or OUTPUT_DIR
    os.makedirs(output_root, exist_ok=True)

    batch_name = cfg.batch_name
    batch_folder = os.path.join(output_root, batch_name)
    os.makedirs(batch_folder, exist_ok=True)

    batch_num = len(glob(os.path.join(batch_folder, "*_settings.txt")))

    video_frames_folder = ""
    flo_folder = ""
    if cfg.animation_mode == "Video Input":
        if not cfg.video_init_path or not os.path.exists(cfg.video_init_path):
            raise FileNotFoundError(f"video_init_path not found: {cfg.video_init_path}")
        video_frames_folder = os.path.join(batch_folder, "video_frames")
        n_frames = extract_video_frames(cfg.video_init_path, video_frames_folder, cfg.extract_nth_frame)
        max_frames = n_frames
        flo_fwd = os.path.join(video_frames_folder, "out_flo_fwd")
        flo_bck = os.path.join(video_frames_folder, "out_flo_bck")
        warp.generate_optical_flows(
            video_frames_folder, flo_fwd, flo_bck,
            (cfg.width // 64) * 64, (cfg.height // 64) * 64,
            check_consistency=cfg.check_consistency,
            consistency_checker_bin=CONSISTENCY_CHECKER_BIN,
        )
        flo_folder = flo_fwd
    elif cfg.animation_mode == "None":
        max_frames = 1
    else:
        max_frames = cfg.max_frames

    args = build_args(
        cfg,
        max_frames=max_frames,
        batch_num=batch_num,
        start_frame=0,
        batch_name=batch_name,
        text_prompts=cfg.normalized_text_prompts(),
        image_prompts=cfg.normalized_image_prompts(),
    )

    models = load_models(cfg)

    partial_folder = os.path.join(batch_folder, "partials") if cfg.intermediates_in_subfolder else batch_folder
    os.makedirs(partial_folder, exist_ok=True)

    rt = disco.Runtime(
        args=args,
        cfg=cfg,
        diffusion=models["diffusion"],
        model=models["model"],
        secondary_model=models["secondary_model"],
        clip_models=models["clip_models"],
        lpips_model=models["lpips_model"],
        normalize=models["normalize"],
        batch_folder=batch_folder,
        partial_folder=partial_folder,
        video_frames_folder=video_frames_folder,
        flo_folder=flo_folder,
    )
    disco.set_runtime(rt)

    cwd = os.getcwd()
    work_dir = os.path.join(batch_folder, "_work")
    os.makedirs(work_dir, exist_ok=True)
    try:
        os.chdir(work_dir)
        disco.do_run()
    finally:
        os.chdir(cwd)
        torch.cuda.empty_cache()

    if cfg.animation_mode != "None" and max_frames > 1:
        video_path = assemble_video(cfg, batch_folder, batch_name, batch_num, flo_folder)
        return {"output_path": video_path, "batch_folder": batch_folder,
                "batch_num": batch_num, "frames": max_frames, "seed": args.seed}

    images = sorted(glob(os.path.join(batch_folder, f"{batch_name}({batch_num})_*.png")))
    return {"output_path": images[-1] if images else "",
            "batch_folder": batch_folder, "batch_num": batch_num,
            "frames": len(images), "seed": args.seed}
