from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DiscoConfig:
    batch_name: str = "Disco"
    seed: int = -1
    n_batches: int = 1
    display_rate: int = 5

    diffusion_model: str = "512x512_diffusion_uncond_finetune_008100"
    use_secondary_model: bool = False
    diffusion_sampling_mode: str = "ddim"
    use_checkpoint: bool = True
    use_fp16: bool = True

    width: int = 512
    height: int = 512
    steps: int = 250
    eta: float = 0.3

    clip_guidance_scale: int = 5000
    tv_scale: int = 0
    range_scale: int = 150
    sat_scale: int = 0
    cutn_batches: int = 4
    skip_augs: bool = False

    clamp_grad: bool = True
    clamp_max: float = 0.05
    randomize_class: bool = True
    clip_denoised: bool = False
    fuzzy_prompt: bool = False
    rand_mag: float = 0.05

    cut_overview: str = "[12]*400+[4]*600"
    cut_innercut: str = "[4]*400+[12]*600"
    cut_ic_pow: float = 1.0
    cut_icgray_p: str = "[0.2]*400+[0]*600"

    clip_vit_b32: bool = True
    clip_vit_b16: bool = False
    clip_vit_l14: bool = False
    clip_vit_l14_336: bool = False
    clip_rn50: bool = True
    clip_rn101: bool = False
    clip_rn50x4: bool = False
    clip_rn50x16: bool = False
    clip_rn50x64: bool = False

    init_image: str = ""
    init_scale: int = 1000
    skip_steps: int = 10
    perlin_init: bool = False
    perlin_mode: str = "mixed"

    animation_mode: str = "None"
    max_frames: int = 1
    interp_spline: str = "Linear"
    key_frames: bool = True

    angle: str = "0:(0)"
    zoom: str = "0:(1)"
    translation_x: str = "0:(0)"
    translation_y: str = "0:(0)"
    translation_z: str = "0:(10.0)"
    rotation_3d_x: str = "0:(0)"
    rotation_3d_y: str = "0:(0)"
    rotation_3d_z: str = "0:(0)"

    midas_depth_model: str = "dpt_large"
    midas_weight: float = 0.3
    near_plane: int = 200
    far_plane: int = 10000
    fov: int = 40
    padding_mode: str = "border"
    sampling_mode: str = "bicubic"

    turbo_mode: bool = False
    turbo_steps: int = 3
    turbo_preroll: int = 10

    vr_mode: bool = False
    vr_eye_angle: float = 0.5
    vr_ipd: float = 5.0

    video_init_path: str = ""
    extract_nth_frame: int = 1
    video_init_seed_continuity: bool = False
    flow_warp: bool = True
    flow_blend: float = 0.5
    check_consistency: bool = True

    frames_scale: int = 1500
    frames_skip_steps: str = "60%"

    intermediate_saves: int = 0
    intermediates_in_subfolder: bool = True

    text_prompts: dict[str, list[str]] = field(
        default_factory=lambda: {"0": ["A beautiful painting of a starry night sky, trending on artstation."]}
    )
    image_prompts: dict[str, list[str]] = field(default_factory=dict)

    blend_mode: str = "optical flow"
    output_blend: float = 0.5
    fps: int = 12
    output_format: str = "mp4"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def width_height(self) -> list[int]:
        return [self.width, self.height]

    @property
    def side_x(self) -> int:
        return (self.width // 64) * 64

    @property
    def side_y(self) -> int:
        return (self.height // 64) * 64

    def normalized_text_prompts(self) -> dict[int, list[str]]:
        return {int(k): v for k, v in self.text_prompts.items()}

    def normalized_image_prompts(self) -> dict[int, list[str]]:
        return {int(k): v for k, v in self.image_prompts.items()}
