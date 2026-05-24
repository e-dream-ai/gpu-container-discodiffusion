from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from tqdm import tqdm

RAFT_WEIGHTS_PATH = os.environ.get("DISCO_RAFT_PATH", "/disco/RAFT/models/raft-things.pth")
TAG_CHAR = np.array([202021.25], np.float32)

_raft_model = None


def load_raft():
    global _raft_model
    if _raft_model is not None:
        return _raft_model
    from raft import RAFT
    ns = argparse.Namespace(small=False, mixed_precision=True)
    model = torch.nn.DataParallel(RAFT(ns))
    model.load_state_dict(torch.load(RAFT_WEIGHTS_PATH))
    model = model.module.cuda()
    model.train(False)
    _raft_model = model
    return _raft_model


def _load_img(path: str, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def get_flow(frame1: torch.Tensor, frame2: torch.Tensor, model=None, iters: int = 20) -> np.ndarray:
    from utils.utils import InputPadder
    model = model or load_raft()
    padder = InputPadder(frame1.shape)
    f1, f2 = padder.pad(frame1, frame2)
    _, flow12 = model(f1, f2, iters=iters, test_mode=True)
    return flow12[0].permute(1, 2, 0).detach().cpu().numpy()


def warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


def write_flow(filename: str, uv: np.ndarray) -> None:
    assert uv.ndim == 3 and uv.shape[2] == 2
    u, v = uv[:, :, 0], uv[:, :, 1]
    height, width = u.shape
    with open(filename, "wb") as f:
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        tmp = np.zeros((height, width * 2))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def read_weights_file(path: str) -> np.ndarray:
    lines = open(path).readlines()
    header = list(map(int, lines[0].split(" ")))
    w, h = header[0], header[1]
    vals = np.zeros((h, w), dtype=np.float32)
    for i in range(1, len(lines)):
        line = lines[i].rstrip().split(" ")
        vals[i - 1] = np.array(list(map(np.float32, line)))
        vals[i - 1] = list(map(lambda x: 0.0 if x < 255.0 else 1.0, vals[i - 1]))
    return np.dstack([vals.astype(np.float32)] * 3)


def warp(frame1: Image.Image, frame2: Image.Image, flo_path: str,
         blend: float = 0.5, weights_path: Optional[str] = None) -> Image.Image:
    flow21 = np.load(flo_path)
    h, w = flow21.shape[0], flow21.shape[1]
    frame1_arr = np.array(frame1.convert("RGB").resize((w, h)))
    frame2_arr = np.array(frame2.convert("RGB").resize((w, h)))
    warped = warp_flow(frame1_arr, flow21)
    if weights_path and os.path.exists(weights_path):
        fw = read_weights_file(weights_path)
        blended = frame2_arr * (1 - blend) + blend * (warped * fw + frame2_arr * (1 - fw))
    else:
        blended = frame2_arr * (1 - blend) + warped * blend
    return PIL.Image.fromarray(blended.astype("uint8"))


def generate_optical_flows(frames_folder: str, flo_fwd_folder: str, flo_bck_folder: str,
                           width: int, height: int, check_consistency: bool = False,
                           consistency_checker_bin: Optional[str] = None) -> None:
    os.makedirs(flo_fwd_folder, exist_ok=True)
    os.makedirs(flo_bck_folder, exist_ok=True)
    for path in glob(os.path.join(flo_fwd_folder, "*")) + glob(os.path.join(flo_bck_folder, "*")):
        os.remove(path)

    frames = sorted(glob(os.path.join(frames_folder, "*.jpg")))
    if len(frames) < 2:
        print(f"Not enough frames ({len(frames)}) to compute optical flow")
        return

    model = load_raft()
    size = (width, height)
    temp_dir = os.path.join(frames_folder, "temp_flo")
    os.makedirs(temp_dir, exist_ok=True)

    for frame1_path, frame2_path in tqdm(list(zip(frames[:-1], frames[1:])), desc="Optical flow"):
        out_flow21 = os.path.join(flo_fwd_folder, os.path.basename(frame1_path))
        out_flow12 = os.path.join(flo_bck_folder, os.path.basename(frame2_path))

        f1 = _load_img(frame1_path, size)
        f2 = _load_img(frame2_path, size)

        flow21 = get_flow(f2, f1, model)
        flow12 = get_flow(f1, f2, model)

        if check_consistency and consistency_checker_bin and os.path.exists(consistency_checker_bin):
            f21_path = os.path.join(temp_dir, "flow21.flo")
            f12_path = os.path.join(temp_dir, "flow12.flo")
            write_flow(f21_path, flow21)
            write_flow(f12_path, flow12)
            import subprocess
            subprocess.run([consistency_checker_bin, f12_path, f21_path, f"{out_flow12}-12.txt"], check=False)
            subprocess.run([consistency_checker_bin, f21_path, f12_path, f"{out_flow21}-21.txt"], check=False)

        np.save(out_flow21, flow21)
        np.save(out_flow12, flow12)
