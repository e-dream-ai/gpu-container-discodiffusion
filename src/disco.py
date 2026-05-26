from __future__ import annotations

import gc
import io
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Optional

import cv2
import lpips
import numpy as np
import pandas as pd
import PIL
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=UserWarning)

from CLIP import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import NormalizeImage, PrepareForNet, Resize
from resize_right import resize

import py3d_tools as p3dT
import disco_xform_utils as dxf

from . import warp as warp_mod


MODEL_DIR = os.environ.get("DISCO_MODEL_DIR", "/disco/models")
CLIP_DOWNLOAD_ROOT = os.environ.get("CLIP_DOWNLOAD_ROOT", os.path.join(MODEL_DIR, "clip"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = DEVICE

if DEVICE.type == "cuda":
    cap = torch.cuda.get_device_capability(DEVICE)
    if cap == (8, 0) or cap >= (9, 0):
        torch.backends.cudnn.enabled = False


@dataclass
class Runtime:
    args: SimpleNamespace
    cfg: object
    diffusion: object
    model: nn.Module
    secondary_model: Optional[nn.Module]
    clip_models: list
    lpips_model: nn.Module
    normalize: T.Normalize
    batch_folder: str = ""
    partial_folder: str = ""
    video_frames_folder: str = ""
    flo_folder: str = ""


_RT: Optional[Runtime] = None
args: Optional[SimpleNamespace] = None


def set_runtime(rt: Runtime) -> None:
    global _RT, args
    _RT = rt
    args = rt.args


_MIDAS_MODELS = {
    "midas_v21_small": "midas_v21_small-70d6b9c8.pt",
    "midas_v21": "midas_v21-f6b98070.pt",
    "dpt_large": "dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": "dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_nyu": "dpt_hybrid_nyu-2ce69ec7.pt",
}


def init_midas_depth_model(midas_model_type="dpt_large", optimize=True):
    midas_model_path = os.path.join(MODEL_DIR, _MIDAS_MODELS[midas_model_type])
    print(f"Initializing MiDaS '{midas_model_type}' depth model...")

    if midas_model_type == "dpt_large":
        midas_model = DPTDepthModel(path=midas_model_path, backbone="vitl16_384", non_negative=True)
        net_w, net_h, resize_mode = 384, 384, "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type in ("dpt_hybrid", "dpt_hybrid_nyu"):
        midas_model = DPTDepthModel(path=midas_model_path, backbone="vitb_rn50_384", non_negative=True)
        net_w, net_h, resize_mode = 384, 384, "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif midas_model_type == "midas_v21":
        midas_model = MidasNet(midas_model_path, non_negative=True)
        net_w, net_h, resize_mode = 384, 384, "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    elif midas_model_type == "midas_v21_small":
        midas_model = MidasNet_small(midas_model_path, features=64, backbone="efficientnet_lite3",
                                     exportable=True, non_negative=True, blocks={"expand": True})
        net_w, net_h, resize_mode = 256, 256, "upper_bound"
        normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise ValueError(f"Unknown midas_model_type: {midas_model_type}")

    midas_transform = T.Compose([
        Resize(net_w, net_h, resize_target=None, keep_aspect_ratio=True,
               ensure_multiple_of=32, resize_method=resize_mode,
               image_interpolation_method=cv2.INTER_CUBIC),
        normalization,
        PrepareForNet(),
    ])
    midas_model.eval()
    if optimize and DEVICE.type == "cuda":
        midas_model = midas_model.to(memory_format=torch.channels_last).half()
    midas_model.to(DEVICE)
    return midas_model, midas_transform, net_w, net_h, resize_mode, normalization


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=None):
    device = device or DEVICE
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width, oct_height = width, height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves, width, height, grayscale, side_x, side_y):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert("RGB")
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())
    return ImageOps.autocontrast(out)


def regen_perlin(perlin_mode_, batch_size_, side_x, side_y):
    if perlin_mode_ == "color":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_x, side_y)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_x, side_y)
    elif perlin_mode_ == "gray":
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_x, side_y)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_x, side_y)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_x, side_y)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_x, side_y)
    init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    del init2
    return init.expand(batch_size_, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith(("http://", "https://")):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


def parse_prompt(prompt):
    if prompt.startswith(("http://", "https://")):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
    input = input.reshape([n * c, 1, h, w])
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])
    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1).normal_(mean=0.8, std=0.3)
                           .clip(float(self.cut_size / max_size), 1.0))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout
        return torch.cat(cutouts, dim=0)


padargs = {}


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        mode = args.animation_mode
        if mode == "None":
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif mode == "Video Input":
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            ])
        else:
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.4),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.3),
            ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            ((sideY - max_size) // 2, (sideY - max_size) // 2,
             (sideX - max_size) // 2, (sideX - max_size) // 2),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        if not _RT.cfg.skip_augs:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True))


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down, ConvBlock(cs[0], cs[1]), ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down, ConvBlock(cs[1], cs[2]), ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down, ConvBlock(cs[2], cs[3]), ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down, ConvBlock(cs[3], cs[4]), ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down, ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]), ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]), self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]), ConvBlock(cs[4], cs[3]), self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]), ConvBlock(cs[3], cs[2]), self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]), ConvBlock(cs[2], cs[1]), self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]), ConvBlock(cs[1], cs[0]), self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


def parse_key_frames(string, prompt_parser=None):
    import re
    pattern = r"((?P<frame>[0-9]+):[\s]*[\(](?P<param>[\S\s]*?)[\)])"
    frames = {}
    for match in re.finditer(pattern, string):
        frame = int(match.groupdict()["frame"])
        param = match.groupdict()["param"]
        frames[frame] = prompt_parser(param) if prompt_parser else param
    if not frames and string:
        raise RuntimeError("Key Frame string not correctly formatted")
    return frames


def get_inbetweens(key_frames, max_frames, interp_spline, integer=False):
    series = pd.Series([np.nan] * max_frames)
    for i, value in key_frames.items():
        series[i] = value
    series = series.astype(float)
    method = interp_spline
    if method == "Cubic" and len(key_frames.items()) <= 3:
        method = "Quadratic"
    if method == "Quadratic" and len(key_frames.items()) <= 2:
        method = "Linear"
    series[0] = series[series.first_valid_index()]
    series[max_frames - 1] = series[series.last_valid_index()]
    series = series.interpolate(method=method.lower(), limit_direction="both")
    return series.astype(int) if integer else series


def split_prompts(prompts, max_frames):
    series = pd.Series([np.nan] * max_frames)
    for i, prompt in prompts.items():
        series[i] = prompt
    return series.ffill().bfill()


def safe_keyframe_series(expr, max_frames, interp_spline):
    expr = str(expr)
    try:
        return get_inbetweens(parse_key_frames(expr), max_frames, interp_spline)
    except RuntimeError:
        return get_inbetweens(parse_key_frames(f"0: ({expr})"), max_frames, interp_spline)


TRANSLATION_SCALE = 1.0 / 200.0


def do_3d_step(img_filepath, frame_num, midas_model, midas_transform):
    if args.key_frames:
        translation_x = args.translation_x_series[frame_num]
        translation_y = args.translation_y_series[frame_num]
        translation_z = args.translation_z_series[frame_num]
        rotation_3d_x = args.rotation_3d_x_series[frame_num]
        rotation_3d_y = args.rotation_3d_y_series[frame_num]
        rotation_3d_z = args.rotation_3d_z_series[frame_num]
    translate_xyz = [-translation_x * TRANSLATION_SCALE,
                     translation_y * TRANSLATION_SCALE,
                     -translation_z * TRANSLATION_SCALE]
    rotate_xyz = [math.radians(rotation_3d_x), math.radians(rotation_3d_y), math.radians(rotation_3d_z)]
    rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
    return dxf.transform_image_3d(
        img_filepath, midas_model, midas_transform, DEVICE,
        rot_mat, translate_xyz, args.near_plane, args.far_plane,
        args.fov, padding_mode=args.padding_mode,
        sampling_mode=args.sampling_mode, midas_weight=args.midas_weight,
    )


def generate_eye_views(trans_scale, batch_folder, filename, frame_num, midas_model, midas_transform):
    for i in range(2):
        theta = _RT.cfg.vr_eye_angle * (math.pi / 180)
        ray_origin = math.cos(theta) * _RT.cfg.vr_ipd / 2 * (-1.0 if i == 0 else 1.0)
        ray_rotation = theta if i == 0 else -theta
        translate_xyz = [-ray_origin * trans_scale, 0, 0]
        rotate_xyz = [0, ray_rotation, 0]
        rot_mat = p3dT.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)
        transformed = dxf.transform_image_3d(
            f"{batch_folder}/{filename}", midas_model, midas_transform, DEVICE,
            rot_mat, translate_xyz, args.near_plane, args.far_plane,
            args.fov, padding_mode=args.padding_mode,
            sampling_mode=args.sampling_mode, midas_weight=args.midas_weight, spherical=True,
        )
        suffix = "_l" if i == 0 else "_r"
        transformed.save(f"{batch_folder}/frame_{frame_num-1:04}{suffix}.png")


def do_run():
    cfg = _RT.cfg
    rt = _RT
    diffusion = rt.diffusion
    model = rt.model
    secondary_model = rt.secondary_model
    clip_models = rt.clip_models
    lpips_model = rt.lpips_model
    normalize = rt.normalize
    batch_folder = rt.batch_folder
    partial_folder = rt.partial_folder
    video_frames_folder = rt.video_frames_folder
    flo_folder = rt.flo_folder

    seed = args.seed
    batch_size = 1
    side_x, side_y = args.side_x, args.side_y
    skip_steps = args.skip_steps
    init_image = args.init_image if args.init_image else None

    midas_model = midas_transform = None
    if args.animation_mode == "3D" and args.midas_weight > 0.0:
        midas_model, midas_transform, *_ = init_midas_depth_model(args.midas_depth_model)

    for frame_num in range(args.start_frame, args.max_frames):
        if args.animation_mode != "None":
            print(f"Frame {frame_num}/{args.max_frames}")

        if args.animation_mode != "Video Input":
            init_image = args.init_image if args.init_image else None
            init_scale = args.init_scale
            skip_steps = args.skip_steps

        if args.animation_mode == "2D":
            if args.key_frames:
                angle = args.angle_series[frame_num]
                zoom = args.zoom_series[frame_num]
                translation_x = args.translation_x_series[frame_num]
                translation_y = args.translation_y_series[frame_num]
            if frame_num > 0:
                seed += 1
                img_0 = cv2.imread("prevFrame.png")
                center = (img_0.shape[1] // 2, img_0.shape[0] // 2)
                trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
                rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
                trans_mat = np.vstack([trans_mat, [0, 0, 1]])
                rot_mat = np.vstack([rot_mat, [0, 0, 1]])
                transformation_matrix = np.matmul(rot_mat, trans_mat)
                img_0 = cv2.warpPerspective(img_0, transformation_matrix,
                                            (img_0.shape[1], img_0.shape[0]),
                                            borderMode=cv2.BORDER_WRAP)
                cv2.imwrite("prevFrameScaled.png", img_0)
                init_image = "prevFrameScaled.png"
                init_scale = args.frames_scale
                skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "3D":
            if frame_num > 0:
                seed += 1
                img_filepath = "prevFrame.png"
                next_step_pil = do_3d_step(img_filepath, frame_num, midas_model, midas_transform)
                next_step_pil.save("prevFrameScaled.png")

                if cfg.turbo_mode:
                    if frame_num == cfg.turbo_preroll:
                        next_step_pil.save("oldFrameScaled.png")
                    elif frame_num > cfg.turbo_preroll:
                        old_frame = do_3d_step("oldFrameScaled.png", frame_num, midas_model, midas_transform)
                        old_frame.save("oldFrameScaled.png")
                        if frame_num % int(cfg.turbo_steps) != 0:
                            filename = f"{args.batch_name}({args.batchNum})_{frame_num:04}.png"
                            blend_factor = ((frame_num % int(cfg.turbo_steps)) + 1) / int(cfg.turbo_steps)
                            new_warped = cv2.imread("prevFrameScaled.png")
                            old_warped = cv2.imread("oldFrameScaled.png")
                            blended = cv2.addWeighted(new_warped, blend_factor, old_warped, 1 - blend_factor, 0.0)
                            cv2.imwrite(f"{batch_folder}/{filename}", blended)
                            next_step_pil.save(img_filepath)
                            continue
                        else:
                            old_warped = cv2.imread("prevFrameScaled.png")
                            cv2.imwrite("oldFrameScaled.png", old_warped)

                init_image = "prevFrameScaled.png"
                init_scale = args.frames_scale
                skip_steps = args.calc_frames_skip_steps

        if args.animation_mode == "Video Input":
            init_scale = args.frames_scale
            skip_steps = args.calc_frames_skip_steps
            if not cfg.video_init_seed_continuity:
                seed += 1
            if cfg.flow_warp:
                if frame_num == 0:
                    skip_steps = args.skip_steps
                    init_image = f"{video_frames_folder}/{frame_num+1:04}.jpg"
                else:
                    prev = PIL.Image.open(f"{batch_folder}/{args.batch_name}({args.batchNum})_{frame_num-1:04}.png")
                    frame1_name = f"{frame_num:04}.jpg"
                    frame2 = PIL.Image.open(f"{video_frames_folder}/{frame_num+1:04}.jpg")
                    flo_path = f"{flo_folder}/{frame1_name}.npy"
                    weights_path = None
                    if cfg.check_consistency:
                        weights_path = f"{flo_folder}/{frame1_name}-21.txt"
                    init_image = "warped.png"
                    warp_mod.warp(prev, frame2, flo_path, blend=cfg.flow_blend,
                                  weights_path=weights_path).save(init_image)
            else:
                init_image = f"{video_frames_folder}/{frame_num+1:04}.jpg"

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
        else:
            frame_prompt = []

        if args.image_prompts_series is not None and frame_num >= len(args.image_prompts_series):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        print(f"Frame {frame_num} Prompt: {frame_prompt}")

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {"clip_model": clip_model, "target_embeds": [], "make_cutouts": None, "weights": []}

            for prompt in frame_prompt:
                _, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                if args.fuzzy_prompt:
                    for _ in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(0, 1))
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(
                    clip_model.visual.input_resolution, cutn, skip_augs=cfg.skip_augs)
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert("RGB")
                    img = TF.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
                    batch = model_stat["make_cutouts"](TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if cfg.fuzzy_prompt:
                        for _ in range(25):
                            model_stat["target_embeds"].append(
                                (embed + torch.randn(embed.shape).cuda() * cfg.rand_mag).clamp(0, 1))
                            model_stat["weights"].extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError("The weights must not sum to 0.")
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert("RGB")
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if cfg.perlin_init:
            init = regen_perlin(cfg.perlin_mode, batch_size, side_x, side_y)

        cur_t = None

        def cond_fn(x, t, y=None):
            nonlocal cur_t
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if cfg.use_secondary_model:
                    alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    sigma = torch.tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)

                for model_stat in model_stats:
                    for _ in range(args.cutn_batches):
                        t_int = int(t.item()) + 1
                        try:
                            input_resolution = model_stat["clip_model"].visual.input_resolution
                        except Exception:
                            input_resolution = 224
                        cuts = MakeCutoutsDango(
                            input_resolution,
                            Overview=args.cut_overview[1000 - t_int],
                            InnerCrop=args.cut_innercut[1000 - t_int],
                            IC_Size_Pow=args.cut_ic_pow,
                            IC_Grey_P=args.cut_icgray_p[1000 - t_int],
                        )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0))
                        dists = dists.view([args.cut_overview[1000 - t_int] + args.cut_innercut[1000 - t_int], n, -1])
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(losses.sum().item())
                        x_in_grad += torch.autograd.grad(
                            losses.sum() * cfg.clip_guidance_scale, x_in)[0] / args.cutn_batches

                tv_losses = tv_loss(x_in)
                if cfg.use_secondary_model:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out["pred_xstart"])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = (tv_losses.sum() * cfg.tv_scale
                        + range_losses.sum() * cfg.range_scale
                        + sat_losses.sum() * cfg.sat_scale)
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if not torch.isnan(x_in_grad).any():
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and not x_is_NaN:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=args.clamp_max) / magnitude
            return grad

        sample_fn = (diffusion.ddim_sample_loop_progressive
                     if args.diffusion_sampling_mode == "ddim"
                     else diffusion.plms_sample_loop_progressive)

        for i in range(args.n_batches):
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if cfg.perlin_init:
                init = regen_perlin(cfg.perlin_mode, batch_size, side_x, side_y)

            shape = (batch_size, 3, args.side_y, args.side_x)
            sampler_kwargs = dict(
                clip_denoised=cfg.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=cfg.randomize_class,
            )
            if args.diffusion_sampling_mode == "ddim":
                samples = sample_fn(model, shape, eta=cfg.eta, **sampler_kwargs)
            else:
                samples = sample_fn(model, shape, order=2, **sampler_kwargs)

            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediate_step = False
                if args.steps_per_checkpoint is not None:
                    if j % args.steps_per_checkpoint == 0 and j > 0:
                        intermediate_step = True
                elif j in args.intermediate_saves:
                    intermediate_step = True

                if j % args.display_rate == 0 or cur_t == -1 or intermediate_step:
                    for image_tensor in sample["pred_xstart"]:
                        save_num = f"{frame_num:04}" if args.animation_mode != "None" else i
                        if cur_t == -1 and args.intermediates_in_subfolder:
                            filename = f"{args.batch_name}({args.batchNum})_{save_num}.png"
                        elif args.steps_per_checkpoint is not None:
                            percent = math.ceil(j / total_steps * 100)
                            filename = f"{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png"
                        else:
                            filename = f"{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png"

                        image = TF.to_pil_image(image_tensor.add(1).div(2).clamp(0, 1))
                        image.save("progress.png")

                        if args.steps_per_checkpoint is not None and j % args.steps_per_checkpoint == 0 and j > 0:
                            target = partial_folder if args.intermediates_in_subfolder else batch_folder
                            image.save(f"{target}/{filename}")
                        elif j in (args.intermediate_saves or []):
                            target = partial_folder if args.intermediates_in_subfolder else batch_folder
                            image.save(f"{target}/{filename}")

                        if cur_t == -1:
                            if frame_num == 0:
                                save_settings()
                            if args.animation_mode != "None":
                                image.save("prevFrame.png")
                            image.save(f"{batch_folder}/{filename}")
                            if args.animation_mode == "3D":
                                if cfg.turbo_mode and frame_num > 0:
                                    blend_factor = 1 / int(cfg.turbo_steps)
                                    new_frame = cv2.imread("prevFrame.png")
                                    prev_warped = cv2.imread("prevFrameScaled.png")
                                    blended = cv2.addWeighted(new_frame, blend_factor, prev_warped, 1 - blend_factor, 0.0)
                                    cv2.imwrite(f"{batch_folder}/{filename}", blended)
                                if cfg.vr_mode:
                                    generate_eye_views(TRANSLATION_SCALE, batch_folder, filename,
                                                       frame_num, midas_model, midas_transform)


def save_settings():
    cfg = _RT.cfg
    settings = {**cfg.to_dict(), "seed": args.seed, "batchNum": args.batchNum}
    path = f"{_RT.batch_folder}/{cfg.batch_name}({args.batchNum})_settings.txt"
    with open(path, "w") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2, default=str)


def build_model_config(diffusion_model_name: str, steps: int, use_checkpoint: bool, use_fp16: bool) -> dict:
    base_cfg = model_and_diffusion_defaults()
    update = {
        "attention_resolutions": "32, 16, 8",
        "class_cond": False,
        "diffusion_steps": (1000 // steps) * steps if steps < 1000 else steps,
        "rescale_timesteps": True,
        "timestep_respacing": f"ddim{steps}" if steps else "ddim250",
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "use_checkpoint": use_checkpoint,
        "use_fp16": use_fp16,
        "use_scale_shift_norm": True,
    }
    if diffusion_model_name == "512x512_diffusion_uncond_finetune_008100":
        update["image_size"] = 512
    elif diffusion_model_name == "256x256_diffusion_uncond":
        update["image_size"] = 256
    else:
        raise ValueError(f"Unknown diffusion model: {diffusion_model_name}")
    base_cfg.update(update)
    return base_cfg


def load_clip_models(cfg) -> list:
    flags = [
        ("ViT-B/32", cfg.clip_vit_b32),
        ("ViT-B/16", cfg.clip_vit_b16),
        ("ViT-L/14", cfg.clip_vit_l14),
        ("ViT-L/14@336px", cfg.clip_vit_l14_336),
        ("RN50", cfg.clip_rn50),
        ("RN101", cfg.clip_rn101),
        ("RN50x4", cfg.clip_rn50x4),
        ("RN50x16", cfg.clip_rn50x16),
        ("RN50x64", cfg.clip_rn50x64),
    ]
    loaded = []
    for name, enabled in flags:
        if enabled:
            print(f"Loading CLIP {name}")
            m, _ = clip.load(name, jit=False, download_root=CLIP_DOWNLOAD_ROOT)
            loaded.append(m.eval().requires_grad_(False).to(device))
    if not loaded:
        raise RuntimeError("At least one CLIP model must be enabled.")
    return loaded


def load_diffusion_model(cfg):
    model_config = build_model_config(cfg.diffusion_model, cfg.steps, cfg.use_checkpoint, cfg.use_fp16)
    model, diffusion = create_model_and_diffusion(**model_config)
    weights_path = os.path.join(MODEL_DIR, f"{cfg.diffusion_model}.pt")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def load_secondary_model():
    sm = SecondaryDiffusionImageNet2()
    sm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "secondary_model_imagenet_2.pth"), map_location="cpu"))
    return sm.eval().requires_grad_(False).to(device)


def load_lpips():
    return lpips.LPIPS(net="vgg").to(device)


CLIP_NORMALIZE = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)
