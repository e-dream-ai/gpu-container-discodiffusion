from __future__ import annotations

import os
import sys
import tempfile
import traceback
import urllib.request
from typing import Any

import runpod

for path in ("/disco", "/disco/CLIP", "/disco/guided-diffusion",
             "/disco/ResizeRight", "/disco/pytorch3d-lite",
             "/disco/MiDaS", "/disco/RAFT/core"):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from edream_sdk.client import create_edream_client
from edream_sdk.types.file_upload_types import FileType

BACKEND_URL = os.environ.get("BACKEND_URL")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY")
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

edream_client = None
if BACKEND_URL and BACKEND_API_KEY:
    edream_client = create_edream_client(backend_url=BACKEND_URL, api_key=BACKEND_API_KEY)


def _download_url(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    fd, path = tempfile.mkstemp(prefix="disco_input_", suffix=suffix)
    os.close(fd)
    if edream_client:
        edream_client.file_client.download_file(url, path)
    else:
        urllib.request.urlretrieve(url, path)
    return path


def _resolve_video_input(cfg_dict: dict[str, Any]) -> None:
    source_uuid = cfg_dict.pop("source_dream_uuid", None)
    video_url = cfg_dict.pop("video_init_url", None)

    if source_uuid:
        if not edream_client:
            raise RuntimeError("BACKEND_URL/BACKEND_API_KEY required for source_dream_uuid")
        dream = edream_client.get_dream(uuid=source_uuid)
        if not dream or not dream.get("original_video"):
            raise ValueError(f"Dream {source_uuid} has no original_video")
        cfg_dict["video_init_path"] = _download_url(dream["original_video"])
    elif video_url:
        cfg_dict["video_init_path"] = _download_url(video_url)


def _resolve_init_image(cfg_dict: dict[str, Any]) -> None:
    init = cfg_dict.get("init_image", "")
    if init and init.startswith(("http://", "https://")):
        cfg_dict["init_image"] = _download_url(init)


def _upload_output(output_path: str, batch_name: str) -> dict:
    if not edream_client:
        raise RuntimeError("BACKEND_URL/BACKEND_API_KEY required for output upload")
    dream = edream_client.file_client.upload_file(
        output_path,
        FileType.DREAM,
        {"name": batch_name},
    )
    dream_uuid = dream.get("uuid")
    dream_detail = edream_client.get_dream(uuid=dream_uuid) if dream_uuid else {}
    r2_url = (dream_detail.get("original_video") or dream_detail.get("video")) if dream_detail else None
    return {
        "dream_uuid": dream_uuid,
        "r2_url": r2_url,
    }


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    if not isinstance(job_input, dict):
        return {"error": "Input must be a JSON object"}

    settings = job_input.get("settings") if "settings" in job_input else job_input
    if not isinstance(settings, dict):
        return {"error": "Missing 'settings' object"}

    settings = dict(settings)
    try:
        _resolve_video_input(settings)
        _resolve_init_image(settings)
        from src.config import DiscoConfig
        cfg = DiscoConfig.from_dict(settings)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Bad config: {exc}"}

    try:
        from src.pipeline import run_job
        result = run_job(cfg)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"run_job failed: {exc}"}

    output_path = result.get("output_path")
    if not output_path or not os.path.exists(output_path):
        return {"error": "No output produced"}

    try:
        upload = _upload_output(output_path, cfg.batch_name)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Upload failed: {exc}"}

    return {
        "status": "success",
        "dream_uuid": upload["dream_uuid"],
        "r2_url": upload["r2_url"],
        "seed": result.get("seed"),
        "frames": result.get("frames"),
        "batch_num": result.get("batch_num"),
        "refresh_worker": REFRESH_WORKER,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
