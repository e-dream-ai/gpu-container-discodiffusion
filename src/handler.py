from __future__ import annotations

import os
import sys
import tempfile
import traceback
import uuid
from typing import Any

import boto3
import requests
import runpod
from botocore.config import Config as BotoConfig

for path in ("/disco", "/disco/CLIP", "/disco/guided-diffusion",
             "/disco/ResizeRight", "/disco/pytorch3d-lite",
             "/disco/MiDaS", "/disco/RAFT/core"):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

BACKEND_URL = os.environ.get("BACKEND_URL")
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY")
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

_edream_client = None


def _get_edream_client():
    global _edream_client
    if _edream_client is None and BACKEND_URL and BACKEND_API_KEY:
        from edream_sdk.client import create_edream_client
        _edream_client = create_edream_client(backend_url=BACKEND_URL, api_key=BACKEND_API_KEY)
    return _edream_client


def _upload_to_r2(job_id: str, file_path: str) -> str:
    bucket = os.environ["R2_BUCKET_NAME"]
    endpoint_url = os.environ["R2_ENDPOINT_URL"]
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    upload_dir = os.environ.get("R2_UPLOAD_DIRECTORY", "video-outputs").strip("/")
    expires_in = int(os.environ.get("R2_PRESIGNED_EXPIRY", "86400"))

    ext = os.path.splitext(file_path)[1].lower() or ".png"
    object_key = f"{upload_dir}/{job_id}{ext}" if upload_dir else f"{job_id}{ext}"
    content_types = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg"}

    s3 = boto3.client("s3", endpoint_url=endpoint_url,
                      aws_access_key_id=access_key, aws_secret_access_key=secret_key,
                      region_name="auto", config=BotoConfig(s3={"addressing_style": "path"}))

    with open(file_path, "rb") as f:
        s3.upload_fileobj(f, bucket, object_key,
                          ExtraArgs={"ContentType": content_types.get(ext, "application/octet-stream")})

    try:
        return s3.generate_presigned_url("get_object",
                                         Params={"Bucket": bucket, "Key": object_key},
                                         ExpiresIn=expires_in)
    except Exception:
        return f"{endpoint_url}/{bucket}/{object_key}"


def _download_url(url: str) -> str:
    suffix = os.path.splitext(url.split("?")[0])[1] or ".mp4"
    fd, path = tempfile.mkstemp(prefix="disco_input_", suffix=suffix)
    os.close(fd)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def _resolve_video_input(cfg_dict: dict[str, Any]) -> None:
    source_uuid = cfg_dict.pop("source_dream_uuid", None)
    video_url = cfg_dict.pop("video_init_url", None)

    if source_uuid:
        client = _get_edream_client()
        if not client:
            raise RuntimeError("BACKEND_URL/BACKEND_API_KEY required to resolve source_dream_uuid")
        dream = client.get_dream(uuid=source_uuid)
        if not dream or not dream.get("original_video"):
            raise ValueError(f"Dream {source_uuid} has no original_video")
        cfg_dict["video_init_path"] = _download_url(dream["original_video"])
    elif video_url:
        cfg_dict["video_init_path"] = _download_url(video_url)


def _resolve_init_image(cfg_dict: dict[str, Any]) -> None:
    init = cfg_dict.get("init_image", "")
    if init and init.startswith(("http://", "https://")):
        cfg_dict["init_image"] = _download_url(init)


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
        download_url = _upload_to_r2(str(job.get("id") or uuid.uuid4()), output_path)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"R2 upload failed: {exc}"}

    return {
        "status": "success",
        "download_url": download_url,
        "seed": result.get("seed"),
        "frames": result.get("frames"),
        "batch_num": result.get("batch_num"),
        "refresh_worker": REFRESH_WORKER,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
