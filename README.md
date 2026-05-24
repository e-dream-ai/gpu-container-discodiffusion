# gpu-container-discodiffusion

RunPod serverless container for **Disco Diffusion v5.2 (warp)** — CLIP-guided
diffusion with RAFT optical-flow video stylization, originally by Sxela, Somnai,
gandamu, zippy, and the Disco Diffusion community.

## What it does

Three animation modes:

| `animation_mode` | Input | Output |
|---|---|---|
| `None` | text prompt | single image |
| `2D` | text prompt + keyframed transforms | mp4 |
| `3D` | text prompt + keyframed transforms + MiDaS depth | mp4 |
| `Video Input` | text prompt + source video | mp4 stylized via optical-flow warping |

## Build

```bash
./build.sh latest
docker push <your-registry>/gpu-container-discodiffusion:latest
```

The first build downloads ~4 GB of model weights (512px and 256px diffusion
checkpoints, secondary denoiser, MiDaS DPT-Large, RAFT, plus ViT-B/32 and RN50
CLIP encoders). Expect ~30 minutes and ~12 GB on the final image.

## Run locally

```bash
docker run --rm --gpus all \
    -v $(pwd)/output:/workspace/output \
    -e R2_ENDPOINT_URL=... \
    -e R2_ACCESS_KEY_ID=... \
    -e R2_SECRET_ACCESS_KEY=... \
    -e R2_BUCKET_NAME=... \
    gpu-container-discodiffusion:latest
```

To test without a RunPod queue, pass `test_input.json` via the runpod-cli or by
overriding the entrypoint:

```bash
docker run --rm --gpus all gpu-container-discodiffusion:latest \
    python -m src.handler --test_input "$(cat test_input.json)"
```

## Input schema

Top-level: `{"input": {"settings": { ... }}}`. Settings are a flat dict of any
field on `DiscoConfig` (see [`src/config.py`](src/config.py)). Highlights:

- `text_prompts`: `{"0": ["prompt string"], "50": ["different prompt at frame 50"]}`
- `width`, `height`: must be multiples of 64 (rounded down)
- `steps`: total diffusion steps (50 is fast, 250 is the original default)
- `clip_guidance_scale`: how hard CLIP pushes toward the prompt
- `cutn_batches`: gradient accumulation, higher = better coherence + slower
- `clip_vit_b32`, `clip_rn50`, ...: which CLIP encoders to ensemble
- `animation_mode`: `"None" | "2D" | "3D" | "Video Input"`
- For `Video Input`: provide either `video_init_path` (a local path) or
  `video_init_url` (will be downloaded). Set `flow_warp`, `flow_blend`,
  `check_consistency` to control the optical-flow stylization.

## Output

If R2 environment variables are set, the output `.mp4` (or `.png` for single
image) is uploaded and a presigned URL is returned. Otherwise the file is
base64-encoded in the response.

```json
{
  "status": "success",
  "video": "https://...",
  "seed": 1337,
  "frames": 60,
  "batch_num": 0
}
```

## Notes

- The diffusion model is the original Katherine Crowson 512×512 fine-tune.
  Outputs have a recognizable painterly/psychedelic aesthetic. For modern AI
  video, prefer LTX or Wan.
- A single 512×512 image at 250 steps takes ~10 minutes on an A100.
- Video Input mode requires RAFT optical-flow generation as a preprocessing
  step (~1 s/frame at 720p).
- The original Colab notebook (`IMPORTANTTTTTTTT.ipynb`) lives at the
  monorepo root for reference.
