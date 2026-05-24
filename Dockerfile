FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ca-certificates \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        build-essential pkg-config \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV DISCO_ROOT=/disco
WORKDIR ${DISCO_ROOT}

RUN git clone --depth 1 https://github.com/openai/CLIP.git CLIP \
 && git clone --depth 1 https://github.com/crowsonkb/guided-diffusion.git guided-diffusion \
 && git clone --depth 1 https://github.com/assafshocher/ResizeRight.git ResizeRight \
 && git clone --depth 1 https://github.com/MSFTserver/pytorch3d-lite.git pytorch3d-lite \
 && git clone --depth 1 https://github.com/isl-org/MiDaS.git MiDaS \
 && git clone --depth 1 https://github.com/alembics/disco-diffusion.git disco-diffusion \
 && git clone --depth 1 https://github.com/princeton-vl/RAFT.git RAFT \
 && mv MiDaS/utils.py MiDaS/midas_utils.py \
 && cp disco-diffusion/disco_xform_utils.py disco_xform_utils.py

RUN pip install --no-cache-dir -e guided-diffusion \
 && pip install --no-cache-dir -e CLIP

ENV PYTHONPATH=${DISCO_ROOT}:${DISCO_ROOT}/CLIP:${DISCO_ROOT}/guided-diffusion:${DISCO_ROOT}/ResizeRight:${DISCO_ROOT}/pytorch3d-lite:${DISCO_ROOT}/MiDaS:${DISCO_ROOT}/RAFT/core

ENV MODEL_DIR=/disco/models
RUN mkdir -p ${MODEL_DIR} RAFT/models

RUN wget -nv -O ${MODEL_DIR}/512x512_diffusion_uncond_finetune_008100.pt \
        "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt" \
 && wget -nv -O ${MODEL_DIR}/256x256_diffusion_uncond.pt \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt" \
 && wget -nv -O ${MODEL_DIR}/secondary_model_imagenet_2.pth \
        "https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth" \
 && wget -nv -O ${MODEL_DIR}/dpt_large-midas-2f21e586.pt \
        "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt"

RUN wget -nv -O RAFT/models/raft-things.pth \
        "https://github.com/e-dream-ai/gpu-container-discodiffusion/releases/download/v1.0/raft-things.pth"

RUN python -c "import clip; clip.load('ViT-B/32', download_root='/disco/models/clip'); clip.load('RN50', download_root='/disco/models/clip')"

RUN python -c "import lpips; lpips.LPIPS(net='vgg')"

WORKDIR /workspace
COPY src/ /workspace/src/
COPY entrypoint.sh /workspace/entrypoint.sh
COPY test_input.json /workspace/test_input.json
RUN chmod +x /workspace/entrypoint.sh

ENV DISCO_OUTPUT_DIR=/workspace/output \
    DISCO_MODEL_DIR=/disco/models \
    DISCO_RAFT_PATH=/disco/RAFT/models/raft-things.pth \
    CLIP_DOWNLOAD_ROOT=/disco/models/clip

RUN mkdir -p ${DISCO_OUTPUT_DIR}

CMD ["/workspace/entrypoint.sh"]
