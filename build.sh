#!/usr/bin/env bash
set -eox pipefail

TAG=${1:-latest}
IMAGE_NAME=${IMAGE_NAME:-gpu-container-discodiffusion}
FULL_TAG="${IMAGE_NAME}:${TAG}"

DOCKER_BUILDKIT=1 docker build \
    --platform linux/amd64 \
    -t "${FULL_TAG}" \
    .

echo ""
echo "Built ${FULL_TAG}"
echo "Push:  docker push ${FULL_TAG}"
