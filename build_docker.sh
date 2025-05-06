#!/bin/bash
set -e

# Detect architecture
ARCH=$(uname -m)
if [[ "${ARCH}" == "arm64" || "${ARCH}" == "aarch64" ]]; then
    CONTAINER_TAG="arm64"
    PLATFORM="linux/arm64"
    DOCKERFILE_PATH="ops/docker/Dockerfile.arm64"
else
    CONTAINER_TAG="amd64"
    PLATFORM="linux/amd64"
    DOCKERFILE_PATH="ops/docker/Dockerfile"
fi

# Image names
SDK_IMAGE="synapse-example-app:latest-${CONTAINER_TAG}"

echo "Building for architecture: $ARCH"

# Build the SDK image
docker build -t $SDK_IMAGE -f "${DOCKERFILE_PATH}" .

echo "Successfully built $SDK_IMAGE"
