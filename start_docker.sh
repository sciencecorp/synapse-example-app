#!/bin/bash
set -e

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  TAG_SUFFIX="amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  TAG_SUFFIX="arm64"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

# Image name
IMAGE="synapse-example-app:latest-${TAG_SUFFIX}"

# Check if image exists
if ! docker image inspect $IMAGE >/dev/null 2>&1; then
  echo "Image $IMAGE not found. Please run build_docker.sh first."
  exit 1
fi

echo "Starting container for architecture: $ARCH"

# Run the container with appropriate mounts
# Adjust volume mappings as needed for your project
docker run -it \
  --rm \
  -v "$(pwd):/workspace" \
  $IMAGE
