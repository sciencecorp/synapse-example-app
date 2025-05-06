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

# Image names
SDK_IMAGE="synapse-example-app:latest-${TAG_SUFFIX}"

echo "Building for architecture: $ARCH"

# Build the SDK image
docker build -t $SDK_IMAGE .

echo "Successfully built $SDK_IMAGE"
