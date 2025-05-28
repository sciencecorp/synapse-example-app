# syntax=docker/dockerfile:1

# Unified builder for both linux/arm64 (native) and linux/amd64 (cross-compile → arm64)
ARG TARGETPLATFORM
FROM ubuntu:20.04 AS base

ARG TARGETPLATFORM
ARG VCPKG_COMMIT=0f88ecb8528605f91980b90a2c5bad88e3cb565f

# -----------------------------------------------------------------------------
# Base environment variables
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    VCPKG_ROOT=/vcpkg \
    PATH="${PATH}:/vcpkg"

# -----------------------------------------------------------------------------
# APT sources – add arm64 repos when we are on an amd64 host (cross-compiling)
# -----------------------------------------------------------------------------
RUN set -eux; \
    if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        dpkg --add-architecture arm64; \
        # Mark the default entries as amd64-only (avoid downloading both arch indices)
        sed 's/^deb http/deb [arch=amd64] http/' -i /etc/apt/sources.list; \
        echo "deb [arch=arm64] http://ports.ubuntu.com/ focal main restricted" >  /etc/apt/sources.list.d/arm64-cross-compile.list; \
        echo "deb [arch=arm64] http://ports.ubuntu.com/ focal-updates main restricted" >> /etc/apt/sources.list.d/arm64-cross-compile.list; \
    fi

# -----------------------------------------------------------------------------
# Core build dependencies (split for native vs cross compile)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        apt-get install -y --no-install-recommends \
            autoconf autoconf-archive build-essential \
            gcc-10-aarch64-linux-gnu g++-10-aarch64-linux-gnu binutils-aarch64-linux-gnu \
            git curl wget ca-certificates gpg unzip tar pkg-config \
            libssl-dev libtool zip ninja-build gosu \
            python3 python3-setuptools python3-jinja2 python3-pip \
            libudev-dev:arm64 libudev-dev qemu-user-static; \
    else \
        apt-get install -y --no-install-recommends \
            autoconf autoconf-archive build-essential \
            gcc-10 g++-10 \
            git curl wget ca-certificates gpg unzip tar pkg-config \
            libssl-dev libtool zip ninja-build gosu \
            python3 python3-setuptools python3-jinja2 python3-pip \
            libudev-dev; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Ruby tool-chain for packaging helpers (common to both)
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ruby-full ruby-dev rubygems build-essential; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Compiler selection tweaks (native vs cross)
# -----------------------------------------------------------------------------
RUN set -eux; \
    if [ "${TARGETPLATFORM}" = "linux/amd64" ]; then \
        # Provide un-suffixed symlinks for the cross compiler to simplify build scripts
        ln -s /usr/bin/aarch64-linux-gnu-gcc-10  /usr/bin/aarch64-linux-gnu-gcc; \
        ln -s /usr/bin/aarch64-linux-gnu-g++-10 /usr/bin/aarch64-linux-gnu-g++; \
    else \
        # Make gcc-10 the system default on native arm64
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 100 --slave /usr/bin/g++ g++ /usr/bin/g++-10; \
    fi

# -----------------------------------------------------------------------------
# CMake 3.28 (identical for both platforms)
# -----------------------------------------------------------------------------
RUN set -eux; \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null; \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' > /etc/apt/sources.list.d/kitware.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends kitware-archive-keyring \
        cmake=3.28.6-0kitware1ubuntu20.04.1 cmake-data=3.28.6-0kitware1ubuntu20.04.1; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Install helpful Ruby gems (same for both)
# -----------------------------------------------------------------------------
RUN gem install dotenv -v 2.8.1 && gem install --no-document fpm

# -----------------------------------------------------------------------------
# vcpkg: clone, bootstrap, and install packages for arm64 target
# -----------------------------------------------------------------------------
ENV VCPKG_FORCE_SYSTEM_BINARIES=true
RUN git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}" && \
    cd "${VCPKG_ROOT}" && \
    git checkout "${VCPKG_COMMIT}" && \
    ./bootstrap-vcpkg.sh -disableMetrics

# copy project-specific ports and manifest before installing
COPY vcpkg.json "${VCPKG_ROOT}/vcpkg.json"
COPY external/sciencecorp/vcpkg "${VCPKG_ROOT}/external/sciencecorp/vcpkg"

RUN cd "${VCPKG_ROOT}" && \
    ./vcpkg install \
    --triplet arm64-linux-dynamic-release \
    --x-install-root "$PWD/build/host/vcpkg_installed" \
    --clean-after-build

# -----------------------------------------------------------------------------
# Install Synapse SDK from internal repository (same steps on both)
# -----------------------------------------------------------------------------
ARG SDK_VERSION=0.4.1
COPY keys/science-repo-public.asc /usr/share/keyrings/scifi-repo-science-public.asc
RUN set -eux; \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates; \
    echo "deb [signed-by=/usr/share/keyrings/scifi-repo-science-public.asc] https://pub-879bfa29e67b4cd6b0c78b0d4cc3aa59.r2.dev/scifi focal main" > /etc/apt/sources.list.d/repo-science.list; \
    apt-get update && apt-get install -y synapse-app-sdk="${SDK_VERSION}"; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Export environment variables used by CMake tool-chain
# -----------------------------------------------------------------------------
ENV VCPKG_INSTALLATION_ROOT="${VCPKG_ROOT}"
ENV CMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
ENV VCPKG_INSTALLED_DIR="${VCPKG_ROOT}/build/host/vcpkg_installed"

# -----------------------------------------------------------------------------
# Final workspace & entrypoint
# -----------------------------------------------------------------------------
WORKDIR /home/workspace
CMD ["/bin/bash"]
