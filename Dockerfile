FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# We need to be able to cross compile
RUN dpkg --add-architecture arm64 && \
    sed 's/^deb http/deb [arch=amd64] http/' -i '/etc/apt/sources.list' && \
    echo "deb [arch=arm64] http://ports.ubuntu.com/ focal main restricted" >> /etc/apt/sources.list.d/arm64-cross-compile.list && \
    echo "deb [arch=arm64] http://ports.ubuntu.com/ focal-updates main restricted" >> /etc/apt/sources.list.d/arm64-cross-compile.list

# Build depens
RUN apt-get update && apt-get install -y \
    autoconf \
    autoconf-archive \
    build-essential \
    gcc-10-aarch64-linux-gnu \
    g++-10-aarch64-linux-gnu \
    binutils-aarch64-linux-gnu \
    git \
    curl \
    wget \
    ca-certificates \
    gpg \
    unzip \
    tar \
    pkg-config \
    libssl-dev \
    libtool \
    zip \
    ninja-build \
    gosu \
    python3 \
    python3-setuptools \
    python3-jinja2 \
    libudev-dev:arm64 \
    libudev-dev \
    qemu-user-static \
    python3-pip \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Package depends
RUN apt-get update && apt-get install -y \
    ruby-full \
    ruby-dev \
    rubygems \
    build-essential

RUN ln -s /usr/bin/aarch64-linux-gnu-gcc-10 /usr/bin/aarch64-linux-gnu-gcc && \
    ln -s /usr/bin/aarch64-linux-gnu-g++-10 /usr/bin/aarch64-linux-gnu-g++

    # Install CMake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' \
    | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    kitware-archive-keyring \
    cmake=3.28.6-0kitware1ubuntu20.04.1 \
    cmake-data=3.28.6-0kitware1ubuntu20.04.1 \
    && rm -rf /var/lib/apt/lists/*

# Older versions of dotenv for older versions of ubuntu
RUN gem install dotenv -v 2.8.1
RUN gem install --no-document fpm

# Entry point for the container
CMD ["/bin/bash"]
