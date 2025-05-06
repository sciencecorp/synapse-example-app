#!/bin/bash
set -e

# Variables for the build
BUILD_ARM64_LINUX=0
CLEAN_BUILD=0
CONFIGURE_BUILD=0

function show_usage() {
    echo "Usage: $0 [options] [-- cmake_options]"
    echo "Options:"
    echo "  -a, --arm | Build for ARM64 Linux"
    echo "  -c, --clean | Clean build"
    echo "  -h, --help | Show this help message and exit"
    echo "  --configure | Force the configuration step to happen"
    echo "  -- | Pass all following arguments directly to CMake"
}

function clean_build() {
    echo "Cleaning build directories"
    rm -rf build
    rm -rf build-aarch64
}

function determine_triplet() {
    UNAME_M=$(uname -m)

    if [ "${UNAME_M}" = "x86_64" ] || [ "${UNAME_M}" = "amd64" ]; then
        TRIPLET="x64-linux"
    else
        # Everyone else should be building with this setup
        TRIPLET="arm64-linux-dynamic-release"
        BUILD_ARM64_LINUX=1
    fi

    # Override triplet if ARM64 Linux build is enabled, regardless of current architecture
    if [ ${BUILD_ARM64_LINUX} -eq 1 ]; then
        TRIPLET="arm64-linux-dynamic-release"
    fi
}

function set_threads() {
    if [ -n "${CI}" ]; then
        # go wild with threads on CI
        THREADS=$(nproc)
        return
    fi

    # Subtract 2 from available processors, similar to Makefile
    THREADS=$(nproc --ignore=2)
}

function is_configuration_up_to_date() {
    local build_dir="build"
    if [ ${BUILD_ARM64_LINUX} -eq 1 ]; then
        build_dir="build-aarch64"
    fi

    # Check if CMakeCache.txt exists and is newer than the CMakeLists.txt
    if [ -f "${build_dir}/CMakeCache.txt" ] && [ "${build_dir}/CMakeCache.txt" -nt "CMakeLists.txt" ]; then
        return 0 # Configuration is up-to-date
    else
        return 1 # Configuration needs to be updated
    fi
}

function configure_build() {
    echo "Configuring build for ${TRIPLET}"

    # Common CMake options
    EXTRA_OPTIONS=()

    # Add any extra options passed after --
    if [ ${#EXTRA_CMAKE_OPTIONS[@]} -gt 0 ]; then
        EXTRA_OPTIONS+=("${EXTRA_CMAKE_OPTIONS[@]}")
    fi

    if [ ${BUILD_ARM64_LINUX} -eq 1 ]; then
        cmake --preset=dynamic-aarch64 -DVCPKG_TARGET_TRIPLET=${TRIPLET} "${EXTRA_OPTIONS[@]}"
    else
        cmake --preset=dynamic -DVCPKG_TARGET_TRIPLET=${TRIPLET} "${EXTRA_OPTIONS[@]}"
    fi
}

function run_build() {
    echo "Running build with ${THREADS} threads..."
    if [ ${BUILD_ARM64_LINUX} -eq 1 ]; then
        cmake --build --preset=cross-release -j"${THREADS}"
    else
        cmake --build --preset=debug -j"${THREADS}"
    fi
}

# Array to store extra CMake options
EXTRA_CMAKE_OPTIONS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arm)
            BUILD_ARM64_LINUX=1
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        --configure)
            CONFIGURE_BUILD=1
            shift
            ;;
        --)
            # Collect all arguments after -- as extra CMake options
            shift
            while [[ $# -gt 0 ]]; do
                EXTRA_CMAKE_OPTIONS+=("$1")
                shift
            done
            break
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

determine_triplet
set_threads

if [ ${CLEAN_BUILD} -eq 1 ]; then
    clean_build
fi

if [ ${CONFIGURE_BUILD} -eq 1 ]; then
    configure_build
else
    # Check if we need to configure the build
    if is_configuration_up_to_date; then
        echo "Configuration up to date, skipping"
    else
        configure_build
    fi
fi

run_build
