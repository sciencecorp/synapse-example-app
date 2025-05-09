#!/bin/bash

SYNAPSE_APP_VERSION="0.1.0"
SYNAPSE_APP_EXE="synapse-example-app"

SCRIPT_DIR=$(dirname "$0")
SOURCE_DIR="${SCRIPT_DIR}/../../"
BUILD_DIR="${SOURCE_DIR}/build-aarch64/"

STAGING_DIR="/tmp/synapse-package"
mkdir -p ${STAGING_DIR}

# Binary install and setup
# TODO: Decide if there is a better place to put this
mkdir -p ${STAGING_DIR}/opt/scifi/bin
cp "${BUILD_DIR}/${SYNAPSE_APP_EXE}" "${STAGING_DIR}/opt/scifi/bin/"

# Launch script
mkdir -p ${STAGING_DIR}/opt/scifi/scripts
cp "${SCRIPT_DIR}/scripts/launch_synapse_app.sh" "${STAGING_DIR}/opt/scifi/scripts/"

# Systemd service install and setup
mkdir -p ${STAGING_DIR}/etc/systemd/system
cp "${SCRIPT_DIR}/systemd/synapse-example-app.service" "${STAGING_DIR}/etc/systemd/system/"

# App SDK
APP_SDK_DOCKER_DIR="/usr/lib/"
APP_SDK_LIB_TARGET_DIR="${STAGING_DIR}/opt/scifi/lib"
mkdir -p ${APP_SDK_LIB_TARGET_DIR}
find "${APP_SDK_DOCKER_DIR}" -name "libsynapse*.so*" -exec cp -av {} ${APP_SDK_LIB_TARGET_DIR}/ \;

fpm -s dir -t deb \
    -n "${SYNAPSE_APP_EXE}" \
    -f \
    -v "${SYNAPSE_APP_VERSION}" \
    -C ${STAGING_DIR} \
    --deb-no-default-config-files \
    --depends "systemd" \
    --vendor "Science Corporation" \
    --description "Synapse Example Application" \
    --architecture arm64 \
    --after-install "${SCRIPT_DIR}/scripts/postinstall.sh" \
    --before-remove "${SCRIPT_DIR}/scripts/preremove.sh" \
    --after-remove "${SCRIPT_DIR}/scripts/postremove.sh" \
    .
