#!/bin/bash
set -e

SYNAPSE_EXAMPLE_APP_EXE="synapse-example-app"

systemctl stop "${SYNAPSE_EXAMPLE_APP_EXE}" || true
systemctl disable "${SYNAPSE_EXAMPLE_APP_EXE}" || true