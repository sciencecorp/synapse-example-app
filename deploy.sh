#!/bin/bash
set -e
PATH_TO_DEPLOY_DIR="deploy"

# TODO: Should we first package the app?
source "${PATH_TO_DEPLOY_DIR}/.venv/bin/activate"
python3 "${PATH_TO_DEPLOY_DIR}/deploy.py" "$1"
deactivate
