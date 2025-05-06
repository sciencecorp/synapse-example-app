#!/bin/bash
set -e

# TODO: There has to be a better way to make this global
SYNAPSE_EXAMPLE_APP_EXE="synapse-example-app"

# Set up and reload udev rules
udevadm control --reload-rules
udevadm trigger

# TODO: Remove this
chown root:root /opt/scifi/bin/"${SYNAPSE_EXAMPLE_APP_EXE}"
chmod 755 /opt/scifi/bin/"${SYNAPSE_EXAMPLE_APP_EXE}"

# Reload and start the service
systemctl daemon-reload
systemctl enable "${SYNAPSE_EXAMPLE_APP_EXE}"
