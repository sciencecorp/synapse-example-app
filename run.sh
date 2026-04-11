#!/usr/bin/env bash
# Usage:  ./run.sh <device-ip>
#         ./run.sh              (demo mode with arrow keys)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_IP="${1:-}"
python3 -c "import pygame"  2>/dev/null || pip install pygame
python3 -c "import numpy"   2>/dev/null || pip install numpy
python3 -c "import synapse" 2>/dev/null || pip install --pre science-synapse
if [ -n "$DEVICE_IP" ]; then
  python3 "${SCRIPT_DIR}/game/bci_game.py" --device-ip "$DEVICE_IP"
else
  echo "No device IP - launching DEMO MODE (arrow keys)"
  python3 "${SCRIPT_DIR}/game/bci_game.py"
fi
