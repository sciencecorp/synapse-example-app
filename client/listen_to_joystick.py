#!/usr/bin/env python3
"""listen_to_joystick.py

A minimal client that connects to a Synapse device tap (default: "joystick_out"),
reads 2-element float tensors published by the FixedWeightDecoder example, and
prints the (x, y) values to stdout.
"""

import argparse
import struct
import sys
from typing import Tuple

import numpy as np
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Listen to a Synapse tap and dump joystick (x, y) values"
    )
    parser.add_argument(
        "--device-ip", required=True, help="IP address of the Synapse device"
    )
    parser.add_argument(
        "--tap-name",
        default="joystick_out",
        help="Name of the tap to connect to (default: joystick_out)",
    )
    return parser.parse_args()


def tensor_to_xy(tensor: Tensor) -> Tuple[float, float]:
    """Convert a 2-element float Tensor message to an (x, y) tuple."""
    # The FixedWeightDecoder publishes little-endian 32-bit floats
    data = tensor.data
    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        # Convert big-endian bytes to little-endian before np.frombuffer
        data = struct.unpack(f">{len(data) // 4}f", data)
        arr = np.array(data, dtype=np.float32)
    else:
        arr = np.frombuffer(data, dtype=np.float32)

    if arr.size < 2:
        raise ValueError(f"Expected at least 2 floats, got {arr.size}")
    return float(arr[0]), float(arr[1])


def main() -> None:
    args = parse_args()

    tap = Tap(args.device_ip)
    if not tap.connect(args.tap_name):
        print(
            f"Failed to connect to tap '{args.tap_name}' at {args.device_ip}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Connected to tap '{args.tap_name}' at {args.device_ip}. Press Ctrl-C to exit."
    )

    try:
        while True:
            raw = tap.read()
            if raw is None:
                continue  # no message yet
            tensor = Tensor()
            tensor.ParseFromString(raw)
            try:
                x, y = tensor_to_xy(tensor)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
                continue
            print(f"x = {x:.3f}\ty = {y:.3f}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        tap.disconnect()


if __name__ == "__main__":
    main()
