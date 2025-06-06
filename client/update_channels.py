#!/usr/bin/env python3
"""update_channels.py

A client that connects to a Synapse device and sends cursor channel updates
to the "set_cursor_channels" tap using ListValue messages with 4 integers.
"""

import argparse
import sys
import time
from typing import List

from google.protobuf.struct_pb2 import ListValue, Value
from synapse.client.taps import Tap


def parse_args() -> argparse.Namespace:
    """CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Send cursor channel updates to a Synapse device"
    )
    parser.add_argument(
        "--device-ip", required=True, help="IP address of the Synapse device"
    )
    parser.add_argument(
        "--channels",
        nargs=4,
        type=int,
        required=True,
        help="Four channel numbers to set (e.g., --channels 0 1 2 3)",
    )
    parser.add_argument(
        "--tap-name",
        default="set_cursor_channels",
        help="Name of the tap to connect to (default: set_cursor_channels)",
    )
    return parser.parse_args()


def create_channel_list_value(channels: List[int]) -> ListValue:
    """Create a ListValue message with the channel numbers.

    Args:
        channels: List of 4 channel numbers

    Returns:
        ListValue: Protobuf message containing the channel numbers
    """
    if len(channels) != 4:
        raise ValueError(f"Expected exactly 4 channels, got {len(channels)}")

    # Validate channel range (based on the C++ code validation)
    for channel in channels:
        if channel < 0 or channel >= 32:
            raise ValueError(f"Channel {channel} is out of range (0-31)")

    list_value = ListValue()
    for channel in channels:
        value = Value()
        value.number_value = float(channel)  # ListValue uses number_value for numbers
        list_value.values.append(value)

    return list_value


def main() -> None:
    args = parse_args()

    print(f"Connecting to Synapse device at {args.device_ip}")
    print(f"Setting cursor channels to: {args.channels}")

    tap = Tap(args.device_ip)
    try:
        # Connect to the set_cursor_channels tap
        if not tap.connect(args.tap_name):
            print(f"Failed to connect to tap '{args.tap_name}' at {args.device_ip}", file=sys.stderr)
            sys.exit(1)

        print(f"Connected to tap '{args.tap_name}' at {args.device_ip}")

        # Create the ListValue message with the channel numbers
        try:
            list_value = create_channel_list_value(args.channels)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Serialize the message
        message_data = list_value.SerializeToString()

        # Send the message
        print("Sending channel update...")
        if tap.send(message_data):
            print(f"Successfully sent cursor channel update: {args.channels}")
        else:
            print("Failed to send message", file=sys.stderr)
            sys.exit(1)

        # Give some time for the message to be processed
        time.sleep(0.5)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        tap.disconnect()
        print("Disconnected from tap")


if __name__ == "__main__":
    main()
