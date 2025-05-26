#!/usr/bin/env python3

import argparse
import sys
from synapse.client.taps import Tap
import zmq
from proto.example_app_pb2 import ResetRequest
import time
def main():
    parser = argparse.ArgumentParser(description="Update the configuration of a Synapse device")
    parser.add_argument("--device-ip", required=True, help="IP address of the Synapse device")

    args = parser.parse_args()

    reset_request = ResetRequest()
    reset_request.reset = True

    tap = Tap(args.device_ip)
    reset_in_tap_port = 34979
    for taps in tap.list_taps():
        print(f"Tap: {taps}")
        if taps.name == "reset_in":
            reset_in_tap_port = taps.endpoint.split(":")[2]

    print(f"Connected to reset tap at {args.device_ip}")

    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    # Connect to your subscriber
    socket.connect(f"tcp://{args.device_ip}:{reset_in_tap_port}")
    time.sleep(1)
    socket.send(reset_request.SerializeToString())
    print(f"Reset request: {reset_request}")

    print(f"Sent reset request to {args.device_ip}:{reset_in_tap_port}")



if __name__ == "__main__":
    main()
