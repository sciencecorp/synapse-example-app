#!/usr/bin/env python3
"""
Structured data collection with clear direction holds.
Collects paired (spike_features, controller_labels) with timing metadata.
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from synapse.api.datatype_pb2 import Tensor
    from synapse.client.taps import Tap
except ImportError:
    print("ERROR: pip install --pre science-synapse")
    sys.exit(1)


def _tensor_to_array(raw_bytes):
    t = Tensor()
    t.ParseFromString(raw_bytes)
    return np.frombuffer(t.data, dtype=np.float32).copy()


def tap_reader_thread(device_ip, tap_name, data_queue, stop_event):
    tap = Tap(device_ip)
    for attempt in range(5):
        if tap.connect(tap_name):
            break
        print(f"[{tap_name}] Retry {attempt+1}...")
        time.sleep(2)
    else:
        print(f"[ERROR] Could not connect to '{tap_name}'")
        stop_event.set()
        return
    print(f"[{tap_name}] Connected")
    n = 0
    try:
        while not stop_event.is_set():
            raw = tap.read()
            if raw is None:
                time.sleep(0.001)
                continue
            data_queue.put((time.monotonic(), _tensor_to_array(raw)))
            n += 1
    finally:
        tap.disconnect()
        print(f"[{tap_name}] Done ({n} samples)")


def pair_streams(feat_queue, label_queue, max_dt_sec=0.05):
    feat_buf, label_buf = [], []
    while not feat_queue.empty():
        feat_buf.append(feat_queue.get_nowait())
    while not label_queue.empty():
        label_buf.append(label_queue.get_nowait())
    if not feat_buf or not label_buf:
        return np.empty((0, 32)), np.empty((0, 2))
    features, labels = [], []
    li = 0
    for ft, farr in feat_buf:
        while (li + 1 < len(label_buf) and
               abs(label_buf[li+1][0] - ft) < abs(label_buf[li][0] - ft)):
            li += 1
        lt, larr = label_buf[li]
        if abs(lt - ft) <= max_dt_sec:
            features.append(farr[:32].astype(np.float32))
            labels.append(larr[:2].astype(np.float32))
    if not features:
        return np.empty((0, 32)), np.empty((0, 2))
    return np.stack(features), np.stack(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-ip', required=True)
    parser.add_argument('--cycles', type=int, default=15,
                        help='Number of full UP/DOWN/LEFT/RIGHT/A cycles (default: 15)')
    parser.add_argument('--out', default='training_data.npz')
    args = parser.parse_args()

    feat_queue  = queue.Queue()
    label_queue = queue.Queue()
    stop_event  = threading.Event()

    feat_thread = threading.Thread(
        target=tap_reader_thread,
        args=(args.device_ip, 'spike_features', feat_queue, stop_event),
        daemon=True)
    feat_thread.start()
    time.sleep(3)
    label_thread = threading.Thread(
        target=tap_reader_thread,
        args=(args.device_ip, 'controller_labels', label_queue, stop_event),
        daemon=True)
    label_thread.start()
    time.sleep(3)

    HOLD_SEC   = 5   # hold each direction
    CENTER_SEC = 3   # rest at center between directions
    DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A_BUTTON']

    all_feats, all_labels = [], []
    total_paired = 0

    print(f"\nStarting {args.cycles} cycles × 5 directions")
    print("Instructions:")
    print("  UP/DOWN/LEFT/RIGHT: push joystick ALL THE WAY to the edge and hold still")
    print("  A_BUTTON: keep joystick centered, press and hold A button")
    print("  CENTER: release everything, joystick at rest\n")

    for cycle in range(1, args.cycles + 1):
        for direction in DIRECTIONS:
            # Drain queues before this hold
            pair_streams(feat_queue, label_queue)  # discard center data

            print(f"Cycle {cycle:2d}/{args.cycles} — CENTER (release joystick)...", end='', flush=True)
            time.sleep(CENTER_SEC)
            print(f" → HOLD {direction} NOW", flush=True)

            t0 = time.time()
            while time.time() - t0 < HOLD_SEC:
                remaining = HOLD_SEC - (time.time() - t0)
                print(f"\r  Holding {direction}... {remaining:.1f}s remaining   ", end='', flush=True)
                time.sleep(0.2)
            print()

            # Collect samples from this hold
            f, l = pair_streams(feat_queue, label_queue)
            if len(f) > 0:
                all_feats.append(f)
                all_labels.append(l)
                total_paired += len(f)
                print(f"  Collected {len(f)} samples (total: {total_paired})")

    stop_event.set()
    feat_thread.join(timeout=3)
    label_thread.join(timeout=3)

    # Final drain
    f, l = pair_streams(feat_queue, label_queue)
    if len(f) > 0:
        all_feats.append(f)
        all_labels.append(l)

    if not all_feats:
        print("No data collected.")
        sys.exit(1)

    features = np.concatenate(all_feats, axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    print(f"\nTotal: {len(features)} samples")
    print(f"Label X range: [{labels[:,0].min():.1f}, {labels[:,0].max():.1f}]")
    print(f"Label Y range: [{labels[:,1].min():.1f}, {labels[:,1].max():.1f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=features, labels=labels)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()