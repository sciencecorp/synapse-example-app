#!/usr/bin/env python3
"""
Collect paired (spike_features, controller_labels) training data.
Requires the modified C++ app running with config/easy_train.json.
Run for ~5 minutes while sweeping the joystick through all directions.
Usage: python3 client/collect_training_data.py --device-ip 192.168.16.238 --duration 300
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
    # Retry connect up to 5 times
    for attempt in range(5):
        if tap.connect(tap_name):
            break
        print(f"[{tap_name}] Connect attempt {attempt+1} failed, retrying...")
        time.sleep(2)
    else:
        print(f"[ERROR] Could not connect to tap '{tap_name}' after 5 attempts")
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
    parser.add_argument('--duration', type=float, default=300)
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
    
    time.sleep(3)  # Wait for first connection to fully establish
    
    label_thread = threading.Thread(
        target=tap_reader_thread,
        args=(args.device_ip, 'controller_labels', label_queue, stop_event),
        daemon=True)
    label_thread.start()

    time.sleep(3)  # Wait for second connection to settle

    print(f"\nRecording {args.duration:.0f}s")
    print("Sweep: UP (hold 2s) -> center -> DOWN -> center -> LEFT -> center -> RIGHT -> center, repeat\n")

    t0 = time.time()
    all_feats, all_labels = [], []
    try:
        while time.time() - t0 < args.duration:
            elapsed = time.time() - t0
            paired = sum(len(f) for f in all_feats)
            print(f"\r  {elapsed:5.1f}s  feat_q={feat_queue.qsize():4d}  "
                  f"label_q={label_queue.qsize():4d}  paired={paired:5d}",
                  end='', flush=True)
            time.sleep(1.0)
            f, l = pair_streams(feat_queue, label_queue)
            if len(f) > 0:
                all_feats.append(f)
                all_labels.append(l)
    except KeyboardInterrupt:
        print("\nStopping early...")

    stop_event.set()
    f, l = pair_streams(feat_queue, label_queue)
    if len(f) > 0:
        all_feats.append(f)
        all_labels.append(l)

    if not all_feats:
        print("\nNo data - check tap names with: synapsectl -u <device> taps list")
        sys.exit(1)

    features = np.concatenate(all_feats, axis=0)
    labels   = np.concatenate(all_labels, axis=0)
    print(f"\n\nCollected {len(features)} samples")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=features, labels=labels)
    print(f"Saved -> {args.out}")

if __name__ == '__main__':
    main()
