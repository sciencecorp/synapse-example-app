#!/usr/bin/env python3
"""High-performance real-time spike waveform viewer using pyqtgraph.

This is a replacement for `plot_spikes.py` when higher throughput is needed.
Parsing/recording runs in a background thread, while the GUI runs in the Qt
main thread.  Each channel gets its own ViewBox arranged in a 10×10 grid.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
import signal
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

import numpy as np
import pyqtgraph as pg  # type: ignore
from pyqtgraph.Qt import QtCore, QtWidgets
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap

# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Live spike viewer (pyqtgraph backend)")
    p.add_argument("--device-ip", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--tap-name", default="spike_waveforms")
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Tensor → tuple helper
# --------------------------------------------------------------------------------------

def tensor_to_tuple(tensor: Tensor) -> Tuple[float, int, np.ndarray]:
    ts_sec = tensor.timestamp_ns / 1e9

    data_bytes = tensor.data
    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        fmt = f">{len(data_bytes) // 4}f"
        waveform = np.array(struct.unpack(fmt, data_bytes), dtype=np.float32)
    else:
        waveform = np.frombuffer(data_bytes, dtype=np.float32)

    ch_id = int(waveform[0]) + 1  # 1-based
    return ts_sec, ch_id, waveform[1:]


# --------------------------------------------------------------------------------------
# Worker thread – receives tap data and dumps JSONL
# --------------------------------------------------------------------------------------

class ReceiverThread(threading.Thread):
    def __init__(self, tap: Tap, outfile: Path, queue_max=20000):
        super().__init__(daemon=True)
        self._tap = tap
        self._out = outfile.open("w", buffering=1)
        self._queue: deque[Tuple[float, int, np.ndarray]] = deque(maxlen=queue_max)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        flush_every = 500  # disk flush interval
        events_since_flush = 0

        tensor_msg = Tensor()
        while not self._stop_event.is_set():
            raw = self._tap.read()
            if raw is None:
                time.sleep(0.0002)
                continue

            tensor_msg.ParseFromString(raw)
            tup = tensor_to_tuple(tensor_msg)

            # Enqueue for GUI
            with self._lock:
                self._queue.append(tup)

            # Persist
            self._out.write(json.dumps([tup[0], tup[1], tup[2].tolist()]) + "\n")
            events_since_flush += 1
            if events_since_flush >= flush_every:
                self._out.flush()
                events_since_flush = 0

    # --------------------------------------------------
    def pop_batch(self, max_items=2000):
        with self._lock:
            batch = [self._queue.popleft() for _ in range(min(max_items, len(self._queue)))]
        return batch

    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def stop(self):
        self._stop_event.set()
        try:
            self._tap.close()
        except Exception:
            pass
        self._out.flush()
        self._out.close()


# --------------------------------------------------------------------------------------
# GUI setup
# --------------------------------------------------------------------------------------

def build_gui() -> tuple[pg.GraphicsLayoutWidget, dict[int, pg.PlotItem]]:
    bg = (34, 34, 34)
    pg.setConfigOption("background", bg)
    pg.setConfigOption("foreground", "w")
    # Enable OpenGL backend for faster rendering if available
    pg.setConfigOption("useOpenGL", True)

    w = pg.GraphicsLayoutWidget(title="Live spike waveforms (10-s windows)")

    plots: dict[int, pg.PlotItem] = {}

    n_rows = n_cols = 10
    ch_to_pos = {}
    for idx, ch in enumerate(range(1, 9), start=1):
        ch_to_pos[ch] = (0, idx)
    ch = 9
    for r in range(1, 9):
        for c in range(n_cols):
            if ch > 88:
                break
            ch_to_pos[ch] = (r, c)
            ch += 1
    for idx, ch in enumerate(range(89, 97), start=1):
        ch_to_pos[ch] = (9, idx)

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in [(0, 0)]:
                # Leave top-left corner empty so the elapsed-time label can be added later
                continue
            if (r, c) in [(0, 9), (9, 0), (9, 9)]:
                w.addLabel("", row=r, col=c)
                continue

            plot = w.addPlot(row=r, col=c)
            plot.setMenuEnabled(False)
            plot.hideAxis('bottom')
            plot.hideAxis('left')
            plot.setYRange(-200, 200)
            plot.setXRange(0, 49)  # waveform length – 1
            plot.setTitle(f"{next(ch for ch, pos in ch_to_pos.items() if pos == (r, c))}", color='w', size="8pt")
            # Add a white border around each ViewBox for visual separation
            plot.getViewBox().setBorder({'color': (200, 200, 200), 'width': 1})
            plots[next(ch for ch, pos in ch_to_pos.items() if pos == (r, c))] = plot

    return w, plots


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tap = Tap(args.device_ip)
    tap.connect(args.tap_name)
    print(f"Connected to {args.tap_name} @ {args.device_ip}")

    rx = ReceiverThread(tap, output_path)
    rx.start()

    app = QtWidgets.QApplication([])
    win, plot_map = build_gui()
    win.showMaximized()

    # Per-channel buffers (cleared every 10 s window)
    buffers: dict[int, list[np.ndarray]] = defaultdict(list)

    # Constant pen used for all spikes (white, 1-px wide)
    WHITE_PEN = pg.mkPen('w', width=1)

    current_window_start: float | None = None
    first_ts: float | None = None
    elapsed_label = pg.LabelItem(justify="left")
    elapsed_label.setFixedWidth(90)  # prevent first column from expanding
    win.addItem(elapsed_label, row=0, col=0)

    # ---------------------- tunables ----------------------
    # Rendered FPS for GUI (ms interval for timer)
    GUI_UPDATE_INTERVAL_MS = 16  # ~60 FPS for smoother updates

    # Max number of spikes popped from the queue per GUI update
    MAX_POP_PER_TICK = 5000  # process more events per tick to keep queue short

    # Sub-sample factor for plotting – set to 1 to plot every spike.  Higher values
    # will plot 1/N spikes (still all are recorded to disk).
    PLOT_SUBSAMPLE = 1

    BACKLOG_THRESHOLD = 4000  # if queue grows beyond this, drain extra without plotting

    def update_gui():
        nonlocal current_window_start, first_ts
        batch = rx.pop_batch(MAX_POP_PER_TICK)
        if not batch:
            return

        # If we are falling behind, quickly drain the surplus without rendering
        while rx.queue_size() > BACKLOG_THRESHOLD:
            _ = rx.pop_batch(MAX_POP_PER_TICK)  # discard plotting, but keeps elapsed correct

        for idx, (ts_sec, ch_id, wf) in enumerate(batch):
            if first_ts is None:
                first_ts = ts_sec
            if current_window_start is None or ts_sec >= current_window_start + 10.0:
                # commit and clear
                for cid, plot in plot_map.items():
                    if buffers[cid]:
                        plot.addItem(pg.PlotDataItem(np.vstack(buffers[cid])[:, 0],
                                                     np.vstack(buffers[cid])[:, 1],
                                                     pen=WHITE_PEN,
                                                     connect="all"))
                    buffers[cid].clear()
                    plot.clear()  # clear old curves
                current_window_start = int(ts_sec // 10) * 10.0

            x = np.arange(wf.size, dtype=float)
            if (idx % PLOT_SUBSAMPLE) == 0:
                buffers[ch_id].append(np.column_stack((x, wf)))

        # update elapsed label
        if first_ts is not None and batch:
            elapsed_label.setText(f"Elapsed: {batch[-1][0] - first_ts:0.1f} s")

        # Draw incremental segments quickly for immediate feedback (respect subsample)
        for cid in buffers:
            if buffers[cid]:
                seg = buffers[cid][-1]
                plot_map[cid].plot(seg[:, 0], seg[:, 1], pen=WHITE_PEN)

    timer = QtCore.QTimer()
    timer.timeout.connect(update_gui)

    timer.start(GUI_UPDATE_INTERVAL_MS)

    def _handle_sigint(_sig, _frame):
        print("\nSIGINT received – shutting down …", file=sys.stderr)
        QtWidgets.QApplication.quit()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Ensure receiver stops when the Qt app is about to quit
    QtCore.QCoreApplication.instance().aboutToQuit.connect(rx.stop)

    try:
        pg.exec()
    finally:
        if rx.is_alive():
            rx.stop()


if __name__ == "__main__":
    main() 