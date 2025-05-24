#!/usr/bin/env python3
"""High-performance real-time spike waveform viewer using pyqtgraph.

This is a replacement for `plot_spikes.py` when higher throughput is needed.
Parsing/recording runs in a background thread, while the GUI runs in the Qt
main thread.  Each channel gets its own ViewBox arranged in a 10×10 grid.
"""

from __future__ import annotations

import argparse
import json
import signal
import struct
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

import numpy as np
import pyqtgraph as pg  # type: ignore
from pyqtgraph.Qt import QtCore, QtWidgets
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap

# Shared pen used for all waveforms
WHITE_PEN = pg.mkPen("w", width=1)

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
# Tensor → tuple helper (now includes sequence number for stats)
# --------------------------------------------------------------------------------------


def tensor_to_tuple(tensor: Tensor) -> Tuple[int, float, int, np.ndarray]:
    """Convert a Tensor to (seq, ts_sec, channel_id, waveform)."""
    ts_sec = tensor.timestamp_ns / 1e9

    data_bytes = tensor.data
    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        fmt = f">{len(data_bytes) // 4}f"
        samples = np.array(struct.unpack(fmt, data_bytes), dtype=np.float32)
    else:
        samples = np.frombuffer(data_bytes, dtype=np.float32)

    seq = int(samples[0])
    ch_id = int(samples[1]) + 1  # 1-based in GUI
    waveform = samples[2:]

    return seq, ts_sec, ch_id, waveform


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
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Stats bookkeeping
        self._last_seq: int | None = None
        self._seq_mod = 2**24  # sequence wrap-around
        self._dropped_total = 0
        self._received_total = 0
        self._bw_samples: deque[tuple[float, int]] = deque()

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
            seq, ts_sec, ch_id, wf = tensor_to_tuple(tensor_msg)

            # ---------------------- stats update ----------------------
            now = time.time()
            with self._stats_lock:
                # Sequence-based loss detection
                if self._last_seq is not None:
                    expected = (self._last_seq + 1) % self._seq_mod
                    if seq != expected:
                        gap = (seq - expected) % self._seq_mod
                        self._dropped_total += gap
                self._last_seq = seq

                self._received_total += 1

                # Bandwidth window (1-s sliding)
                self._bw_samples.append((now, len(raw)))
                while self._bw_samples and now - self._bw_samples[0][0] > 1.0:
                    self._bw_samples.popleft()

            # ---------------------- GUI queue -------------------------
            with self._lock:
                self._queue.append((ts_sec, ch_id, wf))

            # Persist raw data (no seq)
            self._out.write(json.dumps([ts_sec, ch_id, wf.tolist()]) + "\n")
            events_since_flush += 1
            if events_since_flush >= flush_every:
                self._out.flush()
                events_since_flush = 0

    # --------------------------------------------------
    def pop_batch(self, max_items=2000):
        with self._lock:
            batch = [
                self._queue.popleft() for _ in range(min(max_items, len(self._queue)))
            ]
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

    # --------------------------------------------------
    def get_stats(self) -> Tuple[float, float]:
        """Return (drop_pct [0-1], bandwidth_mbps)."""
        with self._stats_lock:
            total = self._received_total + self._dropped_total
            drop_pct = (self._dropped_total / total) if total else 0.0

            total_bytes = sum(b for _, b in self._bw_samples)
            bandwidth_mbps = (total_bytes * 8) / 1e6  # bytes -> bits -> Mbps

        return drop_pct, bandwidth_mbps


# --------------------------------------------------------------------------------------
# GUI setup (one persistent PlotDataItem per channel)
# --------------------------------------------------------------------------------------


def build_gui() -> tuple[
    pg.GraphicsLayoutWidget,
    dict[int, pg.PlotItem],
    dict[int, pg.PlotDataItem],
]:
    bg = (34, 34, 34)
    pg.setConfigOption("background", bg)
    pg.setConfigOption("foreground", "w")
    # Enable OpenGL backend for faster rendering if available
    pg.setConfigOption("useOpenGL", True)

    w = pg.GraphicsLayoutWidget(title="Live spike waveforms (10-s windows)")

    plots: dict[int, pg.PlotItem] = {}
    curves: dict[int, pg.PlotDataItem] = {}

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
            plot.hideAxis("bottom")
            plot.hideAxis("left")

            # Fixed y-axis with no autorange and no visible buttons (removes the "A").
            plot.setYRange(-250, 200, padding=0)
            plot.disableAutoRange(axis="y")
            plot.hideButtons()

            plot.setXRange(0, 49)  # waveform length – 1
            plot.setTitle(
                f"{next(ch for ch, pos in ch_to_pos.items() if pos == (r, c))}",
                color="w",
                size="8pt",
            )

            # Add a white border around each ViewBox for visual separation
            plot.getViewBox().setBorder({"color": (200, 200, 200), "width": 1})

            # Pre-create a single PlotDataItem that will hold *all* spikes for this
            # channel.  Using connect="finite" allows us to insert np.nan between
            # successive waveforms so they are not joined.
            curve = pg.PlotDataItem([], [], pen=WHITE_PEN, connect="finite")
            plot.addItem(curve)

            cid = next(ch for ch, pos in ch_to_pos.items() if pos == (r, c))
            plots[cid] = plot
            curves[cid] = curve

    # Note: column widths will be adjusted dynamically in the main loop

    return w, plots, curves


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
    win, plot_map, curve_map = build_gui()
    # Display the window as a square rather than maximised to fill the screen.
    # A fixed size gives a more consistent layout across monitors.
    win.resize(1200, 1000)  # width, height in pixels – tweak as desired
    win.show()

    # ------------------------------------------------------------------
    # Per-channel storage for waveform data (10-s rolling window)
    # For each channel we keep two Python lists of floats.  Each waveform is
    # appended followed by a NaN so that lines are not connected across
    # spikes.  The lists are converted to numpy once per GUI update.
    # ------------------------------------------------------------------

    chan_x: dict[int, list[float]] = defaultdict(list)
    chan_y: dict[int, list[float]] = defaultdict(list)

    # X-axis template (filled after first spike when we know waveform length)
    X_TEMPLATE: np.ndarray | None = None

    current_window_start: float | None = None
    first_ts: float | None = None
    elapsed_label = pg.LabelItem(justify="left")
    elapsed_label.setFixedWidth(90)  # prevent first column from expanding
    win.addItem(elapsed_label, row=0, col=0)

    # Track latest stats to show under elapsed
    last_stats_update = time.time()
    last_drop: float | None = None
    last_bw: float | None = None

    # ---------------------- tunables ----------------------
    # Rendered FPS for GUI (ms interval for timer)
    GUI_UPDATE_INTERVAL_MS = 16  # ~60 FPS for smoother updates

    # Max number of spikes popped from the queue per GUI update
    MAX_POP_PER_TICK = 5000  # process more events per tick to keep queue short

    # Sub-sample factor for plotting – set to 1 to plot every spike.  Higher values
    # will plot 1/N spikes (still all are recorded to disk).
    PLOT_SUBSAMPLE = 1

    BACKLOG_THRESHOLD = 4000  # if queue grows beyond this, drain extra without plotting

    # Cache last applied column width so we only resize when necessary
    _last_col_width: int | None = None

    def _update_column_widths():
        """Ensure all columns have uniform width that follows window resizing."""
        nonlocal _last_col_width
        layout = win.ci.layout  # type: ignore[attr-defined]
        # Divide available width evenly across 10 columns (guard against 0)
        col_width = max(10, win.width() // 11)
        if col_width != _last_col_width:
            for c in range(10):
                try:
                    layout.setColumnFixedWidth(c, col_width)
                except Exception:
                    # If the Qt binding lacks this method, silently skip
                    pass
            _last_col_width = col_width

    def update_gui():
        nonlocal current_window_start, first_ts, last_stats_update, last_drop, last_bw, X_TEMPLATE
        # Keep grid columns at equal widths adaptive to any window resize
        _update_column_widths()

        batch = rx.pop_batch(MAX_POP_PER_TICK)
        if not batch:
            return

        # If we are falling behind, quickly drain the surplus without rendering
        while rx.queue_size() > BACKLOG_THRESHOLD:
            _ = rx.pop_batch(
                MAX_POP_PER_TICK
            )  # discard plotting, but keeps elapsed correct

        # Track which channels received new data this tick so we only call
        # setData() for those.
        touched: set[int] = set()

        for idx, (ts_sec, ch_id, wf) in enumerate(batch):
            if first_ts is None:
                first_ts = ts_sec

            # -------- 10-second rolling window reset -------------------
            if current_window_start is None or ts_sec >= current_window_start + 10.0:
                for cid in list(chan_x):
                    chan_x[cid].clear()
                    chan_y[cid].clear()
                    curve_map[cid].setData([], [])
                current_window_start = int(ts_sec // 10) * 10.0

            # ---------------- prepare per-spike data -------------------
            if (idx % PLOT_SUBSAMPLE) != 0:
                continue  # subsampling

            # Lazy-initialise the shared X template once we know waveform length.
            if X_TEMPLATE is None:
                X_TEMPLATE = np.arange(wf.size, dtype=np.float32)

            # Append the waveform followed by a NaN gap so adjacent waveforms
            # are not connected.
            chan_x[ch_id].extend(X_TEMPLATE)
            chan_x[ch_id].append(np.nan)

            chan_y[ch_id].extend(wf.astype(np.float32))
            chan_y[ch_id].append(np.nan)

            touched.add(ch_id)

        # Compose info label (elapsed + stats if available)
        if first_ts is not None and batch:
            elapsed_sec = batch[-1][0] - first_ts
        else:
            elapsed_sec = 0.0

        # ------------------ stats refresh ------------------
        if time.time() - last_stats_update >= 1.0:
            last_drop, last_bw = rx.get_stats()
            last_stats_update = time.time()

        # Update main label text each GUI tick
        text_lines = [f"Elapsed: {elapsed_sec:0.1f} s"]
        if last_drop is not None:
            text_lines.extend(
                [
                    f"Drop: {last_drop*100:5.2f}%",
                    f"BW: {last_bw:5.2f} Mbps",
                ]
            )

        elapsed_label.setText("<br/>".join(text_lines))

        # Update only the channels that received new data this tick.
        for cid in touched:
            if chan_x[cid]:
                curve_map[cid].setData(
                    np.asarray(chan_x[cid], dtype=np.float32),
                    np.asarray(chan_y[cid], dtype=np.float32),
                    pen=WHITE_PEN,
                    connect="finite",
                )

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
