#!/usr/bin/env python3
"""High-performance real-time spike waveform viewer using pyqtgraph.

This is a replacement for `plot_spikes.py` when higher throughput is needed.
Parsing/recording runs in a background thread, while the GUI runs in the Qt
main thread.  Each channel gets its own ViewBox arranged in a 10×10 grid.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import struct
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Tuple

# Pin NumPy/OpenBLAS to a single thread so we don't starve the Qt GUI thread
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pyqtgraph as pg  # type: ignore
from pyqtgraph.Qt import QtCore, QtWidgets
from synapse.api.datatype_pb2 import Tensor
from synapse.client.taps import Tap

# Shared pen used for all waveforms
WHITE_PEN = pg.mkPen("w", width=1)

# ---------------------- tunables ----------------------
# Rendered FPS for GUI (ms interval for timer)
GUI_UPDATE_INTERVAL_MS = 100

# Max number of spikes popped from the queue per GUI update
MAX_POP_PER_TICK = 5000  # process more events per tick to keep queue short

# Sub-sample factor for plotting – set to 1 to plot every spike.  Higher values
# will plot 1/N spikes (still all are recorded to disk).
PLOT_SUBSAMPLE = 1

BACKLOG_THRESHOLD = 1000  # if queue grows beyond this, drain extra without plotting

STATS_WINDOW_SEC = 60.0  # horizon for dropped / unplotted stats


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
    dtype = np.int16 if tensor.dtype == Tensor.DType.DT_INT16 else np.float32

    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        fmt = f">{len(data_bytes) // 4}f"
        samples = np.array(struct.unpack(fmt, data_bytes), dtype=dtype).astype(
            np.float32
        )
    else:
        samples = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)

    # First value encodes a sequence counter stored as INT16, but we want to
    # treat it as **unsigned** so that 0…65,535 wraps naturally instead of
    # jumping to negative numbers when the high bit is set.
    seq = int(np.uint16(samples[0]).item())
    ch_id = int(samples[1]) + 1  # 1-based in GUI
    waveform = samples[2:]

    return seq, ts_sec, ch_id, waveform


# --------------------------------------------------------------------------------------
# Worker thread – receives tap data and dumps JSONL
# --------------------------------------------------------------------------------------


class ReceiverThread(threading.Thread):
    def __init__(self, tap: Tap, outfile: Path, queue_max=20000):
        super().__init__(daemon=True)
        import gzip

        self._tap = tap
        self._out = outfile.open("w", buffering=1)
        self._queue: deque[Tuple[float, int, np.ndarray]] = deque(maxlen=queue_max)
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Stats bookkeeping
        self._seq_mod = 2**16  # 16-bit counter wraps at 65,536
        self._dropped_total = 0
        self._received_total = 0
        self._bw_samples: deque[tuple[float, int]] = deque()
        self._seq_window: deque[tuple[float, int]] = deque()
        # Track the last seen sequence globally (one counter for all channels).
        self._last_seq: int | None = None

    def run(self):
        flush_every = 500  # disk flush interval
        events_since_flush = 0

        tensor_msg = Tensor()
        try:
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
                    # Sequence-based loss detection (wrap-around aware)
                    gap = 0
                    if self._last_seq is not None:
                        expected = (self._last_seq + 1) % self._seq_mod
                        if seq != expected:
                            gap = (seq - expected) % self._seq_mod
                            self._dropped_total += gap

                    self._last_seq = seq

                    self._received_total += 1

                    # Maintain 60-s sliding window of sequence stats
                    self._seq_window.append((now, gap))  # gap may be 0
                    window_cutoff = now - STATS_WINDOW_SEC
                    while self._seq_window and self._seq_window[0][0] < window_cutoff:
                        self._seq_window.popleft()

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
        finally:
            # Ensure resources closed exactly once after exit
            try:
                self._tap.close()
            except Exception:
                pass
            self._out.flush()
            self._out.close()

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
        """Signal the thread to exit and wait for clean shutdown."""
        self._stop_event.set()
        # Wait (up to 2 s) for the thread to terminate so that run() can
        # flush and close the file safely.  `join` is safe here because we
        # are always calling `stop()` from another thread.
        self.join(timeout=2.0)

    # --------------------------------------------------
    def get_stats(self) -> Tuple[float, float]:
        """Return (drop_pct [0-1], bandwidth_mbps)."""
        with self._stats_lock:
            # Compute drop statistics over the last 60 seconds
            window_received = len(self._seq_window)
            window_dropped = sum(gap for _, gap in self._seq_window)
            total_window = window_received + window_dropped
            drop_pct = (window_dropped / total_window) if total_window else 0.0

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
            curve.setClipToView(True)  # only render visible points
            curve.setDownsampling(auto=True)  # let pg pick the stride
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

    # Build GUI first so the consumer is ready before we start pulling data.
    app = QtWidgets.QApplication([])
    win, plot_map, curve_map = build_gui()
    # Display the window as a square rather than maximised to fill the screen.
    # A fixed size gives a more consistent layout across monitors.
    win.resize(1200, 1000)  # width, height in pixels – tweak as desired
    win.show()

    # ----------------------------------------------------------------------------
    # Connect to the device *after* the Qt event-loop starts so we can get the
    # window on screen immediately.  The connection handshake can take several
    # seconds, so we run it in a background thread to avoid blocking the GUI.
    # ----------------------------------------------------------------------------

    rx: ReceiverThread | None = None  # will be initialised asynchronously

    def _connect_and_start_receiver():  # runs in a worker thread
        nonlocal rx
        try:
            tap = Tap(args.device_ip)
            tap.connect(args.tap_name)
            print(f"Connected to {args.tap_name} @ {args.device_ip}")

            rx = ReceiverThread(tap, output_path)
            rx.start()
        except Exception as exc:
            # Surface connection errors in the GUI thread so they are visible
            QtCore.QTimer.singleShot(
                0,
                lambda: QtWidgets.QMessageBox.critical(
                    None, "Tap connection failed", str(exc)
                ),
            )

    # Kick off the connection in a daemon thread so startup is instantaneous.
    threading.Thread(target=_connect_and_start_receiver, daemon=True).start()

    # ------------------------------------------------------------------
    # Pre-allocated ring buffers (per-channel)
    # ------------------------------------------------------------------

    # We allocate once we know the waveform length (first spike).
    WAVEFORM_LEN: int | None = None  # length of a single spike waveform
    PER_WF_LEN: int | None = None  # waveform + NaN separator

    RING_CAPACITY = (
        5000  # max #spikes per 10-s window / channel – large enough to avoid wrap
    )

    # Shared, constant X array (one per channel) created once and reused – this
    # lets us call setData with the *same* object each time which speeds up pg.
    shared_ring_x: np.ndarray | None = None

    # Per-channel Y buffers and write heads
    ring_y: dict[int, np.ndarray] = {}
    ring_head: dict[int, int] = defaultdict(int)  # next index (0..RING_CAPACITY-1)
    ring_count: dict[int, int] = defaultdict(int)  # valid spikes stored (≤ capacity)

    current_window_start: float | None = None
    first_ts: float | None = None
    local_start_time: float | None = None  # wall-clock time when we saw the first spike
    elapsed_label = pg.LabelItem(justify="left")
    elapsed_label.setFixedWidth(90)  # prevent first column from expanding
    win.addItem(elapsed_label, row=0, col=0)

    # Track latest stats to show under elapsed
    last_stats_update = time.time()
    last_drop: float | None = None
    last_bw: float | None = None

    # Cache last applied column width so we only resize when necessary
    _last_col_width: int | None = None

    # Sliding-window counts for GUI-render stats: entries are (timestamp, plotted, dropped)
    gui_stats: deque[tuple[float, int, int]] = deque()

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
        nonlocal current_window_start, first_ts, local_start_time, last_stats_update, last_drop, last_bw, WAVEFORM_LEN, PER_WF_LEN, RING_CAPACITY, shared_ring_x, ring_y, ring_head, ring_count, rx

        # Receiver not yet connected – nothing to do
        if rx is None:
            return

        # Keep grid columns at equal widths adaptive to any window resize
        _update_column_widths()

        batch = rx.pop_batch(MAX_POP_PER_TICK)
        if not batch:
            return

        plotted_this_tick = 0  # spikes actually rendered (after subsampling)
        dropped_this_tick = (
            0  # spikes skipped for any reason (subsample or backlog discard)
        )

        # Track which channels received new data this tick so we only call
        # setData() for those.
        touched: set[int] = set()

        # Dynamically increase subsampling if we are falling behind to avoid GUI lag.
        subsample_factor = PLOT_SUBSAMPLE
        qsize = rx.queue_size()
        if qsize > BACKLOG_THRESHOLD * 4:
            subsample_factor *= 4
        elif qsize > BACKLOG_THRESHOLD * 2:
            subsample_factor *= 2

        for idx, (ts_sec, ch_id, wf) in enumerate(batch):
            if first_ts is None:
                first_ts = ts_sec
                local_start_time = time.time()

            # -------- 10-second rolling window reset -------------------
            if current_window_start is None or ts_sec >= current_window_start + 10.0:
                for cid in list(ring_y):
                    curve_map[cid].setData([], [])
                    ring_head[cid] = 0
                    ring_count[cid] = 0
                current_window_start = int(ts_sec // 10) * 10.0

            # ---------------- prepare per-spike data -------------------
            if (idx % subsample_factor) != 0:
                dropped_this_tick += 1  # skipped due to subsampling
                continue

            # Lazy-initialise the shared X template once we know waveform length.
            if WAVEFORM_LEN is None:
                # First ever spike – initialise shared constants and buffers
                WAVEFORM_LEN = wf.size
                PER_WF_LEN = WAVEFORM_LEN + 1  # +1 for NaN separator

                # Shared X template repeated RING_CAPACITY times – immutable
                shared_ring_x = np.tile(
                    np.append(np.arange(WAVEFORM_LEN, dtype=np.float32), np.nan),
                    RING_CAPACITY,
                ).astype(np.float32)

            # Ensure this channel has its ring buffer allocated
            if ch_id not in ring_y:
                ring_y[ch_id] = np.full(
                    PER_WF_LEN * RING_CAPACITY, np.nan, dtype=np.float32
                )
                ring_head[ch_id] = 0

            # Append the waveform followed by a NaN gap so adjacent waveforms
            # are not connected.
            offset = ring_head[ch_id] * PER_WF_LEN
            ring_y[ch_id][offset : offset + WAVEFORM_LEN] = wf
            ring_y[ch_id][offset + WAVEFORM_LEN] = np.nan

            ring_head[ch_id] = (ring_head[ch_id] + 1) % RING_CAPACITY
            ring_count[ch_id] = min(RING_CAPACITY, ring_count[ch_id] + 1)

            touched.add(ch_id)
            plotted_this_tick += 1

        # If we are falling behind, quickly drain the surplus without rendering
        while rx.queue_size() > BACKLOG_THRESHOLD:
            discarded = rx.pop_batch(MAX_POP_PER_TICK)
            dropped_this_tick += len(discarded)

        # Compose info label (elapsed + stats if available)
        if first_ts is not None and batch:
            elapsed_sec = batch[-1][0] - first_ts
        else:
            elapsed_sec = 0.0

        # ------------------ stats refresh ------------------
        now_wall = time.time()
        if plotted_this_tick or dropped_this_tick:
            gui_stats.append((now_wall, plotted_this_tick, dropped_this_tick))

        # Purge old GUI stats beyond window
        while gui_stats and now_wall - gui_stats[0][0] > STATS_WINDOW_SEC:
            gui_stats.popleft()

        # Aggregate GUI stats
        plotted_total = sum(p for _, p, _ in gui_stats)
        dropped_gui_total = sum(d for _, _, d in gui_stats)
        if plotted_total + dropped_gui_total:
            drop_gui_pct = dropped_gui_total / (plotted_total + dropped_gui_total)
        else:
            drop_gui_pct = 0.0

        # Network stats refresh once per second
        if now_wall - last_stats_update >= 1.0:
            last_drop, last_bw = rx.get_stats()
            last_stats_update = now_wall

        # Update main label text each GUI tick
        text_lines = [f"Elapsed: {elapsed_sec:0.1f} s"]

        if local_start_time is not None:
            wall_elapsed = time.time() - local_start_time
            lag_sec = elapsed_sec - wall_elapsed
            text_lines.append(f"Lag: {lag_sec:5.2f} s")

        if last_drop is not None:
            text_lines.extend(
                [
                    f"Dropped: {last_drop*100:5.2f}%",
                    f"Unplotted: {drop_gui_pct*100:5.2f}%",
                    f"BW: {last_bw:5.2f} Mbps",
                ]
            )

        elapsed_label.setText("<br/>".join(text_lines))

        # Update only the channels that received new data this tick.
        for cid in touched:
            if ring_count[cid] == 0:
                continue

            if ring_count[cid] < RING_CAPACITY:
                N = ring_count[cid] * PER_WF_LEN
                curve_map[cid].setData(
                    shared_ring_x[:N],
                    ring_y[cid][:N],
                    pen=WHITE_PEN,
                    connect="finite",
                )
            else:  # wrapped, need two slices
                h = ring_head[cid] * PER_WF_LEN
                y = np.concatenate((ring_y[cid][h:], ring_y[cid][:h]))
                curve_map[cid].setData(
                    shared_ring_x[: y.size],
                    y,
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
    def _stop_rx():
        if rx and rx.is_alive():
            rx.stop()

    QtCore.QCoreApplication.instance().aboutToQuit.connect(_stop_rx)

    try:
        pg.exec()
    finally:
        if rx and rx.is_alive():
            rx.stop()  # stop already joins internally


if __name__ == "__main__":
    main()
