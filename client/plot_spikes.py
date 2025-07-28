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
from typing import Any, Tuple

# Pin NumPy/OpenBLAS to a single thread so we don't starve the Qt GUI thread
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pyqtgraph as pg  # type: ignore
from pyqtgraph.Qt import QtCore, QtWidgets
from synapse.api.datatype_pb2 import Tensor

# Tap API
from synapse.client.taps import Tap

# Shared pen used for all waveforms
WHITE_PEN = pg.mkPen("w", width=1)

# ---------------------- tunables ----------------------
# Rendered FPS for GUI (ms interval for timer)
GUI_UPDATE_INTERVAL_MS = 25  # match 25-ms spike-count bins (40 Hz)

# Max number of spikes popped from the queue per GUI update
MAX_POP_PER_TICK = 5000  # process more events per tick to keep queue short

# Sub-sample factor for plotting – set to 1 to plot every spike.  Higher values
# will plot 1/N spikes (still all are recorded to disk).
PLOT_SUBSAMPLE = 1

BACKLOG_THRESHOLD = 1000  # if queue grows beyond this, drain extra without plotting

STATS_WINDOW_SEC = 60.0  # horizon for dropped / unplotted stats

# --------------------------------------------------------------------------------------
# Raster display scaling
# --------------------------------------------------------------------------------------

# Pixel height allocated per channel row in the binned-spike raster.  Increasing this
# value makes each channel line thicker (default 3 → ~3× taller than before).
RASTER_ROW_HEIGHT = 2

# Upper bound for the raster colour scale (counts ≥ this value map to the top colour).
RASTER_CMAP_MAX = 3

# --------------------------------------------------------------------------------------
# Spike-count bin configuration
# --------------------------------------------------------------------------------------

BIN_MS = 25.0
WINDOW_SEC = 10.0  # raster window length
BIN_PER_SEC = int(1000.0 / BIN_MS)
RASTER_BINS = int(WINDOW_SEC * BIN_PER_SEC)
MAX_CHANNELS = 96


# --------------------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Live spike viewer (pyqtgraph backend)")
    p.add_argument("--device-ip", required=True)
    p.add_argument("--tap-name", default="spike_waveforms")
    p.add_argument("--gpio-tap-name", default="gpio")
    p.add_argument("--spike-count-tap", default="spike_counts")
    p.add_argument(
        "--electrode-map",
        required=True,
        help="Path to electrode map JSON file describing the channel grid",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Electrode-map helper
# --------------------------------------------------------------------------------------


def load_electrode_map(path: Path) -> list[list[int | None]]:
    """Load a grid-based electrode map from *path*.

    The file must contain a JSON array of arrays (2-D list).  Each entry should be an
    integer channel ID or *null*/0 to denote an empty slot.  All rows must have the
    same length.
    """

    with path.open() as fp:
        grid = json.load(fp)

    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        raise ValueError("Electrode map must be a JSON array of arrays")

    row_lens = {len(row) for row in grid}
    if len(row_lens) != 1:
        raise ValueError("All rows in the electrode map must have the same length")

    norm_grid: list[list[int | None]] = []
    for row in grid:
        norm_row: list[int | None] = []
        for val in row:
            if val is None:
                norm_row.append(None)
            elif isinstance(val, int):
                norm_row.append(val)
            else:
                raise ValueError("Electrode map values must be integers or null/0")
        norm_grid.append(norm_row)

    return norm_grid


# --------------------------------------------------------------------------------------
# Tensor → tuple helper (now includes sequence number for stats)
# --------------------------------------------------------------------------------------


def tensor_to_tuple(tensor: Tensor) -> Tuple[int, float, int, np.ndarray]:
    """Convert a Tensor to (seq, ts_sec, channel_id, waveform)."""
    ts_sec = tensor.timestamp_ns / 1e9

    data_bytes = tensor.data

    # ---------------- INT8 payload ----------------------------------
    if tensor.dtype == Tensor.DType.DT_INT8:
        raw = np.frombuffer(data_bytes, dtype=np.int8)

        # Treat first two entries as unsigned for natural wrap-around
        seq = int(np.uint8(raw[0]).item())
        ch_id = int(np.uint8(raw[1]).item())

        # Scale waveform back to µV (original detector clips at ±200/-250)
        wf = raw[2:].astype(np.float32) * (200.0 / 127.0)

        return seq, ts_sec, ch_id, wf

    # ---------------- INT16 payload ---------------------------------
    if tensor.dtype == Tensor.DType.DT_INT16:
        raw = np.frombuffer(data_bytes, dtype=np.int16)
        seq = int(np.uint16(raw[0]).item())
        ch_id = int(raw[1])
        wf = raw[2:].astype(np.float32)
        return seq, ts_sec, ch_id, wf

    # ---------------- FLOAT32 fallback ------------------------------
    dtype = np.float32
    if tensor.endianness == Tensor.Endianness.TENSOR_BIG_ENDIAN:
        fmt = f">{len(data_bytes) // 4}f"
        samples = np.array(struct.unpack(fmt, data_bytes), dtype=dtype)
    else:
        samples = np.frombuffer(data_bytes, dtype=dtype)

    seq = int(np.uint16(samples[0]).item())
    ch_id = int(samples[1])
    waveform = samples[2:]

    return seq, ts_sec, ch_id, waveform


# ------------------------------------------------------------------
# Receiver for spike waveforms (batched or single)
# ------------------------------------------------------------------


class ReceiverThread(threading.Thread):
    def __init__(self, tap: Tap, queue_max=20000):
        super().__init__(daemon=True)
        self._tap = tap
        self._queue: deque[Tuple[float, int, np.ndarray]] = deque(maxlen=queue_max)
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Stats bookkeeping
        self._seq_mod = 2**8  # 8-bit counter wraps at 256
        self._dropped_total = 0
        self._received_total = 0
        self._bw_samples: deque[tuple[float, int]] = deque()
        self._seq_window: deque[tuple[float, int]] = deque()
        # Track the last seen sequence globally (one counter for all channels).
        self._last_seq: int | None = None

    def run(self):
        # Disk persistence removed – no flush bookkeeping needed

        tensor_msg = Tensor()
        try:
            while not self._stop_event.is_set():
                raw = self._tap.read()
                if raw is None:
                    time.sleep(0.0002)
                    continue

                tensor_msg.ParseFromString(raw)

                # ==========================================================
                # Handle batched or single-spike tensors
                # ==========================================================
                def _enqueue(ts_s: float, seq_val: int, cid: int, waveform: np.ndarray):
                    # ---------------- stats update -----------------------
                    now_local = time.time()
                    with self._stats_lock:
                        gap_v = 0
                        if self._last_seq is not None:
                            expected = (self._last_seq + 1) % self._seq_mod
                            if seq_val != expected:
                                gap_v = (seq_val - expected) % self._seq_mod
                                self._dropped_total += gap_v

                        self._last_seq = seq_val

                        self._received_total += 1

                        self._seq_window.append((now_local, gap_v))
                        window_cutoff = now_local - STATS_WINDOW_SEC
                        while (
                            self._seq_window and self._seq_window[0][0] < window_cutoff
                        ):
                            self._seq_window.popleft()

                        self._bw_samples.append((now_local, len(raw)))
                        while (
                            self._bw_samples
                            and now_local - self._bw_samples[0][0] > 1.0
                        ):
                            self._bw_samples.popleft()

                    # --------------- queue push -------------------------
                    with self._lock:
                        self._queue.append((ts_s, cid, waveform))

                ts_sec = tensor_msg.timestamp_ns / 1e9

                if (
                    tensor_msg.dtype == Tensor.DType.DT_INT8
                    and len(tensor_msg.shape) >= 2
                ):
                    rows = tensor_msg.shape[0]
                    row_len = tensor_msg.shape[1]
                    raw_arr = np.frombuffer(tensor_msg.data, dtype=np.int8)
                    raw_arr = raw_arr.reshape(rows, row_len)

                    for r in raw_arr:
                        seq_v = int(np.uint8(r[0]).item())
                        cid_v = int(np.uint8(r[1]).item())
                        wf_v = r[2:].astype(np.float32) * (
                            200.0 / 127.0
                        )  # scale back to µV
                        _enqueue(ts_sec, seq_v, cid_v, wf_v)
                else:
                    seq, ts_sec_s, ch_id, wf = tensor_to_tuple(tensor_msg)
                    _enqueue(ts_sec_s, seq, ch_id, wf)
        finally:
            try:
                self._tap.close()
            except Exception:
                pass

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


# ---------------- GPIO level receiver (small helper) -------------------


class GPIORxThread(threading.Thread):
    """Background reader that tracks the latest GPIO level (0/1).

    Keeps a tight read-loop on the GPIO tap so the ZeroMQ connection stays
    healthy even if the GUI only polls the value once per timer tick.
    """

    def __init__(self, tap: Tap):
        super().__init__(daemon=True)
        self._tap = tap
        self._level = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    # --------------------------------------------------
    def run(self):
        tensor_msg = Tensor()
        import numpy as np

        try:
            while not self._stop_event.is_set():
                raw = self._tap.read()
                if raw is None:
                    time.sleep(0.0002)
                    continue

                tensor_msg.ParseFromString(raw)
                if len(tensor_msg.shape) >= 1:
                    data_arr = np.frombuffer(tensor_msg.data, dtype=np.int16)
                    if data_arr.size:
                        lvl = 1 if data_arr[-1] != 0 else 0
                        with self._lock:
                            self._level = lvl
        finally:
            try:
                self._tap.close()
            except Exception:
                pass

    # --------------------------------------------------
    def get_level(self) -> int:
        with self._lock:
            return self._level

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)


# --------------------------------------------------------------------------------------
# Spike-count receiver
# --------------------------------------------------------------------------------------


class CountRxThread(threading.Thread):
    """Background thread that streams spike-count tensors.

    Provides a queue of (timestamp_sec, counts ndarray) entries.
    """

    def __init__(self, tap: Tap, queue_max=400):
        super().__init__(daemon=True)
        self._tap = tap
        self._queue: deque[Tuple[float, "np.ndarray[Any, np.float32]"]] = deque(
            maxlen=queue_max
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Arrival-rate tracking (timestamps of recent tensors)
        self._ts_window: deque[float] = deque()

    # --------------------------------------------------
    def run(self):
        tensor_msg = Tensor()
        import numpy as np  # local import to avoid circular

        try:
            while not self._stop_event.is_set():
                raw = self._tap.read()
                if raw is None:
                    time.sleep(0.0002)
                    continue

                tensor_msg.ParseFromString(raw)

                if tensor_msg.dtype != Tensor.DType.DT_UINT16:
                    continue  # ignore unexpected

                counts = np.frombuffer(tensor_msg.data, dtype=np.uint16)
                ts_s = tensor_msg.timestamp_ns / 1e9

                with self._lock:
                    self._queue.append((ts_s, counts))

                    # --- update arrival timestamps window (≤1 s) ---
                    self._ts_window.append(time.time())
                    cutoff = self._ts_window[-1] - 1.0
                    while self._ts_window and self._ts_window[0] < cutoff:
                        self._ts_window.popleft()
        finally:
            try:
                self._tap.close()
            except Exception:
                pass

    # --------------------------------------------------
    def pop_next(self):
        with self._lock:
            if self._queue:
                return self._queue.popleft()
        return None

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2.0)

    # --------------------------------------------------
    def get_rate(self) -> float:
        """Return current spike-count tensor rate (bins/s) over the last second."""
        with self._lock:
            return float(len(self._ts_window))


# --------------------------------------------------------------------------------------
# GUI setup (one persistent PlotDataItem per channel)
# --------------------------------------------------------------------------------------


def build_spike_grid(grid: list[list[int | None]]) -> tuple[
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

    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows else 0

    content_row_offset = 0  # grid starts at first row now (stats banner separate)

    for r in range(n_rows):
        for c in range(n_cols):
            grid_row = r + content_row_offset
            cid = grid[r][c]

            # Blank / unused slot – leave empty for visual spacing.
            if cid is None:
                w.addLabel("", row=grid_row, col=c)
                continue

            plot = w.addPlot(row=grid_row, col=c)
            plot.setMenuEnabled(False)
            plot.hideAxis("bottom")
            plot.hideAxis("left")

            # Fixed y-axis with no autorange and no visible buttons (removes the "A").
            plot.setYRange(-250, 200, padding=0)
            plot.disableAutoRange(axis="y")
            plot.hideButtons()

            plot.setXRange(0, 49)  # waveform length – 1
            plot.setTitle(
                f"{cid}",
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

            plots[cid] = plot
            curves[cid] = curve

    # Note: column widths will be adjusted dynamically in the main loop

    return w, plots, curves


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    args = parse_args()

    # Build GUI first so the consumer is ready before we start pulling data.
    app = QtWidgets.QApplication([])
    electrode_grid = load_electrode_map(Path(args.electrode_map))
    n_rows, n_cols = len(electrode_grid), len(electrode_grid[0])

    # ------------------ sizing logic ------------------
    BASE_CELL_W, BASE_CELL_H = 120.0, 100.0
    header_rows = 0  # stats handled by separate widget
    PAD_W, PAD_H = 40, 40  # window frame / margin allowance

    STATS_H = 30  # approximate height for stats banner

    # Determine available desktop geometry (fallback to 1920x1080 if unavailable)
    screen = app.primaryScreen()
    if screen is not None:
        geom = screen.availableGeometry()
        avail_w, avail_h = geom.width(), geom.height()
    else:
        avail_w, avail_h = 1920, 1080

    # Required size at base cell dimensions (no spacing – we set spacing=0 below)
    req_w = n_cols * BASE_CELL_W + PAD_W
    req_h = n_rows * BASE_CELL_H + PAD_H + STATS_H

    # If too large, scale cells down uniformly so the whole window fits.
    scale = min(1.0, (avail_w * 0.9) / req_w, (avail_h * 0.9) / req_h)

    CELL_W = int(BASE_CELL_W * scale)
    CELL_H = int(BASE_CELL_H * scale)

    base_w = int(n_cols * CELL_W + PAD_W)
    base_h = int(n_rows * CELL_H + PAD_H + STATS_H)

    spike_widget, plot_map, curve_map = build_spike_grid(electrode_grid)

    # Remove inter-item spacing to keep computed sizes accurate
    try:
        spike_widget.ci.setSpacing(0)  # type: ignore[attr-defined]
    except Exception:
        pass

    spike_widget.resize(base_w, base_h)
    spike_widget.setMinimumSize(base_w, base_h)

    # Container combines stats + spike grid
    container = QtWidgets.QWidget()
    vlayout = QtWidgets.QVBoxLayout(container)
    vlayout.setContentsMargins(0, 0, 0, 0)
    vlayout.setSpacing(0)

    # Stats banner widget and helpers
    stats_widget = QtWidgets.QWidget()
    stats_layout = QtWidgets.QHBoxLayout(stats_widget)
    stats_layout.setContentsMargins(5, 2, 5, 2)
    stats_layout.setSpacing(5)

    # ----- GPIO status dot (container so it stretches like other stats) -----
    gpio_container = QtWidgets.QWidget()
    gpio_container_layout = QtWidgets.QHBoxLayout(gpio_container)
    gpio_container_layout.setContentsMargins(0, 0, 0, 0)
    gpio_container_layout.setSpacing(4)
    gpio_container_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

    gpio_label = QtWidgets.QLabel("GPIO:")
    gpio_label.setStyleSheet("color: white;")
    gpio_container_layout.addWidget(gpio_label)

    gpio_dot = QtWidgets.QLabel()
    DOT_SIZE = 12
    gpio_dot.setFixedSize(DOT_SIZE, DOT_SIZE)
    gpio_dot.setStyleSheet("background-color: #00ff00; border-radius: 6px;")
    gpio_container_layout.addWidget(gpio_dot)

    stats_layout.addWidget(gpio_container, 1)  # match stretch of other segments
    stats_labels: list[QtWidgets.QLabel] = []

    def _ensure_stats_labels(count: int):
        while len(stats_labels) < count:
            lbl = QtWidgets.QLabel(
                "", alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
            )
            lbl.setStyleSheet("color: white;")
            stats_layout.addWidget(lbl, 1)  # equal stretch
            stats_labels.append(lbl)

    def _update_stats_text(segments: list[str]):
        _ensure_stats_labels(len(segments))
        for idx, seg in enumerate(segments):
            stats_labels[idx].setText(seg)
        for idx in range(len(segments), len(stats_labels)):
            stats_labels[idx].setText("")

    vlayout.addWidget(stats_widget)

    # ---------------------------------------------------
    # Raster plot for spike counts (10-s rolling)
    # ---------------------------------------------------
    raster_plot = pg.PlotWidget()
    raster_plot.setMenuEnabled(False)
    raster_plot.hideAxis("bottom")
    raster_plot.hideAxis("left")
    raster_plot.setAspectLocked(False)

    # Fixed ranges
    raster_plot.setLimits(xMin=0, xMax=RASTER_BINS, yMin=0, yMax=MAX_CHANNELS)
    raster_plot.setXRange(0, RASTER_BINS)
    raster_plot.setYRange(0, MAX_CHANNELS)

    # Make the raster ~3× taller so individual channel rows are easier to see.
    # This simply allocates more vertical space for the widget; the image will scale
    # accordingly, giving each channel several pixels of height instead of one.
    raster_plot.setMinimumHeight(MAX_CHANNELS * RASTER_ROW_HEIGHT)

    # Image item for raster
    raster_img = pg.ImageItem(axisOrder="row-major")
    raster_plot.addItem(raster_img)

    # Ensure full-width view
    raster_plot.setXRange(0, RASTER_BINS - 1, padding=0)
    raster_plot.setYRange(0, MAX_CHANNELS - 1, padding=0)

    # Colour map – fallback to grayscale if unavailable
    try:
        from pyqtgraph.colormap import getFromMatplotlib

        cmap = getFromMatplotlib("viridis")
        raster_img.setLookupTable(cmap.getLookupTable(alpha=False))
    except Exception:
        pass  # default pg LUT

    # Cursor line
    cursor_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("r"))
    raster_plot.addItem(cursor_line)

    vlayout.addWidget(raster_plot)

    vlayout.addWidget(spike_widget)

    container.resize(base_w, base_h)
    container.setMinimumSize(base_w, base_h)
    container.show()

    # Track last applied cell dimensions so we only update when needed
    _last_cell_size: tuple[int, int] | None = None

    # ----------------------------------------------------------------------------
    # Connect to the device *after* the Qt event-loop starts so we can get the
    # window on screen immediately.  The connection handshake can take several
    # seconds, so we run it in a background thread to avoid blocking the GUI.
    # ----------------------------------------------------------------------------

    rx: ReceiverThread | None = None  # spike tap receiver
    gpio_rx: GPIORxThread | None = None  # background GPIO reader
    cnt_rx: CountRxThread | None = None  # spike-count receiver
    gpio_level: int = 0  # last sampled level (0/1)
    prev_gpio_level: int = -1  # sentinel to detect changes

    def _connect_and_start_receiver():  # runs in a worker thread
        nonlocal rx, gpio_rx, cnt_rx
        try:
            tap = Tap(args.device_ip)
            tap.connect(args.tap_name)
            print(f"Connected to {args.tap_name} @ {args.device_ip}")

            rx = ReceiverThread(tap)
            rx.start()

            # ---------------- GPIO tap ----------------
            try:
                tap_g = Tap(args.device_ip)
                tap_g.connect(args.gpio_tap_name)
                print(f"Connected to {args.gpio_tap_name} @ {args.device_ip}")

                gpio_rx = GPIORxThread(tap_g)
                gpio_rx.start()
            except Exception as exc:
                print(f"GPIO tap connection failed: {exc}", file=sys.stderr)

            # --------------- Spike-count tap -----------------
            try:
                tap_c = Tap(args.device_ip)
                tap_c.connect(args.spike_count_tap)
                print(f"Connected to {args.spike_count_tap} @ {args.device_ip}")

                cnt_rx = CountRxThread(tap_c)
                cnt_rx.start()
            except Exception as exc:
                print(f"Spike-count tap connection failed: {exc}", file=sys.stderr)
        except Exception as exc:
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

    # Track latest stats to show under elapsed
    last_stats_update = time.time()
    last_drop: float | None = None
    last_bw: float | None = None
    last_bins_rate: float | None = None

    # Sliding-window counts for GUI-render stats: entries are (timestamp, plotted, dropped)
    gui_stats: deque[tuple[float, int, int]] = deque()

    import numpy as np  # needed for raster arrays

    # Pre-allocate raster buffer (channels × bins)
    raster_buf = np.zeros((MAX_CHANNELS, RASTER_BINS), dtype=np.uint16)
    raster_cursor = 0  # next column to write (0..RASTER_BINS-1)

    def _update_cell_sizes():
        """Adapt column widths and row heights to current window size."""
        nonlocal _last_cell_size

        if n_cols == 0:
            return

        layout = spike_widget.ci.layout  # type: ignore[attr-defined]

        # Compute available size excluding padding.
        avail_w = max(100, container.width() - PAD_W)
        # Subtract raster height to keep waveform grid visible
        avail_h = max(
            100,
            container.height() - stats_widget.height() - raster_plot.height() - PAD_H,
        )

        col_width = int(avail_w / n_cols)
        row_height = int(avail_h / max(1, n_rows))

        if _last_cell_size == (col_width, row_height):
            return  # no change

        # Apply column widths
        for c in range(n_cols):
            try:
                layout.setColumnFixedWidth(c, col_width)
            except Exception:
                pass

        # Apply row heights (including header)
        total_rows = n_rows
        for r in range(total_rows):
            try:
                layout.setRowFixedHeight(r, row_height)
            except Exception:
                pass

        _last_cell_size = (col_width, row_height)

    def update_gui():
        nonlocal current_window_start, first_ts, local_start_time, last_stats_update, last_drop, last_bw, last_bins_rate, WAVEFORM_LEN, PER_WF_LEN, RING_CAPACITY, shared_ring_x, ring_y, ring_head, ring_count, rx, gpio_rx, cnt_rx, gpio_level, prev_gpio_level, gpio_dot, plot_map, raster_buf, raster_cursor, raster_img, cursor_line

        # Receiver not yet connected – nothing to do
        if rx is None:
            return

        # Keep grid columns at equal widths adaptive to any window resize
        _update_cell_sizes()

        batch = rx.pop_batch(MAX_POP_PER_TICK)

        # -----------------------------------------------------------
        # Spike-count raster update (may advance even if no waveforms)
        # -----------------------------------------------------------
        if cnt_rx is not None:
            got = True
            while got:
                item = cnt_rx.pop_next()
                if item is None:
                    got = False
                    break
                _, counts_vec = item
                # Ensure vector length (MAX_CHANNELS rows)
                if counts_vec.size >= MAX_CHANNELS:
                    counts_slice = counts_vec[:MAX_CHANNELS]
                else:
                    counts_slice = np.pad(
                        counts_vec, (0, MAX_CHANNELS - counts_vec.size), "constant"
                    )

                raster_buf[:, raster_cursor] = counts_slice
                # Advance cursor (wrap around)
                raster_cursor = (raster_cursor + 1) % RASTER_BINS

            # Display raster as-is (circular buffer); cursor shows current write position.
            display_img = raster_buf  # channel 0 top row

            # Dynamic level scaling – use max count for colour stretch but
            # avoid degenerate 0 range.
            vmax = int(display_img.max())
            if vmax == 0:
                vmax = 1

            # Force a fresh numpy object each frame so pyqtgraph does not skip
            # the update when the underlying data buffer is mutated in-place.
            # A shallow .copy() is cheap (∼40 kB per frame) and ensures the image
            # refreshes at the full GUI timer rate (~40 Hz).
            raster_img.setImage(
                display_img.copy(),  # new backing array every call
                levels=(0, RASTER_CMAP_MAX),
                autoLevels=False,
                autoRange=False,
            )
            cursor_line.setPos(raster_cursor)

        # If there are no spike batches, we may still update GUI for raster/stats.

        # --- GPIO level from background reader ---------------------------
        if gpio_rx is not None:
            gpio_level = gpio_rx.get_level()
            if gpio_level != prev_gpio_level:
                print(f"GPIO level: {gpio_level}", flush=True)
                prev_gpio_level = gpio_level

        # Update GPIO status dot (0 → bright green, 1 → dim/off)
        color = "#00ff00" if gpio_level == 0 else "#444444"
        gpio_dot.setStyleSheet(f"background-color: {color}; border-radius: 6px;")

        # If there are no spike batches, we are done for this tick.
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

                # Adjust X-axis range of all channel plots to new waveform length
                for _plot in plot_map.values():
                    try:
                        _plot.setXRange(0, WAVEFORM_LEN - 1)
                        _plot.disableAutoRange(axis="x")
                    except Exception:
                        pass

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

        bins_rate = None  # default – will be stored in last_bins_rate when updated

        # Network stats refresh once per second
        if now_wall - last_stats_update >= 1.0:
            last_drop, last_bw = rx.get_stats() if rx else (None, None)
            bins_rate = cnt_rx.get_rate() if cnt_rx else None
            if bins_rate is not None:
                last_bins_rate = bins_rate
            last_stats_update = now_wall

        # Prepare horizontal stats segments for the header row
        segments: list[str] = []
        segments.append(f"Elapsed: {elapsed_sec:0.1f}s")

        if local_start_time is not None:
            wall_elapsed = time.time() - local_start_time
            lag_sec = elapsed_sec - wall_elapsed
            segments.append(f"Lag: {lag_sec:5.2f}s")

        if last_drop is not None:
            segments.append(f"Dropped: {last_drop*100:5.2f}%")
            segments.append(f"Unplotted: {drop_gui_pct*100:5.2f}%")
            if last_bw is not None:
                segments.append(f"BW: {last_bw:5.2f}Mbps")

        if last_bins_rate is not None:
            segments.append(f"Bins/s: {last_bins_rate:4.0f}")

        # Render the stats line (single label spanning columns)
        _update_stats_text(segments)

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

        # ---------------- GPIO indicator ------------------------
        # (gpio_dot updated above)

    timer = QtCore.QTimer()
    try:
        timer.setTimerType(QtCore.Qt.PreciseTimer)  # ensure max resolution (~1 ms)
    except Exception:
        pass  # older PyQt versions

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
        if gpio_rx and gpio_rx.is_alive():
            gpio_rx.stop()
        if cnt_rx and cnt_rx.is_alive():
            cnt_rx.stop()

    QtCore.QCoreApplication.instance().aboutToQuit.connect(_stop_rx)

    try:
        pg.exec()
    finally:
        if rx and rx.is_alive():
            rx.stop()  # stop already joins internally
        if gpio_rx and gpio_rx.is_alive():
            gpio_rx.stop()
        if cnt_rx and cnt_rx.is_alive():
            cnt_rx.stop()


if __name__ == "__main__":
    main()
