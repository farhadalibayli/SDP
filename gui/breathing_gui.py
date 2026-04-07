"""
Biomedical Breathing Feedback Application
==========================================
Receives raw ADC from Arduino, processes signal in Python,
computes all metrics, gives real-time feedback, saves CSV,
and runs ML classification every 20 seconds.

Metrics:
  1. BPM           - Hussain et al. 2023
  2. I:E Ratio     - Jerath et al. 2006
  3. Tidal Volume  - Hoffmann et al. 2011
  4. Variability   - std of breath periods
  5. Apnea         - no breath > 12s

Signal Stability:
  - Double moving average for sensor noise rejection
  - Breath-hold detection: flattens signal when user holds breath
    (post-inhale hold AND post-exhale hold both produce flat line)
  - Baseline frozen during active inhale
  - Threshold frozen during holds
  - All serial samples processed (no data dropped)

ML:
  - Random Forest trained on labeled sessions
  - Classifies: normal / deep / mix
  - Updates every 5 seconds after 20s window
  - Model loaded from breathing_model.pkl
"""

import sys
import time
import csv
import os
from collections import deque

import numpy as np
import serial
import serial.tools.list_ports

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFrame, QMessageBox, QFileDialog
)

try:
    import joblib
    import pandas as pd
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

FEATURE_COLS = [
    'bpm_mean', 'bpm_std', 'bpm_min', 'bpm_max',
    'ie_mean',  'ie_std',
    'tv_mean',  'tv_std',
    'var_mean',
]

ML_LABELS = {
    'normal': ('Normal Breathing',         '#27ae60'),
    'deep':   ('Deep / Relaxed Breathing', '#00d4ff'),
    'mix':    ('Irregular Breathing',      '#f39c12'),
}


# ─────────────────────────────────────────────
#  SIGNAL PROCESSOR
# ─────────────────────────────────────────────

class SignalProcessor:
    FS                = 20      # Hz — Arduino sends at 20 Hz
    MA_N              = 8       # First moving average window
    MA2_N             = 4       # Second moving average window (double smoothing)
    BETA              = 0.008   # Very slow baseline drift correction
    NOISE_ALPHA       = 0.01    # Very slow threshold adaptation
    TH_MULT_UP        = 2.2     # Threshold = this * noise_ema
    TH_MIN            = 1.0     # Minimum threshold floor
    REFRACTORY        = 2.0     # Min seconds between breath detections
    APNEA_SEC         = 12.0    # Seconds without breath before apnea flag
    VAR_WINDOW        = 8       # Number of recent periods for variability
    INIT_SEC          = 4.0     # Calibration period at start — no detection

    # ── Breath-hold flattening ────────────────────────────────────
    # When rolling std of recent signal < HOLD_STD_THRESH → hold detected
    HOLD_STD_THRESH   = 1.2     # ADC units — lower = flatter trigger
    HOLD_WIN          = 10      # Samples for rolling std (0.5 sec @ 20 Hz)
    # During hold: display_x pulled toward x very slowly → flat line
    HOLD_SMOOTH       = 0.04
    # During active breathing: display_x tracks x quickly → responsive
    ACTIVE_SMOOTH     = 0.45

    def __init__(self):
        self.reset()

    def reset(self):
        # ── Smoothing buffers ─────────────────────────────────────
        self._ma_buf        = deque(maxlen=self.MA_N)
        self._ma_sum        = 0.0
        self._ma2_buf       = deque(maxlen=self.MA2_N)

        # ── Baseline ──────────────────────────────────────────────
        self.baseline       = None

        # ── Noise & threshold ─────────────────────────────────────
        self.noise_ema      = 0.0
        self.th_up          = self.TH_MIN
        self.th_dn          = self.TH_MIN * 0.4

        # ── Signal values ─────────────────────────────────────────
        self.x              = 0.0         # centered signal (raw)
        self.display_x      = 0.0         # hold-stabilized signal → plot

        # ── Hold detection ────────────────────────────────────────
        self._hold_win_buf  = deque(maxlen=self.HOLD_WIN)
        self.in_hold        = False

        # ── Breath phase ──────────────────────────────────────────
        self._in_inhale     = False
        self._last_breath_t = None
        self._inhale_start_t= None
        self._peak_t        = None
        self._peak_val      = -1e9

        # ── Cycle tracking ────────────────────────────────────────
        self._last_t        = None
        self._last_x        = 0.0
        self._cycle_max     = -1e9
        self._cycle_min     =  1e9

        # ── Metrics ───────────────────────────────────────────────
        self.bpm            = 0.0
        self.ie_ratio       = None
        self.inhale_time    = None
        self.exhale_time    = None
        self.tidal_volume   = 0.0
        self.variability    = 0.0
        self.apnea          = False
        self._periods       = deque(maxlen=self.VAR_WINDOW)
        self._breath_count  = 0

        # ── Init tracking ─────────────────────────────────────────
        self._start_t       = None

    def push(self, raw: int, t: float):

        # ── Session start ─────────────────────────────────────────
        if self._start_t is None:
            self._start_t = t

        in_init = (t - self._start_t) < self.INIT_SEC

        # ── Stage 1: Double moving average ───────────────────────
        # Two passes of moving average gives a much cleaner signal
        # than a single large window — preserves breath shape better
        if len(self._ma_buf) == self.MA_N:
            self._ma_sum -= self._ma_buf[0]
        self._ma_buf.append(raw)
        self._ma_sum += raw
        S1 = self._ma_sum / len(self._ma_buf)

        self._ma2_buf.append(S1)
        S = sum(self._ma2_buf) / len(self._ma2_buf)

        # ── Stage 2: Baseline estimation ─────────────────────────
        # Rules:
        #   a) No baseline yet → initialize immediately
        #   b) Init period → fast EMA to capture resting pressure fast
        #   c) Active inhale → FREEZE baseline (prevents it chasing signal)
        #   d) Rest / exhale → very slow drift correction
        # This means the elastic-band constant pressure is absorbed into
        # the baseline without distorting the breath signal.
        if self.baseline is None:
            self.baseline = S
        elif in_init:
            self.baseline = 0.15 * S + 0.85 * self.baseline
        elif not self._in_inhale:
            self.baseline = self.BETA * S + (1.0 - self.BETA) * self.baseline
        # else: _in_inhale → baseline is frozen

        x = S - self.baseline
        self.x = x

        # ── Stage 3: Breath-hold detection ───────────────────────
        # Compute rolling std of the last HOLD_WIN x samples.
        # If std is below threshold the signal is stable → hold state.
        # Works for BOTH post-inhale holds and post-exhale holds.
        self._hold_win_buf.append(x)

        if len(self._hold_win_buf) >= max(3, self.HOLD_WIN // 2):
            rolling_std  = float(np.std(list(self._hold_win_buf)))
            self.in_hold = rolling_std < self.HOLD_STD_THRESH
        else:
            self.in_hold = False

        # ── Stage 4: Display signal (hold-stabilized) ────────────
        # During hold  → very slow EMA → flat line on the plot
        # During active → faster EMA   → responsive breath shape
        alpha = self.HOLD_SMOOTH if self.in_hold else self.ACTIVE_SMOOTH
        self.display_x = alpha * x + (1.0 - alpha) * self.display_x

        # ── Stage 5: Noise EMA & adaptive threshold ───────────────
        # Only update noise estimate when:
        #   • not calibrating
        #   • not in a hold (threshold must not collapse during holds)
        #   • signal is actually moving
        dx = abs(x - self._last_x)
        if not in_init and not self.in_hold and dx > 0.1:
            self.noise_ema = ((1.0 - self.NOISE_ALPHA) * self.noise_ema +
                              self.NOISE_ALPHA * abs(x))

        self.th_up = max(self.TH_MIN, self.TH_MULT_UP * self.noise_ema)
        self.th_dn = 0.4 * self.th_up

        # ── Stage 6: Cycle min/max (for tidal volume proxy) ───────
        if x > self._cycle_max: self._cycle_max = x
        if x < self._cycle_min: self._cycle_min = x

        # ── Stage 7: Apnea check ──────────────────────────────────
        if self._last_breath_t is not None:
            self.apnea = (t - self._last_breath_t) > self.APNEA_SEC
        else:
            self.apnea = False

        # ── Stage 8: Breath detection (skip during init) ──────────
        if not in_init:
            if not self._in_inhale:
                ref_ok = (self._last_breath_t is None or
                          (t - self._last_breath_t) > self.REFRACTORY)
                if x > self.th_up and ref_ok:
                    self._in_inhale      = True
                    self._inhale_start_t = t
                    self._peak_t         = t
                    self._peak_val       = x

                    if self._last_breath_t is not None:
                        period = t - self._last_breath_t
                        # Sanity gate: ignore physiologically impossible values
                        if 1.5 < period < 20.0:
                            self._periods.append(period)
                            self.bpm = 60.0 / period
                            if len(self._periods) >= 2:
                                self.variability = float(
                                    np.std(list(self._periods)))

                    self._last_breath_t = t
                    self._breath_count += 1

                    if self._cycle_max > -1e8 and self._cycle_min < 1e8:
                        self.tidal_volume = self._cycle_max - self._cycle_min
                    self._cycle_max = -1e9
                    self._cycle_min =  1e9

            else:
                # Track peak during inhale phase
                if x > self._peak_val:
                    self._peak_val = x
                    self._peak_t   = t

                # Exhale detected when signal drops below lower threshold
                if x < self.th_dn:
                    self._in_inhale = False
                    if (self._inhale_start_t is not None and
                            self._peak_t is not None):
                        inhale_t = max(0.0, self._peak_t - self._inhale_start_t)
                        exhale_t = max(0.0, t - self._peak_t)
                        self.inhale_time = inhale_t
                        self.exhale_time = exhale_t
                        if exhale_t > 0.1:
                            self.ie_ratio = inhale_t / exhale_t

        self._last_t = t
        self._last_x = x


# ─────────────────────────────────────────────
#  FEEDBACK ENGINE
# ─────────────────────────────────────────────

class FeedbackEngine:
    def get(self, bpm, ie_ratio, variability, apnea, breath_count):
        if breath_count < 3:
            return "Calibrating... please breathe normally", "#7f8c8d"
        if apnea:
            return "⚠  APNEA DETECTED — Please breathe", "#e74c3c"

        messages = []
        color    = "#27ae60"

        if bpm > 24:
            messages.append("Breathing too fast — slow down")
            color = "#e74c3c"
        elif bpm > 20:
            messages.append("Slightly fast — try to slow down")
            color = "#f39c12"
        elif bpm < 6:
            messages.append("Breathing very slow — breathe more")
            color = "#e74c3c"
        elif bpm < 12:
            messages.append("Slightly slow — breathe a little more")
            color = "#f39c12"

        if ie_ratio is not None:
            if ie_ratio > 1.5:
                messages.append("Exhale longer than inhale")
                if color == "#27ae60": color = "#e74c3c"
            elif ie_ratio > 1.0:
                messages.append("Try to exhale a bit longer")
                if color == "#27ae60": color = "#f39c12"

        if variability > 2.0:
            messages.append("Irregular breathing — find a rhythm")
            if color == "#27ae60": color = "#f39c12"

        if not messages:
            return "Good breathing pattern ✓", "#27ae60"
        return " | ".join(messages), color


# ─────────────────────────────────────────────
#  ML CLASSIFIER
# ─────────────────────────────────────────────

class MLClassifier:
    WINDOW_SEC = 20.0

    def __init__(self):
        self.model   = None
        self.encoder = None
        self.loaded  = False
        self._load()

    def _load(self):
        if not _ML_AVAILABLE:
            return
        # Look for model next to this script first, then CWD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [os.path.join(script_dir, 'breathing_model.pkl'),
                          'breathing_model.pkl']:
            if os.path.exists(candidate):
                try:
                    bundle       = joblib.load(candidate)
                    self.model   = bundle['model']
                    self.encoder = bundle['encoder']
                    self.loaded  = True
                    print(f"ML model loaded from: {candidate}")
                except Exception as e:
                    print(f"ML model load failed: {e}")
                break

    def predict(self, row_buffer):
        if not self.loaded or len(row_buffer) < 10:
            return None, 0.0
        try:
            df = pd.DataFrame(row_buffer)
            for col in ['bpm', 'tidal_volume', 'variability']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df['ie_ratio'] = pd.to_numeric(df['ie_ratio'], errors='coerce')

            bpm = df['bpm'].values
            ie  = df['ie_ratio'].dropna().values
            tv  = df['tidal_volume'].values
            var = df['variability'].values

            bpm = bpm[bpm > 0]
            tv  = tv[tv  > 0]

            if len(bpm) < 2:
                return None, 0.0

            feats = {
                'bpm_mean': np.mean(bpm),
                'bpm_std':  np.std(bpm),
                'bpm_min':  np.min(bpm),
                'bpm_max':  np.max(bpm),
                'ie_mean':  np.mean(ie)  if len(ie) > 0 else 0.0,
                'ie_std':   np.std(ie)   if len(ie) > 0 else 0.0,
                'tv_mean':  np.mean(tv)  if len(tv) > 0 else 0.0,
                'tv_std':   np.std(tv)   if len(tv) > 0 else 0.0,
                'var_mean': np.mean(var) if len(var) > 0 else 0.0,
            }

            X    = np.array([[feats[c] for c in FEATURE_COLS]])
            pred = self.encoder.inverse_transform(self.model.predict(X))[0]
            conf = float(max(self.model.predict_proba(X)[0])) * 100.0
            return pred, conf
        except Exception as e:
            print(f"ML predict error: {e}")
            return None, 0.0


# ─────────────────────────────────────────────
#  PLOT WIDGET
# ─────────────────────────────────────────────

class PlotWidget(QWidget):
    HISTORY = 400   # ~20 seconds at 20 Hz

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 220)
        self._x     = deque(maxlen=self.HISTORY)   # display_x (stabilized)
        self._th    = deque(maxlen=self.HISTORY)   # threshold
        self._holds = deque(maxlen=self.HISTORY)   # bool: in_hold

    def push(self, x, th, in_hold=False):
        self._x.append(x)
        self._th.append(th)
        self._holds.append(in_hold)
        self.update()

    def clear(self):
        self._x.clear()
        self._th.clear()
        self._holds.clear()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#1a1a2e"))

        if len(self._x) < 2:
            p.setPen(QColor("#aaaaaa"))
            p.setFont(QFont("Courier", 13))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Waiting for signal...")
            p.end()
            return

        r = self.rect().adjusted(12, 20, -12, -12)
        W, H, n = r.width(), r.height(), len(self._x)

        # ── Auto-scaling y-axis (no ceiling — fully limitless) ────
        allv = list(self._x) + list(self._th) + [0.0]
        vmin, vmax = min(allv), max(allv)
        rng = vmax - vmin
        if rng < 0.5:
            rng = 0.5
        vmin -= 0.18 * rng
        vmax += 0.18 * rng
        rng = vmax - vmin

        def ym(v):
            return r.top() + (1.0 - (v - vmin) / rng) * H

        step = W / max(n - 1, 1)

        # ── Zero line ─────────────────────────────────────────────
        p.setPen(QPen(QColor("#2a2a4a"), 1, Qt.PenStyle.DashLine))
        p.drawLine(r.left(), int(ym(0.0)), r.right(), int(ym(0.0)))

        # ── Threshold line ────────────────────────────────────────
        p.setPen(QPen(QColor("#c0392b"), 1, Qt.PenStyle.DotLine))
        pts = list(self._th)
        for i in range(1, len(pts)):
            p.drawLine(int(r.left() + (i-1)*step), int(ym(pts[i-1])),
                       int(r.left() +  i   *step), int(ym(pts[i])))

        # ── Signal line ───────────────────────────────────────────
        # Active breathing  → cyan, width 2
        # Hold / stable     → grey, width 1 (clearly distinguishable)
        pts   = list(self._x)
        holds = list(self._holds)
        for i in range(1, len(pts)):
            is_hold = holds[i] or holds[i-1]
            if is_hold:
                p.setPen(QPen(QColor("#666688"), 1))
            else:
                p.setPen(QPen(QColor("#00d4ff"), 2))
            p.drawLine(int(r.left() + (i-1)*step), int(ym(pts[i-1])),
                       int(r.left() +  i   *step), int(ym(pts[i])))

        # ── Legend ────────────────────────────────────────────────
        p.setPen(QColor("#888899"))
        p.setFont(QFont("Courier", 9))
        p.drawText(
            r.adjusted(4, -16, 0, 0),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            "— signal (active)    — signal (hold)    ··· threshold")
        p.end()


# ─────────────────────────────────────────────
#  METRIC CARD
# ─────────────────────────────────────────────

class MetricCard(QFrame):
    def __init__(self, title, unit=""):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame{background:#16213e;border-radius:8px;"
                           "border:1px solid #0f3460;}")
        lay = QVBoxLayout(self)
        lay.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color:#7f8c8d;font-size:11px;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._val = QLabel("--")
        self._val.setStyleSheet(
            "color:#00d4ff;font-size:22px;font-weight:bold;")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_unit = QLabel(unit)
        lbl_unit.setStyleSheet("color:#555577;font-size:10px;")
        lbl_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(lbl_title)
        lay.addWidget(self._val)
        lay.addWidget(lbl_unit)

    def set_value(self, v, color="#00d4ff"):
        self._val.setText(str(v))
        self._val.setStyleSheet(
            f"color:{color};font-size:22px;font-weight:bold;")


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────

class MainWindow(QWidget):
    ML_WINDOW_SEC = 20.0
    ML_STEP_SEC   =  5.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Breathing Feedback — Biomedical Project")
        self.setStyleSheet("background-color:#0d0d1a;color:#e0e0e0;")

        self.ser           = None
        self.running       = False
        self.processor     = SignalProcessor()
        self.feedback      = FeedbackEngine()
        self.ml            = MLClassifier()
        self.csv_file      = None
        self.csv_writer    = None
        self.session_start = None

        self._ml_buffer = deque()
        self._last_ml_t = 0.0

        self.timer = QTimer(self)
        self.timer.setInterval(40)   # 25 Hz UI refresh
        self.timer.timeout.connect(self._tick)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        btn_style = ("QPushButton{background:#0f3460;color:#e0e0e0;"
                     "border-radius:5px;padding:6px 14px;}"
                     "QPushButton:hover{background:#1a5276;}")

        # ── Top bar ───────────────────────────────────────────────
        top = QHBoxLayout()

        self.portBox = QComboBox()
        self.portBox.setStyleSheet(
            "background:#16213e;color:#e0e0e0;border:1px solid #0f3460;"
            "padding:4px;border-radius:4px;min-width:220px;")

        self.refreshBtn   = QPushButton("⟳ Ports")
        self.startStopBtn = QPushButton("▶  Start")
        self.csvBtn       = QPushButton("💾  Save CSV")
        self.clearBtn     = QPushButton("✕  Clear")

        for b in (self.refreshBtn, self.startStopBtn,
                  self.csvBtn, self.clearBtn):
            b.setStyleSheet(btn_style)

        self.statusDot = QLabel("●")
        self.statusDot.setStyleSheet("color:#e74c3c;font-size:18px;")

        self.holdLabel = QLabel("")
        self.holdLabel.setStyleSheet(
            "color:#888888;font-size:11px;font-weight:bold;")

        top.addWidget(QLabel("Port:"))
        top.addWidget(self.portBox, 1)
        top.addWidget(self.refreshBtn)
        top.addSpacing(10)
        top.addWidget(self.startStopBtn)
        top.addWidget(self.csvBtn)
        top.addWidget(self.clearBtn)
        top.addSpacing(8)
        top.addWidget(self.statusDot)
        top.addSpacing(6)
        top.addWidget(self.holdLabel)

        # ── Plot ──────────────────────────────────────────────────
        self.plot = PlotWidget()

        # ── Metric cards ──────────────────────────────────────────
        self.card_bpm = MetricCard("BREATHING RATE",  "breaths / min")
        self.card_ie  = MetricCard("I:E RATIO",       "inhale / exhale")
        self.card_tv  = MetricCard("TIDAL VOLUME",    "relative units")
        self.card_var = MetricCard("VARIABILITY",     "std periods (s)")
        self.card_ap  = MetricCard("APNEA",           "")

        cards = QHBoxLayout()
        cards.setSpacing(8)
        for c in (self.card_bpm, self.card_ie, self.card_tv,
                  self.card_var, self.card_ap):
            cards.addWidget(c)

        # ── Rule-based feedback ───────────────────────────────────
        self.feedbackLabel = QLabel("Connect sensor to begin")
        self.feedbackLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedbackLabel.setStyleSheet(
            "background:#16213e;color:#7f8c8d;font-size:14px;"
            "font-weight:bold;border-radius:8px;padding:10px;"
            "border:1px solid #0f3460;")
        self.feedbackLabel.setMinimumHeight(44)

        # ── ML panel ──────────────────────────────────────────────
        ml_frame = QFrame()
        ml_frame.setStyleSheet(
            "QFrame{background:#0f1a2e;border-radius:8px;"
            "border:1px solid #1a3460;}")
        ml_lay = QHBoxLayout(ml_frame)
        ml_lay.setContentsMargins(12, 8, 12, 8)

        ml_title = QLabel("ML PATTERN:")
        ml_title.setStyleSheet(
            "color:#7f8c8d;font-size:11px;font-weight:bold;")

        self.mlLabel = QLabel("Waiting for data...")
        self.mlLabel.setStyleSheet(
            "color:#555577;font-size:14px;font-weight:bold;")

        self.mlConfLabel = QLabel("")
        self.mlConfLabel.setStyleSheet("color:#555577;font-size:11px;")

        ml_status = QLabel(
            "✓ Model loaded" if self.ml.loaded
            else "✗ No model (run breathing_ml.py --train first)")
        ml_status.setStyleSheet(
            f"color:{'#27ae60' if self.ml.loaded else '#e74c3c'};"
            f"font-size:10px;")

        ml_lay.addWidget(ml_title)
        ml_lay.addWidget(self.mlLabel, 1)
        ml_lay.addWidget(self.mlConfLabel)
        ml_lay.addStretch()
        ml_lay.addWidget(ml_status)

        # ── Status bar ────────────────────────────────────────────
        self.statusBar = QLabel("Not connected")
        self.statusBar.setStyleSheet("color:#555577;font-size:10px;")

        root.addLayout(top)
        root.addWidget(self.plot)
        root.addLayout(cards)
        root.addWidget(self.feedbackLabel)
        root.addWidget(ml_frame)
        root.addWidget(self.statusBar)

        self.refreshBtn.clicked.connect(self._refresh_ports)
        self.startStopBtn.clicked.connect(self._toggle_run)
        self.csvBtn.clicked.connect(self._toggle_csv)
        self.clearBtn.clicked.connect(self._clear)
        self._refresh_ports()

    # ── Port management ───────────────────────────────────────────

    def _refresh_ports(self):
        self.portBox.clear()
        for p in serial.tools.list_ports.comports():
            self.portBox.addItem(
                f"{p.device} — {p.description}", p.device)
        if self.portBox.count() == 0:
            self.portBox.addItem("No ports found", None)

    # ── Start / Stop ──────────────────────────────────────────────

    def _toggle_run(self):
        if not self.running:
            port = self.portBox.currentData()
            if not port:
                QMessageBox.warning(self, "No port", "Select a COM port.")
                return
            try:
                self.ser = serial.Serial(port, 9600, timeout=0.1)
                time.sleep(1.5)
                self.ser.reset_input_buffer()
            except Exception as e:
                QMessageBox.critical(self, "Connection failed", str(e))
                return

            self.running       = True
            self.session_start = time.time()
            self.processor.reset()
            self._ml_buffer.clear()
            self._last_ml_t = 0.0

            self.statusDot.setStyleSheet("color:#27ae60;font-size:18px;")
            self.startStopBtn.setText("■  Stop")
            self.statusBar.setText(
                f"Connected to {port} @ 9600 baud — 20 Hz — "
                f"Calibrating {SignalProcessor.INIT_SEC:.0f}s...")
            self.timer.start()
        else:
            self._stop()

    def _stop(self):
        self.timer.stop()
        self.running = False
        self.statusDot.setStyleSheet("color:#e74c3c;font-size:18px;")
        self.startStopBtn.setText("▶  Start")
        self.holdLabel.setText("")
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self._close_csv()
        self.statusBar.setText("Disconnected")

    # ── CSV ───────────────────────────────────────────────────────

    def _toggle_csv(self):
        if self.csv_file is not None:
            self._close_csv()
            self.csvBtn.setText("💾  Save CSV")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save breathing data", "breathing_session.csv",
            "CSV files (*.csv)")
        if not path:
            return
        self.csv_file   = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "time_s", "raw_adc", "bpm", "ie_ratio",
            "tidal_volume", "variability", "apnea", "in_hold"])
        self.csvBtn.setText("⏹  Stop CSV")
        self.statusBar.setText(
            f"Recording to {os.path.basename(path)}")

    def _close_csv(self):
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
            self.csv_file   = None
            self.csv_writer = None

    # ── Clear ─────────────────────────────────────────────────────

    def _clear(self):
        self.processor.reset()
        self._ml_buffer.clear()
        self.plot.clear()
        for c in (self.card_bpm, self.card_ie, self.card_tv,
                  self.card_var, self.card_ap):
            c.set_value("--")
        self.mlLabel.setText("Waiting for data...")
        self.mlLabel.setStyleSheet(
            "color:#555577;font-size:14px;font-weight:bold;")
        self.mlConfLabel.setText("")
        self.feedbackLabel.setText("Sensor cleared — press Start")
        self.holdLabel.setText("")

    # ── Main tick ─────────────────────────────────────────────────

    def _tick(self):
        if not self.ser:
            return
        try:
            # Collect ALL samples that arrived since last tick.
            # No data is dropped — every sample is processed with a
            # back-calculated timestamp so timing stays accurate.
            samples = []
            while self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line.lstrip("-").isdigit():
                    samples.append(int(line))

            if not samples:
                return

            t_now          = time.time() - self.session_start
            dt_per_sample  = 1.0 / SignalProcessor.FS

            for i, raw_val in enumerate(samples):
                t = t_now - (len(samples) - 1 - i) * dt_per_sample
                self.processor.push(raw_val, t)

            pr = self.processor
            t  = t_now

            # ── Plot ──────────────────────────────────────────────
            self.plot.push(pr.display_x, pr.th_up, pr.in_hold)

            # ── Hold indicator ────────────────────────────────────
            if pr.in_hold:
                self.holdLabel.setText("● HOLD")
            else:
                self.holdLabel.setText("")

            # ── Status bar during calibration ─────────────────────
            if pr._start_t is not None:
                elapsed = t
                if elapsed < SignalProcessor.INIT_SEC:
                    remaining = SignalProcessor.INIT_SEC - elapsed
                    self.statusBar.setText(
                        f"Calibrating baseline — {remaining:.1f}s remaining"
                        " — breathe normally")
                else:
                    self.statusBar.setText(
                        f"Running — {t:.0f}s  |  "
                        f"Breaths: {pr._breath_count}  |  "
                        f"Hold: {'YES' if pr.in_hold else 'NO'}")

            # ── Metric cards ──────────────────────────────────────
            if pr.bpm > 0:
                bpm_color = "#27ae60"
                if   pr.bpm > 24 or pr.bpm < 6:  bpm_color = "#e74c3c"
                elif pr.bpm > 20 or pr.bpm < 12: bpm_color = "#f39c12"
                self.card_bpm.set_value(f"{pr.bpm:.1f}", bpm_color)
            else:
                self.card_bpm.set_value("--")

            if pr.ie_ratio is not None:
                ie_color = "#27ae60" if pr.ie_ratio <= 1.0 else "#e74c3c"
                self.card_ie.set_value(f"{pr.ie_ratio:.2f}", ie_color)
            else:
                self.card_ie.set_value("--")

            self.card_tv.set_value(f"{pr.tidal_volume:.1f}")

            var_color = "#27ae60" if pr.variability < 1.0 else "#f39c12"
            self.card_var.set_value(f"{pr.variability:.2f}", var_color)

            self.card_ap.set_value(
                "YES" if pr.apnea else "NO",
                "#e74c3c" if pr.apnea else "#27ae60")

            # ── Feedback ──────────────────────────────────────────
            in_init = (pr._start_t is not None and
                       t < SignalProcessor.INIT_SEC)
            if in_init:
                msg   = "Calibrating baseline... breathe normally"
                color = "#f39c12"
            else:
                msg, color = self.feedback.get(
                    pr.bpm, pr.ie_ratio, pr.variability,
                    pr.apnea, pr._breath_count)

            self.feedbackLabel.setText(msg)
            self.feedbackLabel.setStyleSheet(
                f"background:#16213e;color:{color};font-size:14px;"
                f"font-weight:bold;border-radius:8px;padding:10px;"
                f"border:2px solid {color};")

            # ── CSV ───────────────────────────────────────────────
            row = {
                "time_s":       f"{t:.3f}",
                "raw_adc":      samples[-1],
                "bpm":          f"{pr.bpm:.2f}",
                "ie_ratio":     (f"{pr.ie_ratio:.3f}"
                                 if pr.ie_ratio is not None else ""),
                "tidal_volume": f"{pr.tidal_volume:.2f}",
                "variability":  f"{pr.variability:.3f}",
                "apnea":        int(pr.apnea),
                "in_hold":      int(pr.in_hold),
            }
            if self.csv_writer:
                self.csv_writer.writerow(list(row.values()))
                self.csv_file.flush()

            # ── ML buffer (rolling 20s window) ────────────────────
            self._ml_buffer.append(row)
            cutoff = t - self.ML_WINDOW_SEC
            while (self._ml_buffer and
                   float(self._ml_buffer[0]["time_s"]) < cutoff):
                self._ml_buffer.popleft()

            # ── ML prediction (every 5s, after 20s of data) ───────
            if (self.ml.loaded and
                    t - self._last_ml_t >= self.ML_STEP_SEC and
                    len(self._ml_buffer) > 20):
                self._last_ml_t = t
                label, conf = self.ml.predict(list(self._ml_buffer))
                if label is not None:
                    name, ml_color = ML_LABELS.get(
                        label, (label.upper(), "#00d4ff"))
                    self.mlLabel.setText(name)
                    self.mlLabel.setStyleSheet(
                        f"color:{ml_color};font-size:14px;"
                        f"font-weight:bold;")
                    self.mlConfLabel.setText(f"confidence: {conf:.0f}%")
                    self.mlConfLabel.setStyleSheet(
                        f"color:{ml_color};font-size:11px;")

        except serial.SerialException as e:
            # Real hardware disconnect → stop
            self.statusBar.setText(f"Serial error: {e}")
            self._stop()
        except Exception as e:
            # Non-fatal glitch → log and keep running
            self.statusBar.setText(f"Warning: {e}")

    def closeEvent(self, _):
        self._stop()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1100, 700)
    w.show()
    sys.exit(app.exec())