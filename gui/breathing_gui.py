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
  5. Apnea         - no breath > 10s

ML:
  - Random Forest trained on labeled sessions
  - Classifies: normal / deep / mix
  - Updates every 20 seconds
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
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QFrame, QMessageBox, QFileDialog
)

# Try to load ML model
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
    'normal': ('Normal Breathing',       '#27ae60'),
    'deep':   ('Deep / Relaxed Breathing', '#00d4ff'),
    'mix':    ('Irregular Breathing',    '#f39c12'),
}

# ─────────────────────────────────────────────
#  SIGNAL PROCESSOR
# ─────────────────────────────────────────────

class SignalProcessor:
    FS          = 20
    MA_N        = 5
    BETA        = 0.05
    NOISE_ALPHA = 0.05
    TH_MULT_UP  = 2.0
    TH_MIN      = 1.0
    REFRACTORY  = 0.9
    APNEA_SEC   = 10.0
    VAR_WINDOW  = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self._ma_buf        = deque(maxlen=self.MA_N)
        self._ma_sum        = 0.0
        self.baseline       = None
        self.noise_ema      = 1.0
        self.x              = 0.0
        self.th_up          = self.TH_MIN
        self.th_dn          = self.TH_MIN * 0.5
        self._in_inhale     = False
        self._last_breath_t = None
        self._inhale_start_t= None
        self._peak_t        = None
        self._peak_val      = -1e9
        self._cycle_area    = 0.0
        self._last_t        = None
        self._last_x        = 0.0
        self._cycle_max     = -1e9
        self._cycle_min     =  1e9
        self.bpm            = 0.0
        self.ie_ratio       = None
        self.inhale_time    = None
        self.exhale_time    = None
        self.tidal_volume   = 0.0
        self.variability    = 0.0
        self.apnea          = False
        self._periods       = deque(maxlen=self.VAR_WINDOW)
        self._breath_count  = 0

    def push(self, raw: int, t: float):
        if len(self._ma_buf) == self.MA_N:
            self._ma_sum -= self._ma_buf[0]
        self._ma_buf.append(raw)
        self._ma_sum += raw
        S = self._ma_sum / len(self._ma_buf)

        if self.baseline is None:
            self.baseline = S
        else:
            self.baseline = self.BETA * S + (1.0 - self.BETA) * self.baseline

        x = S - self.baseline
        self.x = x

        self.noise_ema = (1.0 - self.NOISE_ALPHA) * self.noise_ema + self.NOISE_ALPHA * abs(x)
        self.th_up = max(self.TH_MIN, self.TH_MULT_UP * self.noise_ema)
        self.th_dn = 0.5 * self.th_up

        if x > self._cycle_max: self._cycle_max = x
        if x < self._cycle_min: self._cycle_min = x

        if self._last_breath_t is not None:
            self.apnea = (t - self._last_breath_t) > self.APNEA_SEC
        else:
            self.apnea = False

        dt = (t - self._last_t) if self._last_t is not None else 0.0

        if not self._in_inhale:
            ref_ok = (self._last_breath_t is None or
                      (t - self._last_breath_t) > self.REFRACTORY)
            if x > self.th_up and ref_ok:
                self._in_inhale       = True
                self._inhale_start_t  = t
                self._peak_t          = t
                self._peak_val        = x
                self._cycle_area      = 0.0

                if self._last_breath_t is not None:
                    period = t - self._last_breath_t
                    self._periods.append(period)
                    self.bpm = 60.0 / period
                    if len(self._periods) >= 2:
                        self.variability = float(np.std(list(self._periods)))

                self._last_breath_t = t
                self._breath_count += 1

                if self._cycle_max > -1e8 and self._cycle_min < 1e8:
                    self.tidal_volume = self._cycle_max - self._cycle_min
                self._cycle_max = -1e9
                self._cycle_min =  1e9
        else:
            if x > self._peak_val:
                self._peak_val = x
                self._peak_t   = t
            if dt > 0:
                a0 = max(0.0, self._last_x)
                a1 = max(0.0, x)
                self._cycle_area += 0.5 * (a0 + a1) * dt
            if x < self.th_dn:
                self._in_inhale = False
                if self._inhale_start_t is not None and self._peak_t is not None:
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
        color = "#27ae60"

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
                messages.append("Exhale longer than you inhale")
                if color == "#27ae60": color = "#e74c3c"
            elif ie_ratio > 1.0:
                messages.append("Try to exhale a bit longer")
                if color == "#27ae60": color = "#f39c12"

        if variability > 2.0:
            messages.append("Irregular breathing — try to find a rhythm")
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
        model_path = 'breathing_model.pkl'
        if os.path.exists(model_path):
            try:
                bundle       = joblib.load(model_path)
                self.model   = bundle['model']
                self.encoder = bundle['encoder']
                self.loaded  = True
            except Exception as e:
                print(f"ML model load failed: {e}")

    def predict(self, row_buffer):
        """
        row_buffer: list of dicts with keys matching CSV columns
        Returns (label, confidence) or (None, 0)
        """
        if not self.loaded or len(row_buffer) < 10:
            return None, 0.0

        df = pd.DataFrame(row_buffer)
        for col in ['bpm', 'tidal_volume', 'variability']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['ie_ratio'] = pd.to_numeric(df['ie_ratio'], errors='coerce')

        bpm = df['bpm'].values
        ie  = df['ie_ratio'].dropna().values
        tv  = df['tidal_volume'].values
        var = df['variability'].values

        bpm = bpm[bpm > 0]
        tv  = tv[tv > 0]

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

        X     = np.array([[feats[c] for c in FEATURE_COLS]])
        pred  = self.encoder.inverse_transform(self.model.predict(X))[0]
        proba = self.model.predict_proba(X)[0]
        conf  = float(max(proba)) * 100.0
        return pred, conf


# ─────────────────────────────────────────────
#  PLOT WIDGET
# ─────────────────────────────────────────────

class PlotWidget(QWidget):
    HISTORY = 400

    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 200)
        self._x  = deque(maxlen=self.HISTORY)
        self._th = deque(maxlen=self.HISTORY)

    def push(self, x, th):
        self._x.append(x)
        self._th.append(th)
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

        r = self.rect().adjusted(12, 12, -12, -12)
        W, H, n = r.width(), r.height(), len(self._x)

        allv = list(self._x) + list(self._th) + [0.0]
        vmin, vmax = min(allv), max(allv)
        rng = max(vmax - vmin, 1.0)
        vmin -= 0.1 * rng
        vmax += 0.1 * rng
        rng = vmax - vmin

        def ym(v): return r.top() + (1.0 - (v - vmin) / rng) * H

        p.setPen(QPen(QColor("#444466"), 1, Qt.PenStyle.DashLine))
        p.drawLine(r.left(), int(ym(0.0)), r.right(), int(ym(0.0)))

        step = W / (n - 1)
        p.setPen(QPen(QColor("#e74c3c"), 1, Qt.PenStyle.DotLine))
        pts = list(self._th)
        for i in range(1, len(pts)):
            p.drawLine(int(r.left()+(i-1)*step), int(ym(pts[i-1])),
                       int(r.left()+i*step),     int(ym(pts[i])))

        p.setPen(QPen(QColor("#00d4ff"), 2))
        pts = list(self._x)
        for i in range(1, len(pts)):
            p.drawLine(int(r.left()+(i-1)*step), int(ym(pts[i-1])),
                       int(r.left()+i*step),     int(ym(pts[i])))

        p.setPen(QColor("#aaaaaa"))
        p.setFont(QFont("Courier", 9))
        p.drawText(r.adjusted(4, 2, 0, 0),
                   Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                   "— signal    - - threshold")
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
        self._title = QLabel(title)
        self._title.setStyleSheet("color:#7f8c8d;font-size:11px;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val = QLabel("--")
        self._val.setStyleSheet("color:#00d4ff;font-size:22px;font-weight:bold;")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unit = QLabel(unit)
        self._unit.setStyleSheet("color:#555577;font-size:10px;")
        self._unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._title)
        lay.addWidget(self._val)
        lay.addWidget(self._unit)

    def set_value(self, v, color="#00d4ff"):
        self._val.setText(str(v))
        self._val.setStyleSheet(
            f"color:{color};font-size:22px;font-weight:bold;")


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────

class MainWindow(QWidget):
    ML_WINDOW_SEC = 20.0   # collect this many seconds before ML prediction
    ML_STEP_SEC   = 5.0    # re-predict every N seconds

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

        # Rolling buffer for ML (stores row dicts)
        self._ml_buffer    = deque()
        self._last_ml_t    = 0.0

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Top bar ───────────────────────────────────────────────
        top = QHBoxLayout()
        self.portBox = QComboBox()
        self.portBox.setStyleSheet(
            "background:#16213e;color:#e0e0e0;border:1px solid #0f3460;"
            "padding:4px;border-radius:4px;")

        btn = ("QPushButton{background:#0f3460;color:#e0e0e0;"
               "border-radius:5px;padding:6px 14px;}"
               "QPushButton:hover{background:#1a5276;}")

        self.refreshBtn   = QPushButton("⟳ Ports")
        self.startStopBtn = QPushButton("▶  Start")
        self.csvBtn       = QPushButton("💾  Save CSV")
        self.clearBtn     = QPushButton("✕  Clear")
        self.statusDot    = QLabel("●")
        self.statusDot.setStyleSheet("color:#e74c3c;font-size:18px;")

        for b in (self.refreshBtn, self.startStopBtn,
                  self.csvBtn, self.clearBtn):
            b.setStyleSheet(btn)

        top.addWidget(QLabel("Port:"))
        top.addWidget(self.portBox, 1)
        top.addWidget(self.refreshBtn)
        top.addSpacing(10)
        top.addWidget(self.startStopBtn)
        top.addWidget(self.csvBtn)
        top.addWidget(self.clearBtn)
        top.addSpacing(8)
        top.addWidget(self.statusDot)

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

        # ── ML prediction panel ───────────────────────────────────
        ml_frame = QFrame()
        ml_frame.setStyleSheet(
            "QFrame{background:#0f1a2e;border-radius:8px;"
            "border:1px solid #1a3460;}")
        ml_lay = QHBoxLayout(ml_frame)
        ml_lay.setContentsMargins(12, 8, 12, 8)

        ml_title = QLabel("ML PATTERN:")
        ml_title.setStyleSheet("color:#7f8c8d;font-size:11px;font-weight:bold;")

        self.mlLabel = QLabel("Waiting for data...")
        self.mlLabel.setStyleSheet(
            "color:#555577;font-size:14px;font-weight:bold;")

        self.mlConfLabel = QLabel("")
        self.mlConfLabel.setStyleSheet("color:#555577;font-size:11px;")

        ml_status = QLabel(
            "✓ Model loaded" if self.ml.loaded else "✗ No model (run --train first)")
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

    def _refresh_ports(self):
        self.portBox.clear()
        for p in serial.tools.list_ports.comports():
            self.portBox.addItem(f"{p.device} — {p.description}", p.device)
        if self.portBox.count() == 0:
            self.portBox.addItem("No ports found", None)

    def _toggle_run(self):
        if not self.running:
            port = self.portBox.currentData()
            if not port:
                QMessageBox.warning(self, "No port", "Select a COM port.")
                return
            try:
                self.ser = serial.Serial(port, 9600, timeout=0.5)
                time.sleep(1.5)
                self.ser.reset_input_buffer()
            except Exception as e:
                QMessageBox.critical(self, "Connection failed", str(e))
                return
            self.running = True
            self.processor.reset()
            self._ml_buffer.clear()
            self._last_ml_t = 0.0
            self.session_start = time.time()
            self.statusDot.setStyleSheet("color:#27ae60;font-size:18px;")
            self.startStopBtn.setText("■  Stop")
            self.statusBar.setText(f"Connected to {port} @ 9600 — 20 Hz")
            self.timer.start()
        else:
            self._stop()

    def _stop(self):
        self.timer.stop()
        self.running = False
        self.statusDot.setStyleSheet("color:#e74c3c;font-size:18px;")
        self.startStopBtn.setText("▶  Start")
        if self.ser:
            try: self.ser.close()
            except: pass
            self.ser = None
        self._close_csv()
        self.statusBar.setText("Disconnected")

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
            "time_s", "raw_x", "bpm", "ie_ratio",
            "tidal_volume", "variability", "apnea"])
        self.csvBtn.setText("⏹  Stop CSV")
        self.statusBar.setText(f"Recording to {os.path.basename(path)}")

    def _close_csv(self):
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
            self.csv_file = None
            self.csv_writer = None

    def _clear(self):
        self.processor.reset()
        self._ml_buffer.clear()
        self.plot._x.clear()
        self.plot._th.clear()
        self.plot.update()
        for c in (self.card_bpm, self.card_ie, self.card_tv,
                  self.card_var, self.card_ap):
            c.set_value("--")
        self.mlLabel.setText("Waiting for data...")
        self.mlLabel.setStyleSheet(
            "color:#555577;font-size:14px;font-weight:bold;")
        self.mlConfLabel.setText("")

    def _tick(self):
        if not self.ser:
            return
        try:
            latest_raw = None
            while self.ser.in_waiting:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line.lstrip("-").isdigit():
                    latest_raw = int(line)

            if latest_raw is None:
                return

            t  = time.time() - self.session_start
            pr = self.processor
            pr.push(latest_raw, t)

            # Plot
            self.plot.push(pr.x, pr.th_up)

            # Metric cards
            bpm_color = "#27ae60"
            if pr.bpm > 24 or pr.bpm < 6:   bpm_color = "#e74c3c"
            elif pr.bpm > 20 or pr.bpm < 12: bpm_color = "#f39c12"
            self.card_bpm.set_value(
                f"{pr.bpm:.1f}" if pr.bpm > 0 else "--", bpm_color)

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

            # Rule-based feedback
            msg, color = self.feedback.get(
                pr.bpm, pr.ie_ratio, pr.variability,
                pr.apnea, pr._breath_count)
            self.feedbackLabel.setText(msg)
            self.feedbackLabel.setStyleSheet(
                f"background:#16213e;color:{color};font-size:14px;"
                f"font-weight:bold;border-radius:8px;padding:10px;"
                f"border:2px solid {color};")

            # CSV row
            row = {
                "time_s":       f"{t:.3f}",
                "raw_x":        latest_raw,
                "bpm":          f"{pr.bpm:.2f}",
                "ie_ratio":     f"{pr.ie_ratio:.3f}" if pr.ie_ratio else "",
                "tidal_volume": f"{pr.tidal_volume:.2f}",
                "variability":  f"{pr.variability:.3f}",
                "apnea":        int(pr.apnea),
            }
            if self.csv_writer:
                self.csv_writer.writerow(list(row.values()))

            # ML buffer — keep last ML_WINDOW_SEC seconds
            self._ml_buffer.append(row)
            cutoff = t - self.ML_WINDOW_SEC
            while self._ml_buffer and \
                  float(self._ml_buffer[0]["time_s"]) < cutoff:
                self._ml_buffer.popleft()

            # ML prediction every ML_STEP_SEC seconds
            if (self.ml.loaded and
                    t - self._last_ml_t >= self.ML_STEP_SEC and
                    len(self._ml_buffer) > 20):
                self._last_ml_t = t
                label, conf = self.ml.predict(list(self._ml_buffer))
                if label is not None:
                    name, ml_color = ML_LABELS.get(
                        label, (label.upper(), "#00d4ff"))
                    self.mlLabel.setText(f"{name}")
                    self.mlLabel.setStyleSheet(
                        f"color:{ml_color};font-size:14px;"
                        f"font-weight:bold;")
                    self.mlConfLabel.setText(f"confidence: {conf:.0f}%")
                    self.mlConfLabel.setStyleSheet(
                        f"color:{ml_color};font-size:11px;")

        except Exception as e:
            self.statusBar.setText(f"Error: {e}")
            self._stop()

    def closeEvent(self, _):
        self._stop()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.resize(1050, 660)
    w.show()
    sys.exit(app.exec())