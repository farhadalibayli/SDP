"""
Breathing Pattern ML Classifier
=================================
Step 1: Train a Random Forest classifier on labeled CSV sessions
Step 2: Save the trained model
Step 3: Load model and classify breathing in real time

Usage:
  python breathing_ml.py --train    # train and save model
  python breathing_ml.py --predict  # load model and predict in real time (console)

Requirements:
  pip install scikit-learn pandas numpy joblib
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
from collections import deque

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
#  We extract features from a rolling window
#  of rows, not single samples
# ─────────────────────────────────────────────

WINDOW_SEC  = 20.0    # seconds of data per feature window
STEP_SEC    = 5.0     # slide step

def extract_features(df_window):
    """
    Given a DataFrame window, return a feature dict.
    Features chosen because they differ across breathing types
    and are justified by literature.
    """
    bpm  = df_window['bpm'].values
    ie   = pd.to_numeric(df_window['ie_ratio'], errors='coerce').dropna().values
    tv   = df_window['tidal_volume'].values
    var  = df_window['variability'].values

    bpm  = bpm[bpm > 0]
    tv   = tv[tv > 0]

    if len(bpm) < 2:
        return None

    features = {
        # BPM features (Hussain et al. 2023)
        'bpm_mean':     np.mean(bpm),
        'bpm_std':      np.std(bpm),
        'bpm_min':      np.min(bpm),
        'bpm_max':      np.max(bpm),

        # I:E ratio (Jerath et al. 2006)
        'ie_mean':      np.mean(ie)   if len(ie) > 0 else 0.0,
        'ie_std':       np.std(ie)    if len(ie) > 0 else 0.0,

        # Tidal volume proxy (Hoffmann et al. 2011)
        'tv_mean':      np.mean(tv)   if len(tv) > 0 else 0.0,
        'tv_std':       np.std(tv)    if len(tv) > 0 else 0.0,

        # Variability
        'var_mean':     np.mean(var)  if len(var) > 0 else 0.0,
    }
    return features

FEATURE_COLS = [
    'bpm_mean', 'bpm_std', 'bpm_min', 'bpm_max',
    'ie_mean',  'ie_std',
    'tv_mean',  'tv_std',
    'var_mean',
]


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────

def train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # ── Load CSV files ─────────────────────────────────────────
    # Put your CSV files in same folder as this script
    # Format: filename → label name
    csv_files = {
    'breathing_session_normal.csv': 'normal',
    'breathing_session_deep.csv':   'deep',
    'breathing_session_mix.csv':    'mix',
     }

    all_features = []
    all_labels   = []

    for filename, label in csv_files.items():
        if not os.path.exists(filename):
            print(f"WARNING: {filename} not found, skipping.")
            continue

        df = pd.read_csv(filename)
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
        df = df.dropna(subset=['time_s']).reset_index(drop=True)

        t_start = df['time_s'].iloc[0]
        t_end   = df['time_s'].iloc[-1]

        # Sliding window feature extraction
        t = t_start
        while t + WINDOW_SEC <= t_end:
            window = df[(df['time_s'] >= t) & (df['time_s'] < t + WINDOW_SEC)]
            feats  = extract_features(window)
            if feats:
                all_features.append(feats)
                all_labels.append(label)
            t += STEP_SEC

        print(f"  {label:10s}: {len(df)} rows, "
              f"{t_end - t_start:.0f}s, "
              f"{sum(1 for l in all_labels if l == label)} windows extracted")

    if not all_features:
        print("No features extracted. Check CSV file paths.")
        return

    X = pd.DataFrame(all_features)[FEATURE_COLS].values
    y = np.array(all_labels)

    print(f"\nTotal windows: {len(X)}")
    print(f"Label counts : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Train ──────────────────────────────────────────────────
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_))

    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring='accuracy')
    print(f"Cross-val accuracy: {cv_scores.mean()*100:.1f}% "
          f"(+/- {cv_scores.std()*100:.1f}%)")

    # ── Feature importance ─────────────────────────────────────
    print("\n=== Feature Importance ===")
    for name, imp in sorted(
            zip(FEATURE_COLS, clf.feature_importances_),
            key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:15s} {bar} {imp:.3f}")

    # ── Save model ─────────────────────────────────────────────
    joblib.dump({'model': clf, 'encoder': le}, 'breathing_model.pkl')
    print("\nModel saved to breathing_model.pkl")


# ─────────────────────────────────────────────
#  REAL-TIME PREDICTION (console mode)
#  Collects WINDOW_SEC seconds of live data,
#  extracts features, predicts label
# ─────────────────────────────────────────────

FEEDBACK_MAP = {
    'normal': ("Normal breathing detected ✓",          "\033[92m"),  # green
    'deep':   ("Deep / relaxed breathing detected ✓",  "\033[96m"),  # cyan
    'mix':    ("Irregular breathing pattern detected",  "\033[93m"),  # yellow
}

def predict():
    import joblib
    import serial
    import serial.tools.list_ports

    if not os.path.exists('breathing_model.pkl'):
        print("Model not found. Run with --train first.")
        return

    bundle = joblib.load('breathing_model.pkl')
    clf    = bundle['model']
    le     = bundle['encoder']

    # Find port
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found. Is Arduino connected?")
        return

    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    idx  = int(input("Select port number: "))
    port = ports[idx].device

    ser = serial.Serial(port, 9600, timeout=1)
    time.sleep(1.5)
    ser.reset_input_buffer()

    print(f"\nConnected to {port}. Collecting {WINDOW_SEC}s windows...\n")

    # Rolling buffer of rows
    buffer = deque()
    t0     = time.time()

    # Simple signal processor (same logic as GUI)
    ma_buf   = deque(maxlen=5)
    baseline = None
    noise    = 1.0
    in_inh   = False
    last_bt  = None
    bpm      = 0.0
    inh_st   = None
    peak_t   = None
    peak_v   = -1e9
    ie_ratio = None
    tv       = 0.0
    cy_max   = -1e9
    cy_min   =  1e9
    periods  = deque(maxlen=8)
    variab   = 0.0
    last_t   = None
    last_x   = 0.0

    def process(raw, t):
        nonlocal baseline, noise, in_inh, last_bt, bpm
        nonlocal inh_st, peak_t, peak_v, ie_ratio, tv
        nonlocal cy_max, cy_min, variab, last_t, last_x

        ma_buf.append(raw)
        S = sum(ma_buf) / len(ma_buf)

        if baseline is None:
            baseline = S
        else:
            baseline = 0.05 * S + 0.95 * baseline

        x = S - baseline
        cy_max = max(cy_max, x)
        cy_min = min(cy_min, x)

        noise = 0.95 * noise + 0.05 * abs(x)
        th_up = max(0.5, 1.2 * noise)   # lowered for weak diaphragm signal
        th_dn = 0.3 * th_up

        dt = (t - last_t) if last_t else 0.0

        if not in_inh:
            ref_ok = last_bt is None or (t - last_bt) > 0.9
            if x > th_up and ref_ok:
                in_inh = True
                inh_st = t
                peak_t = t
                peak_v = x
                if last_bt is not None:
                    period = t - last_bt
                    periods.append(period)
                    bpm = 60.0 / period
                    if len(periods) >= 2:
                        variab = float(np.std(list(periods)))
                last_bt = t
                if cy_max > -1e8:
                    tv = cy_max - cy_min
                cy_max = -1e9
                cy_min =  1e9
        else:
            if x > peak_v:
                peak_v = x
                peak_t = t
            if x < th_dn:
                in_inh = False
                if inh_st and peak_t:
                    inh_t = max(0.0, peak_t - inh_st)
                    exh_t = max(0.0, t - peak_t)
                    if exh_t > 0.1:
                        ie_ratio = inh_t / exh_t

        last_t = t
        last_x = x

        return {
            'time_s':       f"{t - t0:.3f}",
            'bpm':          bpm,
            'ie_ratio':     ie_ratio if ie_ratio else '',
            'tidal_volume': tv,
            'variability':  variab,
        }

    window_start = time.time()

    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if not line.lstrip('-').isdigit():
                continue

            raw = int(line)
            t   = time.time() - t0
            row = process(raw, t)
            buffer.append(row)

            # Remove old rows outside window
            cutoff = t - WINDOW_SEC
            while buffer and float(buffer[0]['time_s']) < cutoff:
                buffer.popleft()

            # Every STEP_SEC seconds make a prediction
            elapsed = time.time() - window_start
            if elapsed >= STEP_SEC and len(buffer) > 20:
                window_start = time.time()
                print(f"  [debug] buffer={len(buffer)} rows, current BPM={bpm:.1f}, breaths detected so far")

                df_win = pd.DataFrame(list(buffer))
                df_win['bpm']          = pd.to_numeric(df_win['bpm'],          errors='coerce').fillna(0)
                df_win['ie_ratio']     = pd.to_numeric(df_win['ie_ratio'],      errors='coerce')
                df_win['tidal_volume'] = pd.to_numeric(df_win['tidal_volume'],  errors='coerce').fillna(0)
                df_win['variability']  = pd.to_numeric(df_win['variability'],   errors='coerce').fillna(0)

                feats = extract_features(df_win)
                if feats:
                    X     = np.array([[feats[c] for c in FEATURE_COLS]])
                    X     = np.array([[feats[c] for c in FEATURE_COLS]])
                    pred  = le.inverse_transform(clf.predict(X))[0]
                    proba = clf.predict_proba(X)[0]
                    conf  = max(proba) * 100

                    msg, color = FEEDBACK_MAP.get(
                        pred, (f"Pattern: {pred}", "\033[97m"))
                    reset = "\033[0m"
                    print(f"{color}[{pred.upper():10s}] {msg}  "
                          f"(confidence: {conf:.0f}%)  "
                          f"BPM:{bpm:.1f}  I:E:{ie_ratio:.2f if ie_ratio else 'N/A'}{reset}")

    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breathing ML Classifier")
    parser.add_argument('--train',   action='store_true', help='Train model from CSV files')
    parser.add_argument('--predict', action='store_true', help='Real-time prediction from Arduino')
    args = parser.parse_args()

    if args.train:
        print("=== TRAINING MODE ===\n")
        train()
    elif args.predict:
        print("=== PREDICTION MODE ===\n")
        predict()
    else:
        print("Usage:")
        print("  python breathing_ml.py --train     # train from CSV files")
        print("  python breathing_ml.py --predict   # real-time prediction")