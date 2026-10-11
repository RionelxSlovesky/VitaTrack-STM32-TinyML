import argparse
import pickle
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd

FS = 700
WINDOW_SECONDS = 60
STEP_SECONDS = 30

FEATURES = [
    "mean_rr_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_pct",
    "rr_range_ms",
]

VALID_LABELS = {1, 2, 3}  # baseline, stress, amusement


def extract_hrv_features(ecg_window):
    # NeuroKit is used only for offline training-data R-peak detection.
    _, info = nk.ecg_peaks(
        ecg_window,
        sampling_rate=FS,
        correct_artifacts=True,
    )

    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)

    if len(peaks) < 8:
        return None

    rr = np.diff(peaks).astype(np.float64) * (1000.0 / FS)

    # Match VitaTrack firmware plausibility bounds.
    rr = rr[(rr >= 300.0) & (rr <= 2000.0)]

    if len(rr) < 8:
        return None

    rr_diff = np.diff(rr)

    return {
        "mean_rr_ms": float(np.mean(rr)),
        "mean_hr_bpm": float(60000.0 / np.mean(rr)),
        "sdnn_ms": float(np.std(rr, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(rr_diff ** 2))),
        "pnn50_pct": float(100.0 * np.mean(np.abs(rr_diff) > 50.0)),
        "rr_range_ms": float(np.max(rr) - np.min(rr)),
    }


def process_subject(pkl_path):
    subject = pkl_path.stem

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    ecg = np.asarray(obj["signal"]["chest"]["ECG"]).squeeze()
    labels = np.asarray(obj["label"]).squeeze()

    if len(ecg) != len(labels):
        raise ValueError(f"{subject}: ECG and label lengths do not match")

    win = WINDOW_SECONDS * FS
    step = STEP_SECONDS * FS

    rows = []

    for start in range(0, len(ecg) - win + 1, step):
        stop = start + win

        y = labels[start:stop]
        valid_mask = np.isin(y, list(VALID_LABELS))

        # Avoid transition-heavy windows.
        if valid_mask.mean() < 0.95:
            continue

        valid_y = y[valid_mask]
        values, counts = np.unique(valid_y, return_counts=True)
        majority_label = int(values[np.argmax(counts)])
        majority_fraction = counts.max() / len(valid_y)

        if majority_fraction < 0.95:
            continue

        features = extract_hrv_features(ecg[start:stop])
        if features is None:
            continue

        # Binary target:
        # 1 = stress
        # 0 = non-stress (baseline or amusement)
        features["subject"] = subject
        features["source_label"] = majority_label
        features["label"] = 1 if majority_label == 2 else 0
        features["start_sec"] = start / FS
        features["stop_sec"] = stop / FS

        rows.append(features)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wesad",
        required=True,
        help="Path to extracted WESAD root containing S2/, S3/, ...",
    )
    parser.add_argument(
        "--out",
        default="wesad_hrv_features.csv",
    )
    args = parser.parse_args()

    root = Path(args.wesad)
    subject_files = sorted(root.glob("S*/S*.pkl"))

    if not subject_files:
        raise FileNotFoundError(
            f"No WESAD subject pickle files found under: {root}"
        )

    rows = []

    for pkl in subject_files:
        print(f"Processing {pkl} ...")
        subject_rows = process_subject(pkl)
        print(f"  accepted windows: {len(subject_rows)}")
        rows.extend(subject_rows)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No usable WESAD windows were extracted.")

    df.to_csv(args.out, index=False)

    print()
    print("=== Dataset summary ===")
    print("Rows:", len(df))
    print("Subjects:", df["subject"].nunique())
    print()
    print("Binary class counts:")
    print(df["label"].value_counts().sort_index())
    print()
    print("Per-subject counts:")
    print(df.groupby(["subject", "label"]).size())
    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
