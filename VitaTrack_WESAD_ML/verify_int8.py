import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = [
    "mean_rr_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_pct",
    "rr_range_ms",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--mean", required=True)
    p.add_argument("--std", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.csv).dropna(
        subset=FEATURES + ["label"]
    )

    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)

    mean = np.load(args.mean)
    std = np.load(args.std)

    Xn = (X - mean) / std

    interpreter = tf.lite.Interpreter(
        model_path=args.model
    )
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    probs = []

    for row in Xn:
        q = np.rint(
            row / in_scale + in_zero
        )
        q = np.clip(q, -128, 127).astype(np.int8)
        q = q.reshape(1, -1)

        interpreter.set_tensor(
            inp["index"],
            q
        )
        interpreter.invoke()

        qo = interpreter.get_tensor(
            out["index"]
        ).reshape(-1)[0]

        prob = (float(qo) - out_zero) * out_scale
        probs.append(prob)

    probs = np.asarray(probs)
    pred = (probs >= 0.5).astype(np.int32)

    print(classification_report(
        y,
        pred,
        digits=4,
        zero_division=0,
    ))

    if len(np.unique(y)) == 2:
        print("ROC-AUC:", roc_auc_score(y, probs))


if __name__ == "__main__":
    main()
