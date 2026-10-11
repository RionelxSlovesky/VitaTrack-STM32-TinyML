
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["mean_rr_ms","mean_hr_bpm","sdnn_ms","rmssd_ms","pnn50_pct","rr_range_ms"]

def get_metrics(y, p, threshold):
    pred = (p >= threshold).astype(np.int32)
    m = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }
    return m, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--float-model", required=True)
    ap.add_argument("--int8-model", required=True)
    ap.add_argument("--mean", required=True)
    ap.add_argument("--std", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=FEATURES+["label","subject"])
    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)
    groups = df["subject"].astype(str).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    _, test_idx = next(splitter.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]
    g_test = groups[test_idx]

    mean = np.load(args.mean)
    std = np.load(args.std)
    Xn = (X_test - mean) / std

    float_model = tf.keras.models.load_model(args.float_model)
    p_float = float_model.predict(Xn, verbose=0).reshape(-1)

    interpreter = tf.lite.Interpreter(model_path=args.int8_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    p_int8 = []
    for row in Xn:
        q = np.rint(row / in_scale + in_zero)
        q = np.clip(q, -128, 127).astype(np.int8).reshape(1, -1)
        interpreter.set_tensor(inp["index"], q)
        interpreter.invoke()
        qo = interpreter.get_tensor(out["index"]).reshape(-1)[0]
        p_int8.append((float(qo) - out_zero) * out_scale)
    p_int8 = np.asarray(p_int8, dtype=np.float32)

    fm, fpred = get_metrics(y_test, p_float, args.threshold)
    im, ipred = get_metrics(y_test, p_int8, args.threshold)

    print("=== Held-out subjects ===")
    print(sorted(set(g_test)))
    print("Samples:", len(y_test))

    print("\n=== FLOAT ===")
    print(confusion_matrix(y_test, fpred))
    print(classification_report(y_test, fpred, digits=4, zero_division=0))
    for k,v in fm.items():
        print(f"{k:18s}: {v:.4f}")

    print("\n=== INT8 ===")
    print(confusion_matrix(y_test, ipred))
    print(classification_report(y_test, ipred, digits=4, zero_division=0))
    for k,v in im.items():
        print(f"{k:18s}: {v:.4f}")

    print("\n=== Quantization drift ===")
    print(f"Mean abs probability delta: {np.mean(np.abs(p_float-p_int8)):.6f}")
    print(f"Max abs probability delta : {np.max(np.abs(p_float-p_int8)):.6f}")
    print(f"Class agreement           : {100*np.mean(fpred==ipred):.2f}%")
    print(f"Input quantization        : scale={in_scale}, zero={in_zero}")
    print(f"Output quantization       : scale={out_scale}, zero={out_zero}")

if __name__ == "__main__":
    main()
