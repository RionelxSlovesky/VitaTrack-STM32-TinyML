
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

FEATURES = [
    "mean_rr_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_pct",
    "rr_range_ms",
]

def split_indices(X, y, groups, seed=42):
    outer = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed
    )
    trainval_idx, test_idx = next(outer.split(X, y, groups))

    X_tv = X[trainval_idx]
    y_tv = y[trainval_idx]
    g_tv = groups[trainval_idx]

    inner = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=seed + 1
    )
    train_local, val_local = next(inner.split(X_tv, y_tv, g_tv))

    train_idx = trainval_idx[train_local]
    val_idx = trainval_idx[val_local]

    return train_idx, val_idx, test_idx

def predict_int8(model_path, Xn):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    in_scale, in_zero = inp["quantization"]
    out_scale, out_zero = out["quantization"]

    probs = []
    for row in Xn:
        q = np.rint(row / in_scale + in_zero)
        q = np.clip(q, -128, 127).astype(np.int8).reshape(1, -1)

        interpreter.set_tensor(inp["index"], q)
        interpreter.invoke()

        qo = interpreter.get_tensor(out["index"]).reshape(-1)[0]
        probs.append((float(qo) - out_zero) * out_scale)

    return np.asarray(probs, dtype=np.float32)

def score_threshold(y, probs, threshold, objective):
    pred = (probs >= threshold).astype(np.int32)

    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    elif objective == "f1":
        return f1_score(y, pred, zero_division=0)
    elif objective == "recall":
        return recall_score(y, pred, zero_division=0)
    else:
        raise ValueError(objective)

def summarize(title, y, probs, threshold):
    pred = (probs >= threshold).astype(np.int32)

    print(f"\n=== {title} @ threshold {threshold:.3f} ===")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4, zero_division=0))

    metrics = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, probs),
    }

    for k, v in metrics.items():
        print(f"{k:18s}: {v:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--float-model", required=True)
    ap.add_argument("--int8-model", required=True)
    ap.add_argument("--mean", required=True)
    ap.add_argument("--std", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--objective",
        choices=["balanced_accuracy", "f1", "recall"],
        default="balanced_accuracy",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(
        subset=FEATURES + ["label", "subject"]
    )

    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)
    groups = df["subject"].astype(str).to_numpy()

    train_idx, val_idx, test_idx = split_indices(
        X, y, groups, seed=args.seed
    )

    mean = np.load(args.mean)
    std = np.load(args.std)

    X_val_n = (X[val_idx] - mean) / std
    X_test_n = (X[test_idx] - mean) / std

    model = tf.keras.models.load_model(args.float_model)

    p_val_float = model.predict(X_val_n, verbose=0).reshape(-1)
    p_test_float = model.predict(X_test_n, verbose=0).reshape(-1)

    p_val_int8 = predict_int8(args.int8_model, X_val_n)
    p_test_int8 = predict_int8(args.int8_model, X_test_n)

    # Optimize only on validation subjects using FLOAT probabilities.
    thresholds = np.arange(0.10, 0.901, 0.005)

    best_t = None
    best_score = -1.0

    for t in thresholds:
        s = score_threshold(
            y[val_idx], p_val_float, float(t), args.objective
        )

        if s > best_score:
            best_score = s
            best_t = float(t)

    print("=== Subject split ===")
    print("Train:", sorted(set(groups[train_idx])))
    print("Val:  ", sorted(set(groups[val_idx])))
    print("Test: ", sorted(set(groups[test_idx])))

    print()
    print(f"Threshold objective: {args.objective}")
    print(f"Best validation threshold: {best_t:.3f}")
    print(f"Best validation score: {best_score:.4f}")

    summarize(
        "VALIDATION FLOAT",
        y[val_idx], p_val_float, best_t
    )

    summarize(
        "VALIDATION INT8",
        y[val_idx], p_val_int8, best_t
    )

    summarize(
        "TEST FLOAT",
        y[test_idx], p_test_float, best_t
    )

    summarize(
        "TEST INT8",
        y[test_idx], p_test_int8, best_t
    )

    print("\n=== INT8 agreement on TEST ===")
    pred_float = (p_test_float >= best_t).astype(np.int32)
    pred_int8 = (p_test_int8 >= best_t).astype(np.int32)

    print(
        f"Class agreement: "
        f"{100*np.mean(pred_float == pred_int8):.2f}%"
    )
    print(
        f"Mean abs probability delta: "
        f"{np.mean(np.abs(p_test_float - p_test_int8)):.6f}"
    )

if __name__ == "__main__":
    main()
