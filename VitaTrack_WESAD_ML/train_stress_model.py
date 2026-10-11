import argparse
import json
from pathlib import Path

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
from sklearn.utils.class_weight import compute_class_weight

FEATURES = [
    "mean_rr_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50_pct",
    "rr_range_ms",
]


def group_split(X, y, groups, test_size, seed):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    a, b = next(splitter.split(X, y, groups))
    return a, b


def make_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),)),
        tf.keras.layers.Dense(12, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def write_metadata_header(path, mean, std, threshold=0.5):
    lines = [
        "#ifndef VITATRACK_STRESS_MODEL_METADATA_H",
        "#define VITATRACK_STRESS_MODEL_METADATA_H",
        "",
        "#define VITATRACK_STRESS_FEATURE_COUNT 6U",
        f"#define VITATRACK_STRESS_THRESHOLD {threshold:.6f}f",
        "",
        "static const float vitatrack_feature_mean[6] = {",
        "    " + ", ".join(f"{x:.9g}f" for x in mean),
        "};",
        "",
        "static const float vitatrack_feature_std[6] = {",
        "    " + ", ".join(f"{x:.9g}f" for x in std),
        "};",
        "",
        "#endif /* VITATRACK_STRESS_MODEL_METADATA_H */",
        "",
    ]

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="stress_model")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv).dropna(
        subset=FEATURES + ["label", "subject"]
    )

    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int32)
    groups = df["subject"].astype(str).to_numpy()

    # First split: subject-independent held-out test set.
    trainval_idx, test_idx = group_split(
        X, y, groups,
        test_size=0.25,
        seed=args.seed,
    )

    X_trainval = X[trainval_idx]
    y_trainval = y[trainval_idx]
    g_trainval = groups[trainval_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]
    g_test = groups[test_idx]

    # Second split: subject-independent validation set.
    train_idx_local, val_idx_local = group_split(
        X_trainval,
        y_trainval,
        g_trainval,
        test_size=0.20,
        seed=args.seed + 1,
    )

    X_train = X_trainval[train_idx_local]
    y_train = y_trainval[train_idx_local]
    g_train = g_trainval[train_idx_local]

    X_val = X_trainval[val_idx_local]
    y_val = y_trainval[val_idx_local]
    g_val = g_trainval[val_idx_local]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight = {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }

    model = make_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train_n,
        y_train,
        validation_data=(X_val_n, y_val),
        epochs=250,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    p = model.predict(X_test_n, verbose=0).reshape(-1)
    pred = (p >= 0.5).astype(np.int32)

    print()
    print("=== Subject-independent evaluation ===")
    print("Train subjects:", sorted(set(g_train)))
    print("Val subjects:  ", sorted(set(g_val)))
    print("Test subjects: ", sorted(set(g_test)))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    print()
    print(classification_report(
        y_test,
        pred,
        digits=4,
        zero_division=0,
    ))

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred)
        ),
        "precision": float(
            precision_score(y_test, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, pred, zero_division=0)
        ),
    }

    if len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, p)
        )

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k:18s}: {v:.4f}")

    # Save float model and metadata.
    model.save(outdir / "stress_float.keras")

    np.save(outdir / "feature_mean.npy", mean)
    np.save(outdir / "feature_std.npy", std)

    with open(outdir / "feature_order.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_metadata_header(
        outdir / "vitatrack_stress_model_metadata.h",
        mean,
        std,
        threshold=0.5,
    )

    # Full integer quantization.
    def representative_dataset():
        n = min(len(X_train_n), 500)
        for i in range(n):
            yield [
                X_train_n[i:i+1].astype(np.float32)
            ]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = outdir / "stress_int8.tflite"
    tflite_path.write_bytes(tflite_model)

    print()
    print("Saved:", tflite_path)
    print("INT8 model size:", len(tflite_model), "bytes")


if __name__ == "__main__":
    main()
