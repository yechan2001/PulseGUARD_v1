"""Fit a separate classifier under each requested split protocol."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import make_model
from .data import LABELS, attach_ptbxl_metadata, load_beats, resample_train, standardize
from .evaluation import score_beats, score_patient_aggregates
from .splits import beat_split, describe_split, patient_split


def train_one(protocol: str, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, args):
    output = Path(args.output_dir) / protocol
    output.mkdir(parents=True, exist_ok=True)
    if protocol == "beat":
        split = beat_split(y, args.seed)
    else:
        split = patient_split(meta, args.patient_split, args.seed)
    describe_split(meta, y, split, output, protocol)
    train, val, test = split
    if args.dry_run:
        return {
            "protocol": protocol,
            "dry_run": True,
            "train_beats": len(train),
            "validation_beats": len(val),
            "test_beats": len(test),
        }

    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    X_train, X_val, X_test, mean, std = standardize(X, train, val, test)
    y_train, y_val, y_test = y[train].copy(), y[val], y[test]
    X_train, y_train, class_weights = resample_train(X_train, y_train, args.balance, args.seed)
    (output / "preprocess.json").write_text(
        json.dumps(
            {
                "train_mean": mean,
                "train_std": std,
                "input_length": args.input_length,
                "label_names": list(LABELS),
                "smote_or_weights": args.balance,
                "source": "existing pre-segmented .npz files",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    model = make_model(args.input_length)
    path = output / "best_model.keras"
    cb = [
        tf.keras.callbacks.ModelCheckpoint(
            str(path), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        class_weight=class_weights,
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(output / "training_history.csv", index=False)
    best = tf.keras.models.load_model(path)
    probs = best.predict(X_test, batch_size=args.batch_size, verbose=0)
    beat_metrics = score_beats(y_test, probs, meta.iloc[test].copy(), output)
    patient_metrics = score_patient_aggregates(y_test, probs, meta.iloc[test].copy(), output)
    return {
        "protocol": protocol,
        **beat_metrics,
        "test_patients": patient_metrics.get("n_patients"),
        "single_label_patient_accuracy": patient_metrics.get("single_label_patient_accuracy"),
        "single_label_patient_macro_f1": patient_metrics.get("single_label_patient_macro_f1"),
        "single_label_patients": patient_metrics.get("n_single_label_patients"),
    }


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir).expanduser()
    X, y, meta = load_beats(data_dir, args.input_length)
    if args.ptbxl_csv:
        meta = attach_ptbxl_metadata(meta, Path(args.ptbxl_csv).expanduser())
    if args.protocol in ("patient", "both") and "patient_id" not in meta:
        raise ValueError(
            "Patient-independent results require --ptbxl-csv. "
            "Cannot replace patient_id with ecg_id or random guess."
        )
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_dir / "all_beats_metadata.csv", index=False)
    results = []
    protocols = ["beat", "patient"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        print("\n" + "=" * 65 + f"\nPROTOCOL: {protocol}\n" + "=" * 65)
        results.append(train_one(protocol, X, y, meta, args))
    pd.DataFrame(results).to_csv(out_dir / "summary_comparison.csv", index=False)
    print("\nResults saved in", out_dir.resolve())
    print(pd.DataFrame(results).to_string(index=False))
    return results
