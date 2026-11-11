"""Fit the PPG classifier and the real-ECG classifier used for reconstructions."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from ..io import fingerprint, reusable, save_json, sha
from ..models import make_model


def run(context, modality):
    import tensorflow as tf

    cfg = context.config
    root = context.output_dir
    manifest = root / "data/beat_manifest.csv"
    data_path, folder = {
        "ppg": (root / "data_ppg/ppg_beats.npz", root / "model_ppg"),
        "ecg": (root / "data/paired_beats.npz", root / "model"),
    }[modality]
    source = Path(__file__)
    files = [context.config_path, manifest, data_path, source, source.parent.parent / "models.py"]
    current_fingerprint = fingerprint(files)
    if reusable(folder, current_fingerprint):
        print(f"Reusing {modality} classifier at {folder}")
        return

    meta = pd.read_csv(manifest)
    with np.load(data_path) as data:
        raw = data[modality]
        labels = data["label"]
        if not np.array_equal(data["beat_id"], meta.beat_id) or not np.array_equal(
            labels, meta.label
        ):
            raise ValueError(f"Beat IDs or labels do not match {manifest}.")
    indices = {
        part: np.flatnonzero(meta.partition.to_numpy() == part)
        for part in ("train", "validation", "test")
    }
    train, validation = indices["train"], indices["validation"]
    mean = float(raw[train].mean(dtype=np.float64))
    std = float(raw[train].std(dtype=np.float64))
    if not np.isfinite(std) or std <= 1e-12:
        raise ValueError(f"Invalid training standard deviation for {modality}.")
    fitted_input = "direct PPG" if modality == "ppg" else "real ECG"
    save_json(
        folder / "normalization.json",
        {
            "mean": mean,
            "std": std,
            "fitted_subjects": sorted(meta.iloc[train].subject.unique()),
            "fitted_input": f"{fitted_input} training beats only, after 180s",
        },
    )
    classes = np.array([0, 1])
    weights = dict(
        zip(
            classes.tolist(),
            compute_class_weight("balanced", classes=classes, y=labels[train]).tolist(),
        )
    )
    save_json(folder / "class_weights.json", weights)
    save_json(folder / "fingerprint.json", current_fingerprint)

    tf.keras.backend.clear_session()
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(cfg["seed"])
    model = make_model(cfg["input_length"], nclasses=2)
    initial = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()
    other = root / ("model" if modality == "ppg" else "model_ppg") / "complete.json"
    if other.is_file():
        previous = json.loads(other.read_text(encoding="utf-8"))
        if initial != previous["initial_weights_sha256"]:
            raise ValueError("Both classifiers must use the same initial weights.")

    features = ((raw - mean) / std).astype("float32")[..., None]
    checkpoint = folder / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=cfg["patience"], restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(str(folder / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    start = time.perf_counter()
    history = model.fit(
        features[train],
        labels[train],
        validation_data=(features[validation], labels[validation]),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=weights,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    ).history
    if not np.isfinite(history["loss"]).all() or not np.isfinite(history["val_loss"]).all():
        raise ValueError(f"Non-finite loss while training the {modality} classifier.")
    save_json(
        folder / "complete.json",
        {
            "passed": True,
            "n_parameters": model.count_params(),
            "training_seconds": time.perf_counter() - start,
            "epochs_trained": len(history["loss"]),
            "selected_epoch": int(np.argmin(history["val_loss"])) + 1,
            "selection": "minimum validation loss",
            "initial_weights_sha256": initial,
            "model_SHA256": sha(checkpoint),
            "counts": {part: len(ids) for part, ids in indices.items()},
            "training_inputs": f"{fitted_input} beats only",
        },
    )
