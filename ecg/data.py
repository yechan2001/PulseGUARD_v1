"""Load and normalize the labeled ECG beats."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

LABELS = ("AFIB", "AFLT", "MI", "NORM", "PSVT", "PVC")
CLASS_IDS = np.arange(len(LABELS), dtype=int)


def load_beats(data_dir: Path, input_length: int = 100):
    """Keep traceability: each beat retains its original ecg_id and beat index.

    Assumes per-record `segments` have shape (n_beats, n_samples) or (n_beats,
    n_samples, 1), with a matching length-n_beats array `labels`. If a source
    record is malformed, fail rather than silently manufacture data/labels.
    """
    files = sorted(data_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files below: {data_dir}")

    X, y, rows, seen_ecg_ids = [], [], [], set()
    raw_length_counts = Counter()
    n_invalid = 0
    for file in files:
        try:
            ecg_id = int(file.stem)
        except ValueError as exc:
            raise ValueError(f".npz basename must equal a numeric PTB-XL ecg_id: {file}") from exc
        if ecg_id in seen_ecg_ids:
            raise ValueError(f"Repeated ecg_id={ecg_id}; duplicate file/record detected: {file}")
        seen_ecg_ids.add(ecg_id)

        with np.load(file, allow_pickle=False) as arr:
            if "segments" not in arr or "labels" not in arr:
                raise KeyError(f"Missing `segments` or `labels`: {file}")
            segments = arr["segments"]
            labels = np.asarray(arr["labels"]).reshape(-1)
        if segments.ndim == 3 and segments.shape[-1] == 1:
            segments = segments[:, :, 0]
        if segments.ndim != 2:
            raise ValueError(f"Expected (N,L) or (N,L,1) segments, got {segments.shape}: {file}")
        if len(labels) != len(segments):
            raise ValueError(f"Number of labels differs from beats: {file}")
        for beat_idx, (beat, label_raw) in enumerate(zip(segments, labels)):
            label = int(label_raw)
            if float(label_raw) != label:
                raise ValueError(f"Non-integer label {label_raw} in {file}")
            if label == 6:
                n_invalid += 1
                continue
            if label not in CLASS_IDS:
                raise ValueError(f"Unsupported label {label} in {file}; expected 0..6")
            beat = np.asarray(beat, dtype=np.float32).reshape(-1)
            if len(beat) < 2 or not np.isfinite(beat).all():
                raise ValueError(f"Invalid beat ecg_id={ecg_id}, index={beat_idx}")
            # Same position-space linear interpolation as the original data.py.
            # It is NOT a physical resampling routine. Different source beat
            # durations should be checked separately before publication.
            t_old = np.linspace(0, 1, len(beat))
            t_new = np.linspace(0, 1, input_length)
            X.append(np.interp(t_new, t_old, beat).astype(np.float32))
            y.append(label)
            rows.append(
                {
                    "ecg_id": ecg_id,
                    "beat_id": beat_idx,
                    "original_length": len(beat),
                    "source_file": str(file),
                }
            )
            raw_length_counts[len(beat)] += 1
    if not X:
        raise ValueError("All input beats were rejected (or labeled 6).")
    meta = pd.DataFrame(rows)
    meta["sample_id"] = np.arange(len(meta), dtype=int)
    print(f"Loaded {len(X)} valid beats from {len(files)} records; excluded label=6: {n_invalid}")
    print("Class distribution:", {LABELS[k]: int(v) for k, v in sorted(Counter(y).items())})
    print("Original beat lengths:", dict(sorted(raw_length_counts.items())))
    return np.stack(X), np.asarray(y, dtype=int), meta


def attach_ptbxl_metadata(meta: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"PTB-XL metadata CSV not found: {metadata_csv}")
    db = pd.read_csv(metadata_csv)
    required = {"ecg_id", "patient_id", "strat_fold"}
    if not required.issubset(db.columns):
        raise KeyError(f"{metadata_csv} missing columns: {required - set(db.columns)}")
    if db["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ecg_id in ptbxl_database.csv")
    keys = db[["ecg_id", "patient_id", "strat_fold"]]
    merged = meta.merge(keys, on="ecg_id", how="left", validate="many_to_one", sort=False)
    if merged[["patient_id", "strat_fold"]].isnull().any().any():
        missing = sorted(merged.loc[merged["patient_id"].isna(), "ecg_id"].unique().tolist())
        raise ValueError(f"Missing PTB-XL patient_id/strat_fold mapping for ecg_id: {missing[:20]}")
    for col in ("patient_id", "strat_fold"):
        merged[col] = merged[col].astype(int)
    if not merged["strat_fold"].between(1, 10).all():
        raise ValueError("PTB-XL strat_fold must be between 1 and 10")
    if (merged.groupby("patient_id")["strat_fold"].nunique() > 1).any():
        raise ValueError(
            "Some patient_ids appear in multiple official folds; metadata inconsistent"
        )
    # Preserve the exact original signal row order.
    return merged.sort_values("sample_id").reset_index(drop=True)


def standardize(X: np.ndarray, idx_train, idx_val, idx_test):
    mean = float(X[idx_train].mean(dtype=np.float64))
    std = float(X[idx_train].std(dtype=np.float64))
    if not np.isfinite(std) or std < 1e-12:
        raise ValueError("Training ECG standard deviation is ~0")

    def transform(idx):
        return ((X[idx] - mean) / std).astype(np.float32)[..., None]

    return transform(idx_train), transform(idx_val), transform(idx_test), mean, std


def resample_train(X_train, y_train, balance: str, seed: int):
    class_weight = None
    if balance == "weights":
        weights = compute_class_weight("balanced", classes=CLASS_IDS, y=y_train)
        class_weight = {int(k): float(v) for k, v in zip(CLASS_IDS, weights)}
        print("Train-only class weights:", class_weight)
    elif balance == "smote":
        from imblearn.over_sampling import SMOTE

        min_c = min(Counter(y_train).values())
        if min_c < 2:
            raise ValueError("SMOTE requires at least two samples in every training class")
        k_neighbors = min(5, min_c - 1)
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
        X_new, y_new = smote.fit_resample(X_train[:, :, 0], y_train)
        X_train = np.asarray(X_new, dtype=np.float32)[..., None]
        y_train = np.asarray(y_new, dtype=int)
        print(
            "Train-only SMOTE:", len(X_train), "beats; synthetic waves must be treated cautiously"
        )
    elif balance != "none":
        raise ValueError(f"Unknown balance: {balance}")
    return X_train, y_train, class_weight
