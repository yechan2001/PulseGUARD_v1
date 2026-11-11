"""Beat-random and patient-disjoint splits."""

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import CLASS_IDS, LABELS


def beat_split(y: np.ndarray, seed: int):
    all_idx = np.arange(len(y))
    train_val, test = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=0.2, stratify=y[train_val], random_state=seed
    )
    return np.asarray(train), np.asarray(val), np.asarray(test)


def patient_split(meta: pd.DataFrame, method: str, seed: int):
    if "patient_id" not in meta:
        raise ValueError("Patient protocol requires ptbxl_database.csv metadata")
    if method == "official":
        train = np.flatnonzero(meta["strat_fold"].between(1, 8).to_numpy())
        val = np.flatnonzero((meta["strat_fold"] == 9).to_numpy())
        test = np.flatnonzero((meta["strat_fold"] == 10).to_numpy())
    else:
        # An alternative only when the *subset* leaves folds 9/10 very sparse.
        # GroupShuffleSplit prevents patient overlap, but is not stratified.
        all_idx = np.arange(len(meta))
        group1 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=seed)
        train_val_local, test_local = next(group1.split(all_idx, groups=meta["patient_id"]))
        train_val = all_idx[train_val_local]
        test = all_idx[test_local]
        group2 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=seed + 1)
        train_local, val_local = next(
            group2.split(train_val, groups=meta.iloc[train_val]["patient_id"])
        )
        train = train_val[train_local]
        val = train_val[val_local]
    for name, arr in zip(("train", "val", "test"), (train, val, test)):
        if len(arr) == 0:
            raise ValueError(
                f"Empty {name} split. Use --patient-split group if official subset sparse."
            )
    group_sets = [set(meta.iloc[idx]["patient_id"].tolist()) for idx in (train, val, test)]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Patient leakage across train/val/test!")
    return np.asarray(train), np.asarray(val), np.asarray(test)


def describe_split(meta, y, split, out_dir: Path, protocol: str):
    train, val, test = split
    parts = ["train", "validation", "test"]
    names = np.empty(len(meta), dtype=object)
    assigned = np.zeros(len(meta), dtype=bool)
    counts = []
    for name, ids in zip(parts, (train, val, test)):
        if assigned[ids].any():
            raise AssertionError(f"Split indices duplicated in {protocol}")
        assigned[ids] = True
        names[ids] = name
        c = Counter(y[ids])
        rec = {"split": name, "n_beats": len(ids), "n_records": meta.iloc[ids]["ecg_id"].nunique()}
        if "patient_id" in meta:
            rec["n_patients"] = meta.iloc[ids]["patient_id"].nunique()
        for label_id, label_name in enumerate(LABELS):
            rec[label_name] = int(c.get(label_id, 0))
        counts.append(rec)
    if not assigned.all():
        raise AssertionError("Some beats were not assigned to any split")
    manifest = meta.copy()
    manifest["label_id"] = y
    manifest["label_name"] = [LABELS[i] for i in y]
    manifest["split"] = names
    manifest.to_csv(out_dir / "split_manifest.csv", index=False)
    pd.DataFrame(counts).to_csv(out_dir / "split_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False))
    if "patient_id" in meta:
        t = set(meta.iloc[train]["patient_id"])
        v = set(meta.iloc[val]["patient_id"])
        e = set(meta.iloc[test]["patient_id"])
        overlaps = {
            "train_test_patients": len(t & e),
            "train_val_patients": len(t & v),
            "val_test_patients": len(v & e),
        }
        (out_dir / "patient_overlap.json").write_text(
            json.dumps(overlaps, indent=2), encoding="utf-8"
        )
        print("Patient overlaps:", overlaps)
        if protocol == "patient" and any(overlaps.values()):
            raise AssertionError("Detected patient leakage in the patient-independent protocol")
    train_counts = Counter(y[train])
    missing = [LABELS[k] for k in CLASS_IDS if not train_counts.get(k, 0)]
    if missing:
        raise ValueError(
            f"Training split has no samples for classes {missing}. "
            "Cannot fit a trustworthy six-class model; collect more records "
            "or change the split plan before training."
        )
    test_missing = [LABELS[k] for k in CLASS_IDS if k not in set(y[test])]
    if test_missing:
        warnings.warn(
            f"Test split is missing classes {test_missing}. Class-specific AUC "
            "and macro AUROC across all six classes will be undefined."
        )
    return counts
