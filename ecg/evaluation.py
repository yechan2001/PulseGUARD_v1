"""Beat scores and patient probability aggregation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .data import CLASS_IDS, LABELS


def auc_per_class(y_true: np.ndarray, probs: np.ndarray):
    """One-vs-rest AUC, leaving undefined classes NaN (not 0.5)."""
    out = []
    for k, name in enumerate(LABELS):
        truth = (y_true == k).astype(int)
        if truth.min() == truth.max():
            roc, ap = np.nan, np.nan
        else:
            roc = float(roc_auc_score(truth, probs[:, k]))
            ap = float(average_precision_score(truth, probs[:, k]))
        out.append(
            {
                "class_id": k,
                "class_name": name,
                "support": int(truth.sum()),
                "roc_auc_ovr": roc,
                "average_precision": ap,
            }
        )
    return pd.DataFrame(out)


def draw_confusion(y_true, y_pred, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom != 0)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(out_dir / "confusion_counts.csv")
    pd.DataFrame(cm_norm, index=LABELS, columns=LABELS).to_csv(out_dir / "confusion_normalized.csv")
    for matrix, title, filename, fmt in [
        (cm, "Confusion matrix (counts)", "confusion_counts.png", "d"),
        (cm_norm, "Confusion matrix (row normalized)", "confusion_normalized.png", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(6), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def score_beats(y_test, probs, meta_test, out_dir: Path, figure=True):
    y_pred = probs.argmax(axis=1).astype(int)
    details = meta_test.copy().reset_index(drop=True)
    details["true_id"] = y_test
    details["true_name"] = [LABELS[i] for i in y_test]
    details["pred_id"] = y_pred
    details["pred_name"] = [LABELS[i] for i in y_pred]
    for k, name in enumerate(LABELS):
        details[f"prob_{name}"] = probs[:, k]
    details.to_csv(out_dir / "test_predictions.csv", index=False)
    report = classification_report(
        y_test, y_pred, labels=CLASS_IDS, target_names=LABELS, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report.csv")
    auc = auc_per_class(y_test, probs)
    auc.to_csv(out_dir / "per_class_auc.csv", index=False)
    valid_auc = auc["roc_auc_ovr"].notna().all()
    metrics = {
        "n_beats": len(y_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(y_test, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)
        ),
        "macro_auroc_all_6": float(auc["roc_auc_ovr"].mean()) if valid_auc else None,
        "macro_auroc_defined_classes_only": float(auc["roc_auc_ovr"].mean())
        if auc["roc_auc_ovr"].notna().any()
        else None,
    }
    if figure:
        draw_confusion(y_test, y_pred, out_dir)
    (out_dir / "beat_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Test beat metrics:", metrics)
    return metrics


def score_patient_aggregates(y_test, probs, meta_test, out_dir: Path):
    """Evaluate patient aggregation transparently, handling multi-label patients.

    Single-label: mean beat probabilities, only where ALL beats share one label.
    Multi-label: any-beat label presence per patient, max per-class probabilities;
    this is exploratory and does not imply a validated patient diagnosis system.
    """
    if "patient_id" not in meta_test:
        return {"n_patients": None, "n_single_label_patients": None}

    frame = meta_test[["patient_id", "ecg_id"]].reset_index(drop=True).copy()
    frame["label"] = y_test
    for k, name in enumerate(LABELS):
        frame[f"p_{name}"] = probs[:, k]
    rows, inclusion, one_y, one_p = [], [], [], []
    for patient, g in frame.groupby("patient_id", sort=True):
        observed = sorted(g["label"].unique().tolist())
        mean_p = g[[f"p_{n}" for n in LABELS]].to_numpy().mean(axis=0)
        max_p = g[[f"p_{n}" for n in LABELS]].to_numpy().max(axis=0)
        is_single = len(observed) == 1
        row = {
            "patient_id": int(patient),
            "n_beats": len(g),
            "n_records": g["ecg_id"].nunique(),
            "beat_label_ids": "|".join(map(str, observed)),
            "single_label_patient": bool(is_single),
            "single_label_true_id": int(observed[0]) if is_single else None,
            "single_label_pred_id": int(mean_p.argmax()) if is_single else None,
        }
        for k, name in enumerate(LABELS):
            row[f"any_beat_true_{name}"] = int(k in observed)
            row[f"max_p_{name}"] = float(max_p[k])
            row[f"mean_p_{name}"] = float(mean_p[k])
        rows.append(row)
        inclusion.append(is_single)
        if is_single:
            one_y.append(observed[0])
            one_p.append(mean_p)
    patient_df = pd.DataFrame(rows)
    patient_df.to_csv(out_dir / "patient_predictions_and_label_audit.csv", index=False)
    n_single = int(sum(inclusion))
    summary = {
        "n_patients": len(rows),
        "n_single_label_patients": n_single,
        "n_mixed_label_patients": len(rows) - n_single,
    }
    if n_single:
        y_single = np.asarray(one_y)
        p_single = np.stack(one_p)
        yp = p_single.argmax(axis=1)
        summary["single_label_patient_accuracy"] = float(accuracy_score(y_single, yp))
        summary["single_label_patient_macro_f1"] = float(
            f1_score(y_single, yp, labels=CLASS_IDS, average="macro", zero_division=0)
        )
        auc_df = auc_per_class(y_single, p_single)
        auc_df.to_csv(out_dir / "single_label_patient_auc.csv", index=False)
        summary["single_label_patient_macro_auroc_all_6"] = (
            float(auc_df["roc_auc_ovr"].mean()) if auc_df["roc_auc_ovr"].notna().all() else None
        )
        pd.DataFrame(
            classification_report(
                y_single,
                yp,
                labels=CLASS_IDS,
                target_names=LABELS,
                output_dict=True,
                zero_division=0,
            )
        ).T.to_csv(out_dir / "single_label_patient_report.csv")
    else:
        summary.update(
            {
                "single_label_patient_accuracy": None,
                "single_label_patient_macro_f1": None,
                "single_label_patient_macro_auroc_all_6": None,
            }
        )

    # Includes mixed-label patients; report AUC per class on binary patient
    # label-presence, not a misleading forced-exclusive six-class diagnosis.
    per_class = []
    for k, name in enumerate(LABELS):
        truth = patient_df[f"any_beat_true_{name}"].to_numpy(dtype=int)
        pred = patient_df[f"max_p_{name}"].to_numpy(dtype=float)
        if truth.min() == truth.max():
            auc, ap = None, None
        else:
            auc = float(roc_auc_score(truth, pred))
            ap = float(average_precision_score(truth, pred))
        per_class.append(
            {
                "class": name,
                "n_positive_patients": int(truth.sum()),
                "patient_presence_auroc": auc,
                "patient_presence_ap": ap,
            }
        )
    pd.DataFrame(per_class).to_csv(out_dir / "patient_any_beat_multilabel_auc.csv", index=False)
    (out_dir / "patient_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Patient-level aggregation (single-label subset):", summary)
    return summary
