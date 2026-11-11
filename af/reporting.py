"""Write the two-arm report with confusion matrices and full ROC/PR coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc

from ..curves import export_curves
from ..render_html import render_html
from .evaluation import LABELS, PREDICTION_FILES
from .pipeline import ARMS

DISPLAY = {"ppg": "Direct PPG", "reconstructed_ecg": "Reconstructed ECG"}
LEVELS = {
    "per_beat": "Beat",
    "per_segment_from_beats": "3-s segment from beats",
    "per_subject_from_beats": "Subject from beats",
}


def draw_confusion(axis, matrix, title):
    axis.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() / 2 else "black"
        axis.text(column, row, str(value), ha="center", va="center", color=color)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )


def save_figure(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def format_metric(row):
    if pd.isna(row.estimate):
        return "N/A"
    if pd.isna(row.ci95_lower):
        return f"{row.estimate:.4f}"
    return f"{row.estimate:.4f} [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}]"


def run(context):
    output = context.output_dir
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    stats = pd.read_csv(output / "statistics/metrics_long.csv")
    confusion_fig, confusion_axes = plt.subplots(len(LEVELS), len(ARMS), figsize=(10, 12))
    roc_fig, roc_axes = plt.subplots(1, len(LEVELS), figsize=(15, 4))
    pr_fig, pr_axes = plt.subplots(1, len(LEVELS), figsize=(15, 4))
    summary_rows, details = [], []
    for row, (level, title) in enumerate(LEVELS.items()):
        for col, arm in enumerate(ARMS):
            folder = output / "evaluation" / arm
            frame = pd.read_csv(folder / PREDICTION_FILES[level])
            counts = pd.read_csv(folder / f"{level}_confusion.csv", index_col=0).to_numpy()
            draw_confusion(confusion_axes[row, col], counts, f"{DISPLAY[arm]} / {title}")
            roc, pr, ap = export_curves(folder, level, frame.label, frame.p_af)
            subset = stats[(stats.level == level) & (stats.modality == arm)]
            accuracy = subset[subset.metric == "accuracy"].iloc[0]
            macro = subset[(subset.scope == "macro_all_classes") & (subset.metric == "f1")].iloc[0]
            af_auc = subset[(subset["class"] == "AF") & (subset.metric == "auroc")].iloc[0]
            if roc is not None:
                if abs(auc(roc.fpr, roc.tpr) - af_auc.estimate) >= 1e-10:
                    raise ValueError(f"ROC integration differs from saved AUC for {arm}, {level}.")
                roc_axes[row].plot(
                    roc.fpr, roc.tpr, label=f"{DISPLAY[arm]} ({af_auc.estimate:.4f})"
                )
                pr_axes[row].plot(pr.recall, pr.precision, label=f"{DISPLAY[arm]} (AP {ap:.4f})")
            summary_rows.append(
                f"| {title} | {DISPLAY[arm]} | {len(frame):,} | {format_metric(accuracy)} "
                f"| {format_metric(macro)} | {format_metric(af_auc)} |"
            )
            details += [
                f"### {DISPLAY[arm]} / {title}",
                "",
                "| Class | Sensitivity | Specificity | PPV | F1 | AUROC |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            metrics = ["sensitivity", "specificity", "ppv", "f1", "auroc"]
            for label, scope in [
                ("non-AF", "class"),
                ("AF", "class"),
                ("macro", "macro_all_classes"),
            ]:
                class_rows = subset[(subset["class"] == label) & (subset.scope == scope)].set_index(
                    "metric"
                )
                cells = [label, *(format_metric(class_rows.loc[metric]) for metric in metrics)]
                details.append("| " + " | ".join(cells) + " |")
            details.append("")
        roc_axes[row].plot([0, 1], [0, 1], linestyle="--", color="gray")
        roc_axes[row].set(xlabel="False positive rate", ylabel="Sensitivity", title=title)
        pr_axes[row].set(xlabel="Recall", ylabel="Precision", title=title)
        for axis in (roc_axes[row], pr_axes[row]):
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)

    save_figure(confusion_fig, figures / "confusion.png")
    save_figure(roc_fig, figures / "roc.png")
    save_figure(pr_fig, figures / "pr.png")
    report = "\n".join(
        [
            "# AF beat ablation",
            "",
            "The two arms share beat IDs and real ECG reference boundaries. The PPG classifier "
            "is trained on direct PPG. The reconstructed ECG arm uses a classifier trained on "
            "real ECG. Classification uses p_AF > 0.5.",
            "",
            "| Unit | Input | N | Accuracy [95% CI] | Macro F1 [95% CI] | AF AUROC [95% CI] |",
            "| --- | --- | --- | --- | --- | --- |",
            *summary_rows,
            "",
            "![Confusion matrices](figures/confusion.png)",
            "",
            "![Receiver operating characteristic](figures/roc.png)",
            "",
            "![Precision recall](figures/pr.png)",
            "",
            "## Per-class and macro estimates with 95% confidence intervals",
            "",
            *details,
            "Beat and segment confidence intervals use patient-cluster percentile bootstrap. "
            "The two arms share the resampling plan. Subject-level rates use exact binomial "
            "intervals. The independent AF test cohort has seven patients.",
            "",
        ]
    )
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    render_html(report, output, output / "REPORT.html")
    print(f"AF report written to {output / 'REPORT.html'}")
