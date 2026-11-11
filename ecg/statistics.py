"""Replay the ECG-only statistics from original six-class prediction exports."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

from .. import statistics
from ..curves import export_curves
from ..io import save_json, sha
from .data import CLASS_IDS, LABELS


def run(args):
    output = args.output_dir.resolve()
    root = args.predictions_dir.resolve()
    statistics.initialize(output, args.bootstrap)
    protocols = ("beat", "patient") if args.protocol == "both" else (args.protocol,)
    sources = {}
    for protocol in protocols:
        predictions = root / protocol / "test_predictions.csv"
        patient_audit = root / protocol / "patient_predictions_and_label_audit.csv"
        sources.update({str(path): sha(path) for path in (predictions, patient_audit)})
        beat = pd.read_csv(predictions)
        patients = pd.read_csv(patient_audit)
        beat_scores = beat[[f"prob_{name}" for name in LABELS]].to_numpy()
        if not np.array_equal(beat.pred_id, beat_scores.argmax(axis=1)):
            raise ValueError(f"Saved beat predictions differ from argmax in {predictions}.")
        analyze(
            output,
            protocol,
            "per_segment_beat",
            beat.patient_id,
            beat.true_id,
            beat.pred_id,
            beat_scores,
        )

        single = patients[patients.single_label_patient.astype(str).str.lower().eq("true")]
        if not single.empty:
            scores = single[[f"mean_p_{name}" for name in LABELS]].to_numpy()
            pred = single.single_label_pred_id.to_numpy(dtype=int)
            if not np.array_equal(pred, scores.argmax(axis=1)):
                raise ValueError(
                    f"Saved patient predictions differ from argmax in {patient_audit}."
                )
            analyze(
                output,
                protocol,
                "per_subject_single_label",
                single.patient_id,
                single.single_label_true_id.to_numpy(dtype=int),
                pred,
                scores,
                excluded=len(patients) - len(single),
            )
        truth = patients[[f"any_beat_true_{name}" for name in LABELS]].to_numpy(dtype=bool)
        scores = patients[[f"max_p_{name}" for name in LABELS]].to_numpy()
        analyze(
            output,
            protocol,
            "per_subject_multilabel_exploratory",
            patients.patient_id,
            truth,
            scores >= 0.5,
            scores,
            multilabel=True,
        )
    statistics.export_tables()
    save_json(output / "source_manifest.json", sources)
    print(f"ECG statistics written to {output}")


def analyze(
    output, protocol, level, patients, truth, predicted, scores, multilabel=False, excluded=0
):
    statistics.analyze(
        "ECG6",
        protocol,
        "ecg",
        level,
        patients,
        truth,
        predicted,
        scores,
        LABELS,
        multilabel=multilabel,
        excluded=excluded,
    )
    folder = output / protocol / level
    folder.mkdir(parents=True, exist_ok=True)
    if multilabel:
        matrices = multilabel_confusion_matrix(truth, predicted)
        for name, matrix in zip(LABELS, matrices, strict=True):
            pd.DataFrame(
                matrix,
                index=["true_absent", "true_present"],
                columns=["pred_absent", "pred_present"],
            ).to_csv(folder / f"{name}_confusion_counts.csv")
        binary_truth = truth
    else:
        matrix = confusion_matrix(truth, predicted, labels=CLASS_IDS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(folder / "confusion_counts.csv")
        binary_truth = np.asarray(truth)[:, None] == CLASS_IDS
    for idx, name in enumerate(LABELS):
        export_curves(folder, name, binary_truth[:, idx], scores[:, idx])
