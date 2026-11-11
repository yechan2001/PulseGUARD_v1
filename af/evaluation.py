"""Score direct PPG and reconstructed ECG on identical held-out beats."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from .. import statistics
from ..io import save_json, sha
from .pipeline import ARMS

LABELS = ("non-AF", "AF")
PREDICTION_FILES = {
    "per_beat": "beat_predictions.csv",
    "per_segment_from_beats": "segment_predictions_from_beats.csv",
    "per_subject_from_beats": "subject_predictions_from_beats.csv",
}


def aggregate(beats, group):
    frame = beats.groupby(group, as_index=False).agg(
        p_af=("p_af", "mean"), n_beats=("p_af", "size")
    )
    frame["modality"] = beats["modality"].iloc[0]
    frame["prediction"] = (frame.p_af > 0.5).astype(int)
    frame["p_non_af"] = 1 - frame.p_af
    return frame


def predict(arm, test, train, raw, fit_input, folder):
    import tensorflow as tf

    path = folder / "best_model.keras"
    completed = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    normalization_path = folder / "normalization.json"
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if sha(path) != completed["model_SHA256"]:
        raise ValueError(f"Checkpoint SHA256 does not match {folder / 'complete.json'}.")
    if set(normalization["fitted_subjects"]) != set(train.subject):
        raise ValueError(f"Normalization for {arm} must be fitted to the training patients.")
    indices = train.beat_id.to_numpy()
    mean = float(fit_input[indices].mean(dtype=np.float64))
    std = float(fit_input[indices].std(dtype=np.float64))
    if (
        not np.isfinite(std)
        or std <= 1e-12
        or (mean, std)
        != (
            normalization["mean"],
            normalization["std"],
        )
    ):
        raise ValueError(f"Normalization for {arm} does not match its reference training data.")
    features = ((raw[test.beat_id.to_numpy()] - mean) / std).astype("float32")[..., None]
    model = tf.keras.models.load_model(path, compile=False)
    probabilities = model.predict(features, batch_size=32, verbose=0)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite predictions for {arm}.")
    direct = model(features[:64], training=False).numpy()
    replay_error = float(np.max(np.abs(direct - probabilities[:64])))
    if replay_error >= 1e-5:
        raise ValueError(f"Checkpoint replay diverged for {arm}: {replay_error}")
    predictions = test.copy()
    predictions["modality"] = arm
    predictions[["p_non_af", "p_af"]] = probabilities
    predictions["prediction"] = (predictions.p_af > 0.5).astype(int)
    return predictions, {
        "input": arm,
        "classifier_trained_on": "direct_ppg_beats" if arm == "ppg" else "real_ecg_beats",
        "classifier_model": str(path),
        "classifier_SHA256": sha(path),
        "normalization_SHA256": sha(normalization_path),
        "checkpoint_replay_max_abs_probability_error": replay_error,
        "reference_assisted_segmentation": True,
        "threshold": 0.5,
        "threshold_operator": ">",
    }


def run(context):
    output = context.output_dir
    artifacts = context.artifacts_dir
    meta = pd.read_csv(artifacts / "data/beat_manifest.csv")
    split = meta.groupby("partition").subject.nunique().to_dict()
    if split != {"train": 22, "validation": 6, "test": 7}:
        raise ValueError(f"Unexpected patient counts: {split}.")
    if meta.groupby("subject").partition.nunique().max() != 1 or meta.start_s.min() < 180:
        raise ValueError("Beat manifest crosses a patient partition or the 180-second boundary.")
    if not np.array_equal(meta.beat_id, np.arange(len(meta))):
        raise ValueError("Beat IDs must be contiguous and ordered.")
    test = meta[meta.partition == "test"]
    train = meta[meta.partition == "train"]
    with (
        np.load(artifacts / "data/paired_beats.npz") as ecg,
        np.load(artifacts / "data_ppg/ppg_beats.npz") as ppg,
    ):
        for archive in (ecg, ppg):
            if not np.array_equal(archive["beat_id"], meta.beat_id) or not np.array_equal(
                archive["label"], meta.label
            ):
                raise ValueError("Beat IDs and labels must match between modalities and manifest.")
        inputs = {
            "ppg": (ppg["ppg"], ppg["ppg"], artifacts / "model_ppg"),
            "reconstructed_ecg": (ecg["reconstructed_ecg"], ecg["ecg"], artifacts / "model"),
        }

    statistics.initialize(output / "statistics", context.config["bootstrap_replicates"])
    summary, links, all_predictions = [], [], []
    for arm in ARMS:
        folder = output / "evaluation" / arm
        folder.mkdir(parents=True, exist_ok=True)
        beats, link = predict(arm, test, train, *inputs[arm])
        all_predictions.append(beats)
        links.append(link)
        save_json(folder / "model_link.json", link)
        views = {
            "per_beat": beats,
            "per_segment_from_beats": aggregate(beats, ["subject", "label", "r_parent_sample_id"]),
            "per_subject_from_beats": aggregate(beats, ["subject", "label"]),
        }
        for level, frame in views.items():
            frame.to_csv(folder / PREDICTION_FILES[level], index=False)
            statistics.analyze(
                "AF_180s_beat",
                "patient",
                arm,
                level,
                frame.subject,
                frame.label,
                frame.prediction,
                frame[["p_non_af", "p_af"]],
                LABELS,
            )
            counts = confusion_matrix(frame.label, frame.prediction, labels=[0, 1])
            pd.DataFrame(
                counts, index=["true_non_AF", "true_AF"], columns=["pred_non_AF", "pred_AF"]
            ).to_csv(folder / f"{level}_confusion.csv")
            tn, fp, fn, tp = map(int, counts.ravel())
            summary.append(
                {
                    "modality": arm,
                    "level": level,
                    "n_units": len(frame),
                    "n_subjects": frame.subject.nunique(),
                    "accuracy": accuracy_score(frame.label, frame.prediction),
                    "macro_f1": f1_score(
                        frame.label, frame.prediction, average="macro", zero_division=0
                    ),
                    "auroc": roc_auc_score(frame.label, frame.p_af),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )

    if links[0]["classifier_SHA256"] == links[1]["classifier_SHA256"]:
        raise ValueError("The PPG and ECG classifiers must be separate.")
    pd.concat(all_predictions).to_csv(output / "evaluation/all_beat_predictions.csv", index=False)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    pd.DataFrame(summary).to_csv(reports / "summary.csv", index=False)
    statistics.export_tables()
    save_json(
        reports / "verification.json",
        {
            "input_arms": list(ARMS),
            "classifier_links": links,
            "paired_beat_IDs_and_labels_match": True,
            "patient_partition_counts": split,
            "n_test_beats": len(test),
            "n_covered_test_segments": int(test.r_parent_sample_id.nunique()),
            "statistics_checks": statistics.CHECKS,
        },
    )
