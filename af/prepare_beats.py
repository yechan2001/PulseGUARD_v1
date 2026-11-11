"""Paired, reference-assisted single-beat extraction. No test-tuned thresholds."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from ..io import fingerprint, reusable, save_json


def qrs(x, CFG):
    fs = CFG["fs"]
    c = CFG["detector"]
    z = sosfiltfilt(
        butter(c["order"], c["bandpass_Hz"], btype="bandpass", fs=fs, output="sos"),
        np.asarray(x, dtype=float),
    )
    width = max(1, int(c["energy_average_s"] * fs))
    envelope = np.convolve(np.diff(z, prepend=z[0]) ** 2, np.ones(width) / width, mode="same")
    candidates, _ = find_peaks(
        envelope,
        distance=int(c["minimum_distance_s"] * fs),
        prominence=max(1e-12, c["prominence_p95_fraction"] * np.quantile(envelope, 0.95)),
    )
    refined = []
    radius = int(c["refinement_s"] * fs)
    for p in candidates:
        a = max(0, p - radius)
        b = min(len(z), p + radius + 1)
        point = a + int(np.argmax(np.abs(z[a:b])))
        if refined and point - refined[-1] < int(c["minimum_distance_s"] * fs):
            if abs(z[point]) > abs(z[refined[-1]]):
                refined[-1] = point
        else:
            refined.append(point)
    return np.asarray(refined, dtype=int)


def run(context):
    ROOT = context.output_dir
    PARENT = context.upstream_dir
    CFG = context.config
    dest = ROOT / "data"
    dest.mkdir(exist_ok=True)
    meta = pd.read_csv(PARENT / "data_prepared/manifest.csv")
    paths = [
        context.config_path,
        Path(__file__),
        PARENT / "data_prepared/manifest.csv",
        PARENT / "data_prepared/paired.npz",
    ]
    paths += sorted((PARENT / "personalization").glob("*/generated_ecg.npz"))
    fp = fingerprint(paths)
    if reusable(dest, fp):
        print("BEAT_DATA_REUSED", flush=True)
        return
    save_json(dest / "fingerprint.json", fp)
    paired = np.load(PARENT / "data_prepared/paired.npz")
    real = paired["ecg"][:, :, 0]
    rows = []
    beats = []
    recon = []
    exclusions = []
    runs = []
    qrs_rows = []
    fs = CFG["fs"]
    selected = meta[meta.role == "classification"].copy()
    assert selected.start_s.min() >= 180
    for subject, g in selected.groupby("subject", sort=True):
        g = g.sort_values("start_s")
        generated = np.load(PARENT / "personalization" / subject / "generated_ecg.npz")
        lookup = {int(s): i for i, s in enumerate(generated["sample_id"])}
        chunks = np.split(
            np.arange(len(g)), np.flatnonzero(np.diff(g.start_s.to_numpy()) > 3.00001) + 1
        )
        count_before = len(rows)
        for chunk_id, positions in enumerate(chunks):
            f = g.iloc[positions]
            ids = f.sample_id.to_numpy(dtype=int)
            x = real[ids].reshape(-1)
            z = generated["ecg_restored"][[lookup[int(s)] for s in ids]].reshape(-1)
            assert len(x) == len(z) and np.isfinite(x).all() and np.isfinite(z).all()
            peaks = qrs(x, CFG)
            base = float(f.start_s.iloc[0])
            kept = 0
            for i, r in enumerate(peaks):
                qrs_rows.append(
                    {
                        "subject": subject,
                        "partition": f.partition.iloc[0],
                        "label": int(f.label.iloc[0]),
                        "run_id": chunk_id,
                        "r_time_s": base + r / fs,
                    }
                )
                reason = None
                a = b = None
                if i == 0 or i == len(peaks) - 1:
                    reason = "first_or_last_peak_no_two_neighbors"
                else:
                    a = int((peaks[i - 1] + r) // 2)
                    b = int((r + peaks[i + 1]) // 2)
                    if not CFG["minimum_beat_s"] <= (b - a) / fs <= CFG["maximum_beat_s"]:
                        reason = "beat_duration_outside_fixed_range"
                    elif np.ptp(x[a:b]) <= 1e-10:
                        reason = "flat_real_beat"
                if reason:
                    exclusions.append(
                        {
                            "subject": subject,
                            "partition": f.partition.iloc[0],
                            "label": int(f.label.iloc[0]),
                            "run_id": chunk_id,
                            "r_time_s": base + r / fs,
                            "reason": reason,
                        }
                    )
                    continue

                def interp(v):
                    return np.interp(
                        np.linspace(0, 1, CFG["input_length"]), np.linspace(0, 1, len(v)), v
                    ).astype("float32")

                rows.append(
                    {
                        "beat_id": len(rows),
                        "subject": subject,
                        "patient_id": subject,
                        "partition": f.partition.iloc[0],
                        "label": int(f.label.iloc[0]),
                        "run_id": chunk_id,
                        "r_time_s": base + r / fs,
                        "start_s": base + a / fs,
                        "end_s": base + b / fs,
                        "duration_s": (b - a) / fs,
                        "rr_before_s": (r - peaks[i - 1]) / fs,
                        "rr_after_s": (peaks[i + 1] - r) / fs,
                        "r_parent_sample_id": int(ids[r // 375]),
                        "first_parent_sample_id": int(ids[a // 375]),
                        "last_parent_sample_id": int(ids[(b - 1) // 375]),
                        "crosses_generation_window": bool(a // 375 != (b - 1) // 375),
                    }
                )
                beats.append(interp(x[a:b]))
                recon.append(interp(z[a:b]))
                kept += 1
            runs.append(
                {
                    "subject": subject,
                    "partition": f.partition.iloc[0],
                    "label": int(f.label.iloc[0]),
                    "run_id": chunk_id,
                    "start_s": base,
                    "end_s": base + len(x) / fs,
                    "n_windows": len(ids),
                    "n_detected_peaks": len(peaks),
                    "n_retained_beats": kept,
                    "detected_peak_rate_bpm": len(peaks) / (len(x) / fs) * 60,
                }
            )
        print("EXTRACTED", subject, "beats", len(rows) - count_before, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(dest / "beat_manifest.csv", index=False)
    pd.DataFrame(exclusions).to_csv(dest / "excluded_beats.csv", index=False)
    pd.DataFrame(runs).to_csv(dest / "detection_runs.csv", index=False)
    pd.DataFrame(qrs_rows).to_csv(dest / "detected_peaks.csv", index=False)
    for p, n in [("train", 22), ("validation", 6), ("test", 7)]:
        assert result[result.partition == p].subject.nunique() == n
    subjects = {
        p: set(result[result.partition == p].subject) for p in ["train", "validation", "test"]
    }
    assert not (
        subjects["train"] & subjects["validation"]
        or subjects["train"] & subjects["test"]
        or subjects["validation"] & subjects["test"]
    )
    assert (
        result.start_s.min() >= 180
        and (result.start_s < result.r_time_s).all()
        and (result.r_time_s < result.end_s).all()
    )
    counts = (
        result.groupby(["partition", "label"])
        .agg(n_beats=("beat_id", "size"), n_subjects=("subject", "nunique"))
        .reset_index()
    )
    counts.to_csv(dest / "class_counts.csv", index=False)
    result.groupby(["partition", "subject", "label"]).size().rename("n_beats").reset_index().to_csv(
        dest / "subject_counts.csv", index=False
    )
    np.savez_compressed(
        dest / "paired_beats.npz",
        ecg=np.asarray(beats),
        reconstructed_ecg=np.asarray(recon),
        label=result.label.to_numpy(dtype="int64"),
        beat_id=result.beat_id.to_numpy(dtype="int64"),
    )
    save_json(
        dest / "complete.json",
        {
            "passed": True,
            "n_beats": len(result),
            "n_subjects": 35,
            "n_excluded_peaks": len(exclusions),
            "patient_split": {p: sorted(v) for p, v in subjects.items()},
            "same_reference_boundaries_for_real_and_reconstructed": True,
            "classification_only_after_180s": True,
            "class_counts": counts.to_dict("records"),
            "detector_parameters_fixed_before_fitting": True,
            "expert_R_peak_annotations_available": False,
        },
    )
    print(counts.to_string(index=False), flush=True)
