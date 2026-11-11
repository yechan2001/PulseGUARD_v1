"""Add direct PPG inputs at existing beat boundaries without changing ECG data."""

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import fingerprint, reusable, save_json, sha


def run(context):
    ROOT = context.output_dir
    PARENT = context.upstream_dir
    CFG = context.config
    folder = ROOT / "data_ppg"
    folder.mkdir(exist_ok=True)
    fp = fingerprint(
        [
            context.config_path,
            Path(__file__),
            ROOT / "data/beat_manifest.csv",
            PARENT / "data_prepared/manifest.csv",
            PARENT / "data_prepared/paired.npz",
        ]
    )
    if reusable(folder, fp):
        print("PPG_BEATS_REUSED", flush=True)
        return
    save_json(folder / "fingerprint.json", fp)
    meta = pd.read_csv(ROOT / "data/beat_manifest.csv")
    source = pd.read_csv(PARENT / "data_prepared/manifest.csv")
    waves = np.load(PARENT / "data_prepared/paired.npz")["ppg"][:, :, 0]
    out = np.empty((len(meta), CFG["input_length"]), dtype="float32")
    for subject, g in meta.groupby("subject"):
        frames = source[
            (source.subject == subject) & (source.role == "classification")
        ].sort_values("start_s")
        windows = {int(round(r.start_s * CFG["fs"])): int(r.sample_id) for r in frames.itertuples()}
        for r in g.itertuples():
            start = int(round(r.start_s * CFG["fs"]))
            end = int(round(r.end_s * CFG["fs"]))
            first = (start // 375) * 375
            last = ((end - 1) // 375) * 375
            keys = list(range(first, last + 1, 375))
            assert all(k in windows for k in keys)
            x = waves[[windows[k] for k in keys]].reshape(-1)[start - first : end - first]
            assert len(x) == end - start and np.isfinite(x).all() and start >= 22500
            out[r.beat_id] = np.interp(
                np.linspace(0, 1, CFG["input_length"]), np.linspace(0, 1, len(x)), x
            ).astype("float32")
        print("PPG_BEATS_EXTRACTED", subject, len(g), flush=True)
    np.savez_compressed(
        folder / "ppg_beats.npz",
        ppg=out,
        label=meta.label.to_numpy(dtype="int64"),
        beat_id=meta.beat_id.to_numpy(dtype="int64"),
    )
    save_json(
        folder / "complete.json",
        {
            "passed": True,
            "n_beats": len(meta),
            "n_subjects": meta.subject.nunique(),
            "n_test_beats": int((meta.partition == "test").sum()),
            "reference_manifest_SHA256": sha(ROOT / "data/beat_manifest.csv"),
            "same_synchronous_boundaries_as_ECG": True,
            "pulse_delay_shift_seconds": 0,
            "after180s_only": True,
            "nonfinite_samples": int((~np.isfinite(out)).sum()),
        },
    )
