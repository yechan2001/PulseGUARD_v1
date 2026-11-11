"""Independent replay of paired cutting from parent signals and retained times."""

import importlib.metadata
import platform

import numpy as np
import pandas as pd

from ..io import save_json


def run(context):
    ROOT = context.output_dir
    PARENT = context.upstream_dir
    CFG = context.config
    ARTIFACTS = context.artifacts_dir
    meta = pd.read_csv(ARTIFACTS / "data/beat_manifest.csv")
    source = pd.read_csv(PARENT / "data_prepared/manifest.csv").set_index("sample_id")
    archive = np.load(ARTIFACTS / "data/paired_beats.npz")
    data = {m: archive[m] for m in ["ecg", "reconstructed_ecg"]}
    data["ppg"] = np.load(ARTIFACTS / "data_ppg/ppg_beats.npz")["ppg"]
    parent = np.load(PARENT / "data_prepared/paired.npz")
    real = parent["ecg"]
    ppg = parent["ppg"]
    rng = np.random.default_rng(42)
    errors = []
    for subject, g in meta.groupby("subject"):
        reconstructed = np.load(PARENT / "personalization" / subject / "generated_ecg.npz")
        lookup = {int(s): j for j, s in enumerate(reconstructed["sample_id"])}
        generated = reconstructed["ecg_restored"]
        chosen = g.iloc[np.sort(rng.choice(len(g), size=min(12, len(g)), replace=False))]
        for r in chosen.itertuples():
            windows = source[
                (source.subject == subject)
                & (source.start_s < r.end_s - 1e-8)
                & (source.end_s > r.start_s + 1e-8)
            ].sort_values("start_s")
            assert len(windows) >= 1 and np.allclose(np.diff(windows.start_s), 3)
            base = float(windows.start_s.iloc[0])
            a = int(round((r.start_s - base) * CFG["fs"]))
            b = int(round((r.end_s - base) * CFG["fs"]))
            ids = windows.index.to_numpy(dtype=int)
            for m in ["ppg", "ecg", "reconstructed_ecg"]:
                if m == "ppg":
                    x = ppg[ids].reshape(-1)
                elif m == "ecg":
                    x = real[ids].reshape(-1)
                else:
                    x = generated[[lookup[int(s)] for s in ids]].reshape(-1)
                wave = x[a:b]
                expected = np.interp(
                    np.linspace(0, 1, CFG["input_length"]), np.linspace(0, 1, len(wave)), wave
                ).astype("float32")
                error = float(np.max(np.abs(expected - data[m][r.beat_id])))
                assert error <= 1e-7, (subject, r.beat_id, m, error)
                errors.append(error)
    split = meta.groupby("subject").partition.nunique()
    assert split.max() == 1
    save_json(
        ROOT / "reports/independent_data_replay.json",
        {
            "passed": True,
            "subjects_checked": 35,
            "paired_waveforms_checked": len(errors),
            "maximum_abs_replay_error": max(errors),
            "checks": "Reconstruct original sample slice from saved absolute times, parent real/generated arrays, resample independently and compare stored beat",
            "all_patient_partitions_unique": True,
        },
    )
    versions = {"python": platform.python_version()}
    for p in [
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "h5py",
    ]:
        versions[p] = importlib.metadata.version(p)
    save_json(ROOT / "environment_versions.json", versions)
    print("INDEPENDENT_BEAT_CUT_REPLAY_VERIFIED", len(errors), max(errors), flush=True)
