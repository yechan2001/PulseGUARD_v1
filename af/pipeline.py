"""Run the two-arm AF experiment against local reconstruction artifacts."""

import json
from dataclasses import dataclass
from pathlib import Path

from ..io import configure_runtime, reject_output_overlap, save_json

ARMS = ("ppg", "reconstructed_ecg")
STAGES = ("prepare", "prepare_ppg", "train", "train_ppg", "evaluate", "verify", "report")
REPLAY_STAGES = ("evaluate", "verify", "report")


@dataclass(frozen=True)
class Context:
    config_path: Path
    upstream_dir: Path
    output_dir: Path
    artifacts_dir: Path
    config: dict


def run(args):
    upstream = args.upstream_dir.resolve()
    output = args.output_dir.resolve()
    artifacts = args.artifacts_dir.resolve() if args.artifacts_dir else output
    config_path = args.config.resolve()
    reject_output_overlap(
        output, upstream, args.artifacts_dir.resolve() if args.artifacts_dir else None
    )
    for path in (upstream / "data_prepared/manifest.csv", upstream / "data_prepared/paired.npz"):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["classifier_inputs"]) != ARMS:
        raise ValueError(f"AF evaluation requires these two inputs: {ARMS}.")
    if config["fs"] != 125:
        raise ValueError("This dataset uses 125 Hz and 375 samples per input window.")

    stages = (
        (REPLAY_STAGES if args.artifacts_dir else STAGES) if args.stage == "all" else (args.stage,)
    )
    if args.artifacts_dir and any(stage not in REPLAY_STAGES for stage in stages):
        raise ValueError("--artifacts-dir supports evaluate, verify, report, or all (replay).")
    if (output / "evaluation/ecg").exists():
        raise ValueError("Choose a fresh output directory for the two-arm experiment.")

    configure_runtime(output)
    context = Context(config_path, upstream, output, artifacts, config)
    if any(stage in stages for stage in ("train", "train_ppg", "evaluate")):
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.get_logger().setLevel("ERROR")

    from . import (
        evaluation,
        prepare_beats,
        prepare_ppg_beats,
        reporting,
        training,
        verify_beat_data,
    )

    actions = {
        "prepare": prepare_beats.run,
        "prepare_ppg": prepare_ppg_beats.run,
        "train": lambda ctx: training.run(ctx, "ecg"),
        "train_ppg": lambda ctx: training.run(ctx, "ppg"),
        "evaluate": evaluation.run,
        "verify": verify_beat_data.run,
        "report": reporting.run,
    }
    for stage in stages:
        print(f"AF stage {stage}", flush=True)
        actions[stage](context)
    save_json(output / "run_config.json", config)
