#!/usr/bin/env python3
"""Evaluate saved RECAP Value checkpoints on the configured held-out split."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


METRIC_MARKER = "__VALUE_EVAL_JSON__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--config-name", default="repro_recap_value_model_sft")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spearman-tolerance", type=float, default=0.005)
    return parser.parse_args()


def discover_checkpoints(root: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        weights = path / "actor" / "model_state_dict" / "full_weights.pt"
        if match and weights.is_file() and weights.stat().st_size > 0:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        raise RuntimeError(f"No complete checkpoints found under {root}")
    return sorted(checkpoints)


def evaluate_checkpoint(
    repo_path: Path,
    config_name: str,
    step: int,
    checkpoint_path: Path,
    output_root: Path,
) -> dict[str, float | int | str]:
    eval_dir = output_root / f"global_step_{step}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(
            repo_path
            / "examples/offline_rl/advantage_labeling/recap/train_value.py"
        ),
        "--config-path",
        str(repo_path / "examples/offline_rl/config"),
        "--config-name",
        config_name,
        "runner.eval_only=true",
        f"runner.resume_dir={checkpoint_path}",
        f"runner.logger.log_path={eval_dir}",
        f"runner.logger.experiment_name=success_eval_step_{step}",
        "runner.logger.logger_backends=[]",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = eval_dir / "eval.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed for step {step}; see {log_path}"
        )

    marker_lines = [
        line for line in completed.stdout.splitlines() if line.startswith(METRIC_MARKER)
    ]
    if not marker_lines:
        raise RuntimeError(
            f"Evaluation metrics marker missing for step {step}; see {log_path}"
        )
    metrics = json.loads(marker_lines[-1][len(METRIC_MARKER) :])
    required = ("value_spearman", "mae", "loss")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(
            f"Step {step} is missing metrics {missing}; see {log_path}"
        )
    return {
        "checkpoint_step": step,
        "checkpoint_path": str(checkpoint_path),
        "spearman": float(metrics["value_spearman"]),
        "mae": float(metrics["mae"]),
        "loss": float(metrics["loss"]),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [
        evaluate_checkpoint(
            repo_path=args.repo_path,
            config_name=args.config_name,
            step=step,
            checkpoint_path=path,
            output_root=args.output,
        )
        for step, path in discover_checkpoints(args.checkpoint_root)
    ]

    best_spearman = max(item["spearman"] for item in candidates)
    near_best = [
        item
        for item in candidates
        if item["spearman"] >= best_spearman - args.spearman_tolerance
    ]
    selected = min(
        near_best,
        key=lambda item: (
            item["mae"],
            item["loss"],
            item["checkpoint_step"],
        ),
    )

    for item in candidates:
        marker = "SELECTED" if item is selected else "candidate"
        print(
            f"[{marker}] step={item['checkpoint_step']} "
            f"success_eval_spearman={item['spearman']:.6f} "
            f"success_eval_mae={item['mae']:.6f} "
            f"success_eval_loss={item['loss']:.6f}",
            file=sys.stderr,
        )

    summary_path = args.output / "success_eval_metrics.json"
    summary_path.write_text(
        json.dumps(
            {"candidates": candidates, "selected": selected},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    selected_model_dir = (
        Path(selected["checkpoint_path"]) / "actor" / "model_state_dict"
    )
    (args.checkpoint_root / "selected_value_checkpoint.txt").write_text(
        f"checkpoint={selected_model_dir}\n"
        f"step={selected['checkpoint_step']}\n"
        "eval_subset=task0_success_only_27\n"
        f"eval_value_spearman={selected['spearman']:.9f}\n"
        f"eval_mae={selected['mae']:.9f}\n"
        f"eval_loss={selected['loss']:.9f}\n",
        encoding="utf-8",
    )
    print(selected_model_dir)


if __name__ == "__main__":
    main()
