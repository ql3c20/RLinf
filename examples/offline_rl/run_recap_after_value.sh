#!/usr/bin/env bash

# Continue the RLinf RECAP reproduction after the currently running Value Model
# reaches step 18,000:
#   1. select the best saved checkpoint by validation ranking quality;
#   2. compute advantage labels for SFT + rollout data on GPUs 0-3;
#   3. verify both label files;
#   4. train the CFG-conditioned pi0.5 policy for 30,000 steps on GPUs 0-3.

set -Eeuo pipefail

REPO_PATH="/pfs/pfs-oHNwH0/lqb/vla_rl/RLinf"
ENV_PATH="/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap"
VALUE_PID="3753869"
VALUE_RUN_DIR="${REPO_PATH}/logs/value_sft/repro_recap_value_model_sft-20261007-14:27:38"
VALUE_EXPERIMENT="recap_value_sft_only_task0_18k"
VALUE_CHECKPOINT_ROOT="${VALUE_RUN_DIR}/${VALUE_EXPERIMENT}/checkpoints"
VALUE_FINAL_WEIGHTS="${VALUE_CHECKPOINT_ROOT}/global_step_18000/actor/model_state_dict/full_weights.pt"
VALUE_LOG="${VALUE_RUN_DIR}/run_value_sft.log"
VALUE_TENSORBOARD_DIR="${VALUE_RUN_DIR}/tensorboard"

SFT_DATA="/pfs/pfs-oHNwH0/lqb/datasets/RECAP-Libero10-Task0-48succ-Data-git/libero10_task0_sft"
ROLLOUT_DATA="/pfs/pfs-oHNwH0/lqb/datasets/RECAP-Libero10-Task0-48succ-Data-git/libero10_task0_train"
PIPELINE_LOG_DIR="${REPO_PATH}/logs/recap_pipeline"
PIPELINE_LOG="${PIPELINE_LOG_DIR}/after_value_$(date +'%Y%m%d-%H%M%S').log"

mkdir -p "${PIPELINE_LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() {
    date '+%F %T'
}

fail() {
    echo "[$(timestamp)] ERROR: $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    echo "[$(timestamp)] Pipeline stopped with exit code ${exit_code}. See ${PIPELINE_LOG}" >&2
    exit "${exit_code}"
}
trap on_error ERR

echo "[$(timestamp)] Waiting for Value Model PID ${VALUE_PID} to finish."
echo "[$(timestamp)] Expected final checkpoint: ${VALUE_FINAL_WEIGHTS}"

while kill -0 "${VALUE_PID}" 2>/dev/null; do
    sleep 60
done

[[ -s "${VALUE_FINAL_WEIGHTS}" ]] || fail "The step-18000 Value Model checkpoint is missing or empty. Downstream stages will not start."

if tail -n 4000 "${VALUE_LOG}" | grep -Eq 'Traceback|Error executing job|CUDA out of memory'; then
    fail "The Value Model log contains a fatal error near the end."
fi

echo "[$(timestamp)] Value Model finished and checkpoint verification passed."

source "${ENV_PATH}/bin/activate"
export REPO_PATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/pfs/pfs-oHNwH0/lqb/cache/huggingface"
export HF_DATASETS_CACHE="/pfs/pfs-oHNwH0/lqb/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/pfs/pfs-oHNwH0/lqb/cache/transformers"
export TMPDIR="/pfs/pfs-oHNwH0/lqb/tmp"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

cd "${REPO_PATH}"

# Spearman is the primary model-selection metric because Step 3 uses value
# ranking to identify the top-30% samples. To avoid choosing a checkpoint for a
# negligible noisy Spearman gain, checkpoints within 0.005 of the best
# Spearman are treated as tied and the lower validation MAE wins.
VALUE_CHECKPOINT="$({
python - "${VALUE_CHECKPOINT_ROOT}" "${VALUE_TENSORBOARD_DIR}" <<'PY'
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

checkpoint_root = Path(sys.argv[1])
tensorboard_dir = Path(sys.argv[2])
spearman_tolerance = 0.005

checkpoints = []
for path in checkpoint_root.glob("global_step_*"):
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    weights = path / "actor" / "model_state_dict" / "full_weights.pt"
    if match and weights.is_file() and weights.stat().st_size > 0:
        checkpoints.append((int(match.group(1)), path))
if not checkpoints:
    raise RuntimeError(f"No complete checkpoints found under {checkpoint_root}")

events = EventAccumulator(str(tensorboard_dir), size_guidance={"scalars": 0})
events.Reload()
required_tags = ("eval/value_spearman", "eval/mae", "eval/loss")
available = set(events.Tags().get("scalars", []))
missing = set(required_tags).difference(available)
if missing:
    raise RuntimeError(f"Missing TensorBoard validation metrics: {sorted(missing)}")

series = {tag: {event.step: event.value for event in events.Scalars(tag)} for tag in required_tags}
candidates = []
for checkpoint_step, checkpoint_path in sorted(checkpoints):
    # RLinf saves global_step=N but logs that iteration with zero-based step N-1.
    metric_step = checkpoint_step - 1
    if all(metric_step in series[tag] for tag in required_tags):
        candidates.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": checkpoint_path,
                "spearman": series["eval/value_spearman"][metric_step],
                "mae": series["eval/mae"][metric_step],
                "loss": series["eval/loss"][metric_step],
            }
        )
if not candidates:
    raise RuntimeError("No saved checkpoint has matching validation metrics")

best_spearman = max(item["spearman"] for item in candidates)
near_best = [
    item for item in candidates
    if item["spearman"] >= best_spearman - spearman_tolerance
]
selected = min(near_best, key=lambda item: (item["mae"], item["loss"], item["checkpoint_step"]))

for item in candidates:
    marker = "SELECTED" if item is selected else "candidate"
    print(
        f"[{marker}] step={item['checkpoint_step']} "
        f"eval_spearman={item['spearman']:.6f} "
        f"eval_mae={item['mae']:.6f} eval_loss={item['loss']:.6f}",
        file=sys.stderr,
    )

selected_model_dir = selected["checkpoint_path"] / "actor" / "model_state_dict"
(checkpoint_root / "selected_value_checkpoint.txt").write_text(
    f"checkpoint={selected_model_dir}\n"
    f"step={selected['checkpoint_step']}\n"
    f"eval_value_spearman={selected['spearman']:.9f}\n"
    f"eval_mae={selected['mae']:.9f}\n"
    f"eval_loss={selected['loss']:.9f}\n",
    encoding="utf-8",
)
print(selected_model_dir)
PY
} 2> >(tee -a "${PIPELINE_LOG}" >&2))"

[[ -s "${VALUE_CHECKPOINT}/full_weights.pt" ]] || fail "Selected Value Model checkpoint is invalid: ${VALUE_CHECKPOINT}"
SELECTED_CHECKPOINT_NAME="$(basename "$(dirname "$(dirname "${VALUE_CHECKPOINT}")")")"
SELECTED_STEP="${SELECTED_CHECKPOINT_NAME#global_step_}"
[[ "${SELECTED_STEP}" =~ ^[0-9]+$ ]] || fail "Could not parse selected checkpoint step from ${VALUE_CHECKPOINT}"
ADVANTAGE_TAG="fail300_N10_ckpt${SELECTED_STEP}_q30"

echo "[$(timestamp)] Selected Value Model checkpoint: ${VALUE_CHECKPOINT}"
echo "[$(timestamp)] Advantage tag: ${ADVANTAGE_TAG}"

echo "[$(timestamp)] Stage 3/4: computing advantages for SFT and rollout datasets."
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
    repro_recap_compute_advantages \
    --nproc 4 \
    "advantage.value_checkpoint=${VALUE_CHECKPOINT}" \
    "advantage.tag=${ADVANTAGE_TAG}"

SFT_LABELS="${SFT_DATA}/meta/advantages_${ADVANTAGE_TAG}.parquet"
ROLLOUT_LABELS="${ROLLOUT_DATA}/meta/advantages_${ADVANTAGE_TAG}.parquet"
[[ -s "${SFT_LABELS}" ]] || fail "SFT advantage labels were not generated."
[[ -s "${ROLLOUT_LABELS}" ]] || fail "Rollout advantage labels were not generated."

python - "${SFT_LABELS}" "${ROLLOUT_LABELS}" <<'PY'
import sys
import pandas as pd

for path in sys.argv[1:]:
    frame = pd.read_parquet(path)
    required = {"advantage", "advantage_continuous"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    if len(frame) == 0:
        raise RuntimeError(f"{path}: empty advantage table")
    print(
        f"verified {path}: samples={len(frame)}, "
        f"positive_rate={frame['advantage'].astype(float).mean():.4f}, "
        f"advantage_min={frame['advantage_continuous'].min():.6f}, "
        f"advantage_max={frame['advantage_continuous'].max():.6f}"
    )
PY

echo "[$(timestamp)] Advantage-label verification passed."
echo "[$(timestamp)] Stage 4/4: starting pi0.5 CFG training for 30,000 optimizer steps."

bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
    repro_cfg_rl_openpi \
    "data.advantage_tag=${ADVANTAGE_TAG}" \
    "runner.max_steps=30000" \
    "runner.save_interval=3000" \
    "actor.optim.total_training_steps=30000"

echo "[$(timestamp)] RECAP pipeline completed successfully."
echo "[$(timestamp)] Pipeline log: ${PIPELINE_LOG}"
