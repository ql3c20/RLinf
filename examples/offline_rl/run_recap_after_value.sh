#!/usr/bin/env bash

# Continue the RLinf RECAP reproduction after the currently running Value Model
# reaches step 18,000:
#   1. compute advantage labels for SFT + rollout data on GPUs 0-3;
#   2. verify both label files;
#   3. train the CFG-conditioned pi0.5 policy for 30,000 steps on GPUs 0-3.

set -Eeuo pipefail

REPO_PATH="/pfs/pfs-oHNwH0/lqb/vla_rl/RLinf"
ENV_PATH="/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap"
VALUE_PID="3753869"
VALUE_RUN_DIR="${REPO_PATH}/logs/value_sft/repro_recap_value_model_sft-20261007-14:27:38"
VALUE_EXPERIMENT="recap_value_sft_only_task0_18k"
VALUE_STEP="18000"
VALUE_CHECKPOINT="${VALUE_RUN_DIR}/${VALUE_EXPERIMENT}/checkpoints/global_step_${VALUE_STEP}/actor/model_state_dict"
VALUE_WEIGHTS="${VALUE_CHECKPOINT}/full_weights.pt"
VALUE_LOG="${VALUE_RUN_DIR}/run_value_sft.log"

SFT_DATA="/pfs/pfs-oHNwH0/lqb/datasets/RECAP-Libero10-Task0-48succ-Data-git/libero10_task0_sft"
ROLLOUT_DATA="/pfs/pfs-oHNwH0/lqb/datasets/RECAP-Libero10-Task0-48succ-Data-git/libero10_task0_train"
ADVANTAGE_TAG="fail300_N10_ckpt18000_q30"
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
echo "[$(timestamp)] Expected checkpoint: ${VALUE_CHECKPOINT}"

while kill -0 "${VALUE_PID}" 2>/dev/null; do
    sleep 60
done

[[ -s "${VALUE_WEIGHTS}" ]] || fail "The step-${VALUE_STEP} Value Model checkpoint is missing or empty. Downstream stages will not start."

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
