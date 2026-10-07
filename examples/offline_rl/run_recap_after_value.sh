#!/usr/bin/env bash

# Continue the RLinf RECAP reproduction after the currently running Value Model
# reaches step 6,000:
#   1. select the best saved checkpoint by validation ranking quality;
#   2. compute advantage labels for SFT + rollout data on GPUs 0-3;
#   3. verify both label files;
#   4. train the CFG-conditioned pi0.5 policy for 10,000 steps on GPUs 0-3.
#   5. evaluate the final CFG checkpoint on all 50 LIBERO-10 Task-0 states.

set -Eeuo pipefail

REPO_PATH="/pfs/pfs-oHNwH0/lqb/vla_rl/RLinf"
ENV_PATH="/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap"
WANDB_ENV_FILE="/pfs/pfs-oHNwH0/lqb/.config/wandb/api_key.env"
VALUE_PID="${VALUE_PID:-}"
VALUE_RUN_DIR="${VALUE_RUN_DIR:?VALUE_RUN_DIR must point to the completed Value run}"
VALUE_EXPERIMENT="recap_value_sft_rollout_task0_6k"
VALUE_CHECKPOINT_ROOT="${VALUE_RUN_DIR}/${VALUE_EXPERIMENT}/checkpoints"
VALUE_FINAL_WEIGHTS="${VALUE_CHECKPOINT_ROOT}/global_step_6000/actor/model_state_dict/full_weights.pt"
VALUE_LOG="${VALUE_RUN_DIR}/run_value_sft.log"
VALUE_EVAL_DIR="${VALUE_RUN_DIR}/checkpoint_eval"

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

echo "[$(timestamp)] Checking completed Value Model run ${VALUE_RUN_DIR}."
echo "[$(timestamp)] Expected final checkpoint: ${VALUE_FINAL_WEIGHTS}"

if [[ -n "${VALUE_PID}" ]]; then
    while kill -0 "${VALUE_PID}" 2>/dev/null; do
        sleep 60
    done
fi

[[ -s "${VALUE_FINAL_WEIGHTS}" ]] || fail "The step-6000 Value Model checkpoint is missing or empty. Downstream stages will not start."

if tail -n 4000 "${VALUE_LOG}" | grep -Eq 'Traceback|Error executing job|CUDA out of memory'; then
    fail "The Value Model log contains a fatal error near the end."
fi

echo "[$(timestamp)] Value Model finished and checkpoint verification passed."

source "${ENV_PATH}/bin/activate"
if [[ -r "${WANDB_ENV_FILE}" ]]; then
    # Kept outside the repository with mode 600; never commit API credentials.
    source "${WANDB_ENV_FILE}"
fi
export REPO_PATH
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_HOME="/pfs/pfs-oHNwH0/lqb/cache/huggingface"
export HF_DATASETS_CACHE="/pfs/pfs-oHNwH0/lqb/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/pfs/pfs-oHNwH0/lqb/cache/transformers"
export TMPDIR="/pfs/pfs-oHNwH0/lqb/tmp"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TMPDIR}"

cd "${REPO_PATH}"

# Re-evaluate every saved checkpoint on the held-out mixed Task-0 eval set
# (27 successful and 37 failed episodes), then select by ranking quality.
VALUE_CHECKPOINT="$({
python examples/offline_rl/advantage_labeling/recap/evaluate_value_checkpoints.py \
    --checkpoint-root "${VALUE_CHECKPOINT_ROOT}" \
    --repo-path "${REPO_PATH}" \
    --config-name repro_recap_value_model_sft \
    --output "${VALUE_EVAL_DIR}" \
    --eval-subset-name task0_mixed_success_failure_64
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
echo "[$(timestamp)] Stage 4/4: starting pi0.5 CFG training for 10,000 optimizer steps."

CFG_RUN_DIR="${REPO_PATH}/logs/cfg_rl/repro_cfg_rl_openpi-quick-$(date +'%Y%m%d-%H%M%S')"
CFG_EXPERIMENT="recap_cfg_task0_quick_10k"

bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
    repro_cfg_rl_openpi \
    "data.advantage_tag=${ADVANTAGE_TAG}" \
    "runner.logger.log_path=${CFG_RUN_DIR}" \
    "runner.logger.experiment_name=${CFG_EXPERIMENT}" \
    "runner.max_steps=10000" \
    "runner.save_interval=2000" \
    "actor.optim.lr_warmup_steps=1000" \
    "actor.optim.total_training_steps=10000"

CFG_CHECKPOINT="${CFG_RUN_DIR}/${CFG_EXPERIMENT}/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt"
[[ -s "${CFG_CHECKPOINT}" ]] || fail "The final CFG checkpoint is missing or empty: ${CFG_CHECKPOINT}"

echo "[$(timestamp)] CFG training completed and final checkpoint verification passed."
echo "[$(timestamp)] Evaluating LIBERO-10 Task 0 on all 50 official initial states."

bash evaluations/run_eval.sh libero repro_recap_cfg_task0_eval \
    "runner.ckpt_path=${CFG_CHECKPOINT}" \
    "env.eval.rollout_epoch=1"

echo "[$(timestamp)] RECAP training and Task-0 evaluation completed successfully."
echo "[$(timestamp)] Final CFG checkpoint: ${CFG_CHECKPOINT}"
echo "[$(timestamp)] Pipeline log: ${PIPELINE_LOG}"
