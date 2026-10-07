#!/usr/bin/env bash
set -Eeuo pipefail

REPO_PATH=/pfs/pfs-oHNwH0/lqb/vla_rl/RLinf
ENV_PATH=/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap
WANDB_ENV_FILE=/pfs/pfs-oHNwH0/lqb/.config/wandb/api_key.env
RUN_STAMP="$(date +%Y%m%d-%H:%M:%S)"
VALUE_RUN_DIR="$REPO_PATH/logs/value_sft/repro_recap_value_model_sft-$RUN_STAMP"
MASTER_LOG_DIR="$REPO_PATH/logs/recap_quick_pipeline"
MASTER_LOG="$MASTER_LOG_DIR/pipeline-$RUN_STAMP.log"

mkdir -p "$MASTER_LOG_DIR" "$VALUE_RUN_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

source "$ENV_PATH/bin/activate"
if [[ -r "$WANDB_ENV_FILE" ]]; then
  # API credentials stay in this mode-600 file outside the repository.
  source "$WANDB_ENV_FILE"
fi

export REPO_PATH
export VIRTUAL_ENV="$ENV_PATH"
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/pfs/pfs-oHNwH0/lqb/cache/huggingface
export HF_DATASETS_CACHE=/pfs/pfs-oHNwH0/lqb/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/pfs/pfs-oHNwH0/lqb/cache/transformers
export TMPDIR=/pfs/pfs-oHNwH0/lqb/tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$REPO_PATH"

echo "[$(date '+%F %T')] Step 1/4: computing trajectory returns."
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  repro_recap_compute_returns

echo "[$(date '+%F %T')] Step 2/4: training the mixed-data Value Model for 6,000 steps."
RUN_DIR_OVERRIDE="$VALUE_RUN_DIR" VALUE_TEE=1 \
  bash examples/offline_rl/start_recap_value_mixed.sh

echo "[$(date '+%F %T')] Steps 3-4: selecting the Value checkpoint, labeling advantages, training CFG for 10,000 steps, then evaluating."
VALUE_RUN_DIR="$VALUE_RUN_DIR" VALUE_PID="" \
  bash examples/offline_rl/run_recap_after_value.sh

echo "[$(date '+%F %T')] Quick RECAP pipeline completed."
echo "Master log: $MASTER_LOG"
