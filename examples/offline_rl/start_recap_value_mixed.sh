#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=/pfs/pfs-oHNwH0/lqb/vla_rl/RLinf
PYTHON_BIN=/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap/bin/python
RUN_STAMP="$(date +%Y%m%d-%H:%M:%S)"
RUN_DIR="$RLINF_ROOT/logs/value_sft/repro_recap_value_model_sft-$RUN_STAMP"

mkdir -p "$RUN_DIR"
cd "$RLINF_ROOT"
export VIRTUAL_ENV=/pfs/pfs-oHNwH0/lqb/miniconda3/envs/rlinf_recap
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export REPO_PATH="$RLINF_ROOT"
export HF_HOME=/pfs/pfs-oHNwH0/lqb/cache/huggingface
export HF_DATASETS_CACHE=/pfs/pfs-oHNwH0/lqb/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/pfs/pfs-oHNwH0/lqb/cache/transformers
export TMPDIR=/pfs/pfs-oHNwH0/lqb/tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "$RUN_DIR" > /pfs/pfs-oHNwH0/lqb/vla_rl/RLinf/logs/value_sft/current_mixed_run_dir.txt
exec "$PYTHON_BIN" \
  examples/offline_rl/advantage_labeling/recap/train_value.py \
  --config-path "$RLINF_ROOT/examples/offline_rl/config" \
  --config-name repro_recap_value_model_sft \
  "runner.logger.log_path=$RUN_DIR" \
  >"$RUN_DIR/run_value_sft.log" 2>&1
