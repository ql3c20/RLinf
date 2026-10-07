#!/usr/bin/env bash

set -Eeuo pipefail

LQB_BASE="/pfs/pfs-oHNwH0/lqb"
REPO="${LQB_BASE}/vla_rl/RLinf"
PY_ENV="${LQB_BASE}/miniconda3/envs/rlinf_pirl"
WANDB_ENV="${LQB_BASE}/.config/wandb/api_key.env"
STATE_DIR="${LQB_BASE}/vla_rl/state/pirl_4way_300_bs512"
MASTER_LOG_DIR="${LQB_BASE}/logs/pirl_4way_300_bs512"
RUN_ROOT="${LQB_BASE}/results/pirl_4way_300_bs512"
RAY_TMP="${LQB_BASE}/tmp/ray_pirl_4way"
RAY_PORT="6381"
RAY_DASHBOARD_PORT="8266"
RAY_ADDRESS="127.0.0.1:${RAY_PORT}"

mkdir -p "${STATE_DIR}" "${MASTER_LOG_DIR}" "${RUN_ROOT}" "${RAY_TMP}"
exec > >(tee -a "${MASTER_LOG_DIR}/pipeline.log") 2>&1

source "${LQB_BASE}/vla_rl/deploy/scripts/common_env.sh"
source "${PY_ENV}/bin/activate"

if [[ ! -r "${WANDB_ENV}" ]]; then
  echo "Missing private W&B environment file: ${WANDB_ENV}" >&2
  exit 1
fi
# The API key remains in this private chmod-600 file and is never embedded here.
source "${WANDB_ENV}"
export WANDB_DIR="${LQB_BASE}/wandb"
export WANDB_MODE="online"

# The dedicated Ray head advertises all physical ranks so placement 4-7 maps
# exactly to GPUs 4-7. Every experiment config pins actor/env/rollout there.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export RAY_ADDRESS
export RAY_AUTH_MODE="disabled"
export LIBERO_CONFIG_PATH="${LQB_BASE}/.config/libero_pirl"
export ROBOT_PLATFORM="LIBERO"
export LIBERO_TYPE="standard"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export TOKENIZERS_PARALLELISM="false"
export EMBODIED_PATH="${REPO}/examples/embodiment"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

ray_pid=""
cleanup() {
  local rc=$?
  if [[ -n "${ray_pid}" ]] && kill -0 "${ray_pid}" 2>/dev/null; then
    kill "${ray_pid}" 2>/dev/null || true
    wait "${ray_pid}" 2>/dev/null || true
  fi
  if (( rc != 0 )); then
    touch "${STATE_DIR}/FAILED"
    echo "[$(date -Is)] Pipeline failed with exit code ${rc}."
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

cd "${REPO}"

# Initialize LIBERO's absolute package paths once before four Ray env workers
# import it concurrently. A dedicated config also avoids racing with jobs from
# other Python environments through /root/.libero/config.yaml.
mkdir -p "${LIBERO_CONFIG_PATH}"
python - <<'PY'
import os
import yaml
import libero.libero as libero

config = {key: libero.get_libero_path(key) for key in (
    "benchmark_root", "bddl_files", "init_states", "datasets", "assets"
)}
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    if not os.path.isdir(config[key]):
        raise SystemExit(f"LIBERO path is invalid: {key}={config[key]}")
print("Validated dedicated LIBERO config:", yaml.safe_dump(config, sort_keys=True))
PY

python - "${RAY_PORT}" "${RAY_DASHBOARD_PORT}" <<'PY'
import socket, sys
for port in map(int, sys.argv[1:]):
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"Port {port} is already occupied; refusing to touch another Ray cluster.")
PY

echo "[$(date -Is)] Starting dedicated Ray at ${RAY_ADDRESS}."
ray start --head --block \
  --port="${RAY_PORT}" \
  --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --temp-dir="${RAY_TMP}" \
  --num-gpus=8 \
  --num-cpus=64 \
  --disable-usage-stats &
ray_pid=$!

for _ in $(seq 1 60); do
  if ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${ray_pid}" 2>/dev/null; then
    echo "Dedicated Ray head exited during startup." >&2
    exit 1
  fi
  sleep 2
done
ray status --address="${RAY_ADDRESS}"

run_one() {
  local config="$1"
  local experiment="$2"
  local marker="$3"
  shift 3
  local eval_overrides=("$@")

  if [[ -f "${STATE_DIR}/${marker}" ]]; then
    echo "[$(date -Is)] Skip completed experiment ${config}."
    return 0
  fi

  local stamp run_dir train_log checkpoint eval_log_dir
  stamp="$(date +'%Y%m%d-%H%M%S')"
  run_dir="${RUN_ROOT}/${stamp}-${config}"
  train_log="${run_dir}/run_embodiment.log"
  mkdir -p "${run_dir}"

  echo "[$(date -Is)] TRAIN ${config}: GPUs 4-7, global BS 512, 300 steps."
  python "${REPO}/examples/embodiment/train_embodied_agent.py" \
    --config-path "${REPO}/examples/embodiment/config/" \
    --config-name "${config}" \
    "runner.logger.log_path=${run_dir}" \
    2>&1 | tee "${train_log}"

  checkpoint="${run_dir}/${experiment}/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing final checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  printf '%s\n' "${checkpoint}" > "${STATE_DIR}/${marker}.checkpoint"

  sleep 10
  echo "[$(date -Is)] EVAL ${config}: one 48-episode Task-0 evaluation."
  bash "${REPO}/evaluations/run_eval.sh" libero repro_pi05_rl_task0_eval \
    "runner.ckpt_path=${checkpoint}" \
    "runner.logger.experiment_name=${experiment}_eval" \
    "${eval_overrides[@]}"

  eval_log_dir="$(find "${REPO}/logs" -maxdepth 1 -type d -name '*-repro_pi05_rl_task0_eval' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
  printf '%s\n' "${eval_log_dir}" > "${STATE_DIR}/${marker}.eval_dir"
  touch "${STATE_DIR}/${marker}"
  echo "[$(date -Is)] DONE ${config}; eval=${eval_log_dir}."
}

run_one repro_pi05_ppo_flow_sde_task0 pi05_task0_ppo_flow_sde 01_ppo_flow_sde_done \
  rollout.model.add_value_head=true rollout.model.num_steps=5 \
  rollout.model.openpi.value_after_vlm=true rollout.model.openpi.noise_method=flow_sde \
  rollout.model.openpi.joint_logprob=false

run_one repro_pi05_grpo_flow_sde_task0 pi05_task0_grpo_flow_sde 02_grpo_flow_sde_done \
  rollout.model.add_value_head=false rollout.model.num_steps=4 \
  rollout.model.openpi.value_after_vlm=false rollout.model.openpi.noise_method=flow_sde \
  rollout.model.openpi.joint_logprob=false

run_one repro_pi05_ppo_flow_noise_task0 pi05_task0_ppo_flow_noise 03_ppo_flow_noise_done \
  rollout.model.add_value_head=true rollout.model.num_steps=5 \
  rollout.model.openpi.value_after_vlm=true rollout.model.openpi.noise_method=flow_noise \
  +rollout.model.openpi.noise_logvar_range='[0.08,0.16]' rollout.model.openpi.joint_logprob=true

run_one repro_pi05_grpo_flow_noise_task0 pi05_task0_grpo_flow_noise 04_grpo_flow_noise_done \
  rollout.model.add_value_head=false rollout.model.num_steps=4 \
  rollout.model.openpi.value_after_vlm=false rollout.model.openpi.noise_method=flow_noise \
  +rollout.model.openpi.noise_logvar_range='[0.08,0.16]' rollout.model.openpi.joint_logprob=true

touch "${STATE_DIR}/COMPLETE"
echo "[$(date -Is)] All four train-and-evaluate experiments completed."
