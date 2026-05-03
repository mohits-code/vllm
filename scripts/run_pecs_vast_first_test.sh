#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [results_dir]" >&2
  exit 1
fi

: "${MODEL:?set MODEL to the model id or local path}"
: "${PECS_PREDICTOR_PATH:?set PECS_PREDICTOR_PATH to the checkpoint directory}"

results_dir="${1:-${RESULTS_DIR:-pecs_vast_first_test}}"
mkdir -p "${results_dir}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_pecs_control_case.sh}"
BENCHMARK_CMD_TEMPLATE="${BENCHMARK_CMD_TEMPLATE:-}"
RUN_EP_CASES="${RUN_EP_CASES:-0}"
GPU_COUNT="${GPU_COUNT:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')}"

if [[ "${RUN_EP_CASES}" == "1" ]]; then
  cases=(baseline pecs pecs_eplb)
else
  cases=(baseline pecs)
fi

if [[ "${RUN_EP_CASES}" == "1" && "${GPU_COUNT}" -lt 2 ]]; then
  echo "[pecs-vast-first-test] RUN_EP_CASES=1 requires at least 2 visible GPUs" >&2
  exit 1
fi

wait_for_server() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://${host}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

shutdown_server() {
  local pid="${1:-}"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
}

server_pid=""
trap 'shutdown_server "${server_pid}"' EXIT

echo "[pecs-vast-first-test] gpu_count=${GPU_COUNT} results_dir=${results_dir}"
"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count())" \
  | tee "${results_dir}/torch_env.txt"

for idx in "${!cases[@]}"; do
  case_name="${cases[$idx]}"
  port=$((BASE_PORT + idx))
  case_dir="${results_dir}/${case_name}"
  mkdir -p "${case_dir}"

  server_log="${case_dir}/server.log"
  bench_log="${case_dir}/benchmark.log"

  extra_env=()
  if [[ "${case_name}" == "pecs_eplb" ]]; then
    extra_env+=(TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}")
    extra_env+=(DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}")
  fi

  echo "[pecs-vast-first-test] launching ${case_name} on ${HOST}:${port}"
  env \
    MODEL="${MODEL}" \
    PECS_PREDICTOR_PATH="${PECS_PREDICTOR_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    HOST="${HOST}" \
    PORT="${port}" \
    "${extra_env[@]}" \
    "${RUN_SCRIPT}" "${case_name}" >"${server_log}" 2>&1 &
  server_pid="$!"

  if ! wait_for_server "${HOST}" "${port}" "${SERVER_WAIT_SECONDS}"; then
    echo "[pecs-vast-first-test] server failed to become healthy for ${case_name}" >&2
    shutdown_server "${server_pid}"
    server_pid=""
    exit 1
  fi

  if [[ -n "${BENCHMARK_CMD_TEMPLATE}" ]]; then
    benchmark_cmd="${BENCHMARK_CMD_TEMPLATE//\{port\}/${port}}"
    benchmark_cmd="${benchmark_cmd//\{host\}/${HOST}}"
    benchmark_cmd="${benchmark_cmd//\{case\}/${case_name}}"
    echo "[pecs-vast-first-test] benchmark ${case_name}: ${benchmark_cmd}"
    bash -lc "${benchmark_cmd}" >"${bench_log}" 2>&1
  else
    echo "[pecs-vast-first-test] no benchmark configured for ${case_name}" | tee "${bench_log}"
  fi

  shutdown_server "${server_pid}"
  server_pid=""
done

echo "[pecs-vast-first-test] complete -> ${results_dir}"
