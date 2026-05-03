#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [results_dir]" >&2
  exit 1
fi

: "${MODEL:?set MODEL to the model id or local path}"

results_dir="${1:-${RESULTS_DIR:-pecs_cluster_results}}"
mkdir -p "${results_dir}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8000}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-600}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_pecs_control_case.sh}"
BENCHMARK_CMD_TEMPLATE="${BENCHMARK_CMD_TEMPLATE:-}"

if [[ -n "${PECS_CASES:-}" ]]; then
  # shellcheck disable=SC2206
  cases=( ${PECS_CASES} )
else
  cases=(baseline pecs pecs_dbo pecs_eplb pecs_dbo_eplb)
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

for idx in "${!cases[@]}"; do
  case_name="${cases[$idx]}"
  port=$((BASE_PORT + idx))
  case_dir="${results_dir}/${case_name}"
  mkdir -p "${case_dir}"

  server_log="${case_dir}/server.log"
  bench_log="${case_dir}/benchmark.log"
  meta_file="${case_dir}/meta.env"

  {
    echo "CASE=${case_name}"
    echo "PORT=${port}"
    echo "HOST=${HOST}"
    echo "MODEL=${MODEL}"
    echo "START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${meta_file}"

  echo "[pecs-matrix] launching ${case_name} on port ${port}"
  PORT="${port}" HOST="${HOST}" PYTHON_BIN="${PYTHON_BIN}" \
    "${RUN_SCRIPT}" "${case_name}" >"${server_log}" 2>&1 &
  server_pid="$!"

  if ! wait_for_server "${HOST}" "${port}" "${SERVER_WAIT_SECONDS}"; then
    echo "[pecs-matrix] server failed to become healthy for ${case_name}" >&2
    shutdown_server "${server_pid}"
    server_pid=""
    exit 1
  fi

  if [[ -n "${BENCHMARK_CMD_TEMPLATE}" ]]; then
    benchmark_cmd="${BENCHMARK_CMD_TEMPLATE//\{port\}/${port}}"
    benchmark_cmd="${benchmark_cmd//\{host\}/${HOST}}"
    benchmark_cmd="${benchmark_cmd//\{case\}/${case_name}}"
    echo "[pecs-matrix] benchmark ${case_name}: ${benchmark_cmd}"
    bash -lc "${benchmark_cmd}" >"${bench_log}" 2>&1
  else
    echo "[pecs-matrix] BENCHMARK_CMD_TEMPLATE not set; skipping workload for ${case_name}" | tee "${bench_log}"
  fi

  shutdown_server "${server_pid}"
  server_pid=""
done

echo "[pecs-matrix] complete -> ${results_dir}"
