#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: scripts/run_pecs_control_case.sh CASE

CASE:
  baseline
  pecs
  pecs_dbo
  pecs_eplb
  pecs_dbo_eplb

Required env:
  MODEL                  Hugging Face model id or local path

Optional env:
  PYTHON_BIN             Python executable (default: python)
  HOST                   Server host (default: 127.0.0.1)
  PORT                   Server port (default: 8000)
  TENSOR_PARALLEL_SIZE   --tensor-parallel-size
  DATA_PARALLEL_SIZE     --data-parallel-size
  PIPELINE_PARALLEL_SIZE --pipeline-parallel-size
  GPU_MEMORY_UTILIZATION --gpu-memory-utilization
  MAX_MODEL_LEN          --max-model-len
  MAX_NUM_SEQS           --max-num-seqs
  MAX_NUM_BATCHED_TOKENS --max-num-batched-tokens
  PECS_PREDICTOR_PATH    Required for PECS cases
  PECS_CONFIRMED_CAPACITY --pecs-confirmed-capacity (default: 2)
  PECS_PREDICTOR_DTYPE   --pecs-predictor-dtype (default: auto)
  DBO_DECODE_THRESHOLD   --dbo-decode-token-threshold
  DBO_PREFILL_THRESHOLD  --dbo-prefill-token-threshold
  EPLB_CONFIG_JSON       --eplb-config JSON string
  API_SERVER_MODULE      Python module for serving
                          (default: vllm.entrypoints.openai.api_server)
  EXTRA_ARGS             Extra args appended to the vLLM server command
  DRY_RUN                If 1, print the command and exit
EOF
  exit 1
fi

CASE="$1"
: "${MODEL:?set MODEL to the model id or local path}"

PYTHON_BIN="${PYTHON_BIN:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PECS_CONFIRMED_CAPACITY="${PECS_CONFIRMED_CAPACITY:-2}"
PECS_PREDICTOR_DTYPE="${PECS_PREDICTOR_DTYPE:-auto}"
API_SERVER_MODULE="${API_SERVER_MODULE:-vllm.entrypoints.openai.api_server}"

cmd=(
  "$PYTHON_BIN" -m "$API_SERVER_MODULE"
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
)

optional_flag() {
  local env_name="$1"
  local flag_name="$2"
  local value="${!env_name:-}"
  if [[ -n "$value" ]]; then
    cmd+=("$flag_name" "$value")
  fi
}

optional_flag TENSOR_PARALLEL_SIZE --tensor-parallel-size
optional_flag DATA_PARALLEL_SIZE --data-parallel-size
optional_flag PIPELINE_PARALLEL_SIZE --pipeline-parallel-size
optional_flag GPU_MEMORY_UTILIZATION --gpu-memory-utilization
optional_flag MAX_MODEL_LEN --max-model-len
optional_flag MAX_NUM_SEQS --max-num-seqs
optional_flag MAX_NUM_BATCHED_TOKENS --max-num-batched-tokens

case "$CASE" in
  baseline)
    ;;
  pecs|pecs_dbo|pecs_eplb|pecs_dbo_eplb)
    : "${PECS_PREDICTOR_PATH:?set PECS_PREDICTOR_PATH for PECS cases}"
    cmd+=(
      --enable-pecs
      --pecs-predictor-path "$PECS_PREDICTOR_PATH"
      --pecs-confirmed-capacity "$PECS_CONFIRMED_CAPACITY"
      --pecs-predictor-dtype "$PECS_PREDICTOR_DTYPE"
    )
    ;;
  *)
    echo "Unknown CASE: $CASE" >&2
    exit 1
    ;;
esac

if [[ "$CASE" == *"dbo"* ]]; then
  cmd+=(--enable-dbo)
  optional_flag DBO_DECODE_THRESHOLD --dbo-decode-token-threshold
  optional_flag DBO_PREFILL_THRESHOLD --dbo-prefill-token-threshold
fi

if [[ "$CASE" == *"eplb"* ]]; then
  cmd+=(--enable-expert-parallel --enable-eplb)
  optional_flag EPLB_CONFIG_JSON --eplb-config
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_ARGS} )
  cmd+=("${extra_args[@]}")
fi

printf 'Launching control-path case %s\n' "$CASE"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
