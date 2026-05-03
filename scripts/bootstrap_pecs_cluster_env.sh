#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  echo "[pecs-bootstrap] error: run this from a real git checkout, not a copied tree without .git" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[pecs-bootstrap] error: uv is required. Install it first: https://astral.sh/uv/" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_MODE="${INSTALL_MODE:-precompiled}"
RUN_PECS_TEST="${RUN_PECS_TEST:-0}"
INSTALL_TEST_DEPS="${INSTALL_TEST_DEPS:-0}"
RUN_COLLECT_ENV="${RUN_COLLECT_ENV:-1}"
VLLM_SKIP_API_HELP_CHECK="${VLLM_SKIP_API_HELP_CHECK:-0}"
VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-}"
VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-}"

echo "[pecs-bootstrap] python=${PYTHON_BIN} venv=${VENV_DIR} mode=${INSTALL_MODE}"

uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

uv pip install -U pip setuptools wheel setuptools-scm packaging

case "${INSTALL_MODE}" in
  precompiled)
    export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]; then
      export VLLM_PRECOMPILED_WHEEL_COMMIT
      echo "[pecs-bootstrap] using precompiled wheel commit ${VLLM_PRECOMPILED_WHEEL_COMMIT}"
    fi
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION}" ]]; then
      export VLLM_PRECOMPILED_WHEEL_LOCATION
      echo "[pecs-bootstrap] using custom precompiled wheel location ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
    fi
    uv pip install --editable . --torch-backend=auto
    ;;
  source_no_deps)
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
    uv pip uninstall -y torch torchvision torchaudio vllm || true
    uv pip install --index-url "${TORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}"
    export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
    uv pip install --editable . --no-build-isolation --no-deps
    ;;
  *)
    echo "[pecs-bootstrap] unsupported INSTALL_MODE=${INSTALL_MODE}" >&2
    exit 1
    ;;
esac

if [[ "${INSTALL_TEST_DEPS}" == "1" ]]; then
  uv pip install -r requirements/test/cuda.in
fi

"${VENV_DIR}/bin/python" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
if [[ "${RUN_COLLECT_ENV}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m vllm.collect_env | tee /tmp/pecs_vllm_collect_env.txt
fi

if [[ "${VLLM_SKIP_API_HELP_CHECK}" != "1" ]]; then
  "${VENV_DIR}/bin/python" -m vllm.entrypoints.openai.api_server --help >/tmp/vllm_api_server_help.txt
  tail -n 20 /tmp/vllm_api_server_help.txt
else
  echo "[pecs-bootstrap] skipping api_server --help check"
fi

if [[ "${RUN_PECS_TEST}" == "1" ]]; then
  "${VENV_DIR}/bin/python" -m pytest \
    tests/model_executor/layers/fused_moe/test_pecs_runtime.py -q
fi

echo "[pecs-bootstrap] environment ready"
