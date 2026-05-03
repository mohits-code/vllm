# PECS First Vast.ai Test

This runbook is for the first paid runtime attempt on Vast.ai.

The goal is not a full benchmark sweep. The goal is to avoid wasting cluster
time on packaging, CUDA, or model-fit mistakes before the first meaningful
PECS result.

## What Counts As Success

The first Vast session is successful if all of the following are true:

- the branch installs with the precompiled Python-only path
- `python -m vllm.entrypoints.openai.api_server --help` works
- the PECS runtime unit test passes
- a `baseline` server becomes healthy
- a `pecs` server becomes healthy
- PECS metrics appear in the PECS run logs

That is enough for the first rental. Do not expand to the full matrix until
those checks are green.

## Instance Choice

### If you want the cheapest useful first session

Use `2x A100 80GB` on one node.

Why:

- your current predictor checkpoints are Mixtral-shaped and cover 32 layers
- the existing PECS examples and artifacts are centered on
  `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `pecs_eplb` requires at least 2 visible GPUs to exercise expert-parallel
  remap behavior

### When `1x A100 80GB` is acceptable

Use `1x A100 80GB` only for:

- install/bootstrap validation
- Python-only PECS unit-test validation
- a smaller MoE smoke model that definitely fits one GPU

Do not use `1x A100` if your goal is a faithful first Mixtral runtime check.

## Recommended First Session

### 1. Upload or clone a real checkout

From local:

```bash
cd /home/ms/projects/cis8000/_vllm_pecs_worktree
scripts/create_pecs_git_bundle.sh /tmp/vllm-pecs.bundle HEAD
```

On the Vast box:

```bash
git clone /path/to/vllm-pecs.bundle /workspace/vllm-pecs
cd /workspace/vllm-pecs
```

### 2. Bootstrap with the Python-only precompiled path

```bash
cd /workspace/vllm-pecs
scripts/bootstrap_pecs_cluster_env.sh
```

If your branch is too fresh and the merge-base wheel is not ready yet, retry
with:

```bash
VLLM_PRECOMPILED_WHEEL_COMMIT=nightly scripts/bootstrap_pecs_cluster_env.sh
```

This follows the current vLLM guidance for Python-only development on top of
precompiled wheels and avoids a full local compile unless you changed C++ or
CUDA code.

## 3. Validate PECS before serving

```bash
RUN_PECS_TEST=1 INSTALL_TEST_DEPS=1 scripts/bootstrap_pecs_cluster_env.sh
```

## 4. Run the smallest useful PECS server sequence

For a first paid session, run only `baseline` then `pecs`:

```bash
export MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export PECS_PREDICTOR_PATH="/workspace/artifacts/checkpoints_real"
export PYTHON_BIN="/workspace/vllm-pecs/.venv/bin/python"
export GPU_MEMORY_UTILIZATION="0.90"

scripts/run_pecs_vast_first_test.sh /workspace/pecs_first_test
```

If the machine has 2 GPUs and the first two cases are green, add EPLB:

```bash
RUN_EP_CASES=1 \
DATA_PARALLEL_SIZE=2 \
TENSOR_PARALLEL_SIZE=1 \
scripts/run_pecs_vast_first_test.sh /workspace/pecs_first_test_ep
```

## 5. Only Then Run The Larger Matrix

After the first sequence is green, move on to:

```bash
export MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export PECS_PREDICTOR_PATH="/workspace/artifacts/checkpoints_real"
export PYTHON_BIN="/workspace/vllm-pecs/.venv/bin/python"

scripts/run_pecs_cluster_matrix.sh /workspace/pecs_matrix_runs
```

## Notes

- Prefer `uv`-based installs for vLLM commit/nightly wheels.
- Do not fall back to a full source compile unless the precompiled path fails
  and the failure is clearly not model or environment related.
- If you use Docker instead of a raw host env, the current official image is
  `vllm/vllm-openai`, and vLLM also documents source builds with
  `VLLM_USE_PRECOMPILED=1` when you only changed Python code.
