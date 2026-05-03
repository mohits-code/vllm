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

## Lessons From The First Real Vast Sessions

The first paid PECS-on-vLLM attempts produced several environment-level failure
patterns that should be treated as hard acceptance gates for future rentals.

### Instance Acceptance Gate

Before cloning anything, run:

```bash
df -h
nvidia-smi
```

Reject the instance immediately if any of the following are true:

- `/workspace` is a tiny overlay or nearly full
- the GPUs show large idle memory use with no visible processes
- fewer than 2 GPUs are visible

Practical thresholds used during debugging:

- require at least about `150G` free writable disk under `/workspace`
- reject a box if idle `nvidia-smi` shows tens of GB already consumed per GPU

### What Failed And Why

1. Some `vLLM` template containers exposed only a `32G` writable overlay, which
   broke:
   - Hugging Face model downloads
   - temporary-file creation
   - log/result writing

2. Some otherwise promising `2x A100 80GB` boxes exposed effectively full GPU
   memory at worker startup. In one case `nvidia-smi` later showed about
   `75.99 GiB / 80 GiB` used on each GPU with no visible process table entry.
   `fuser -v /dev/nvidia*` later revealed leftover `vllm`, `EngineCore`, and
   `Worker_TP` processes holding the device files. Future retries on the same
   box must kill stale holders before concluding the host is broken.

3. `2x A100 40GB` was too fragile for this first Mixtral validation. Even when
   the host was clean, startup and KV-cache budgeting left too little headroom
   for a low-friction first runtime proof.

### Working Host Class

The first cleanly usable target for this exact runtime validation was:

- `2x A100 SXM4 80GB`
- at least about `200G` free writable disk on `/workspace`
- clean idle GPUs

### Working Python Install Path

On a host that already has the heavy vLLM dependencies installed, the fastest
working path was:

```bash
cd /workspace
git clone --branch pecs-prototype-work https://github.com/mohits-code/vllm.git vllm-pecs
cd /workspace/vllm-pecs
export VLLM_USE_PRECOMPILED=1
pip install -e . --no-deps
```

### Working Runtime Environment

```bash
mkdir -p /workspace/hf_home /workspace/tmp /workspace/pecs_results
export TMPDIR=/workspace/tmp
export TEMP=/workspace/tmp
export TMP=/workspace/tmp
export HF_HOME=/workspace/hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/hf_home/hub
export RESULTS_DIR=/workspace/pecs_results
export MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export PECS_PREDICTOR_PATH="/workspace/vllm-pecs/pecs_predictors/checkpoints_real_humaneval_temporal_w256"
export PYTHON_BIN="$(which python)"
```

### Baseline Configuration That Reached A Healthy Server

The `baseline` case succeeded on the clean `2x A100 80GB` host with:

```bash
BASE_PORT=8000 \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=512 \
GPU_MEMORY_UTILIZATION=0.80 \
RUN_EP_CASES=0 \
scripts/run_pecs_vast_first_test.sh
```

Important nuance:

- lower `gpu_memory_utilization` values avoided some startup checks on dirty
  hosts but later failed because no KV cache blocks could be allocated
- on the clean host, increasing `gpu_memory_utilization` and shrinking
  `max_model_len` was the correct direction

### PECS Result

The `pecs` case exposed a real integration bug under compiled execution.

Observed failure path:

- `pre_route()`
- `_maybe_load_predictor()`
- `checkpoint_path.exists()`

This path executes inside the forward path and is picked up by
`torch.compile` / Dynamo. That filesystem existence check is not safe inside
the compiled path.

The same `pecs` configuration succeeded when eager mode disabled
`torch.compile` and CUDA graphs:

```bash
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
PECS_PREDICTOR_PATH="/workspace/vllm-pecs/pecs_predictors/checkpoints_real_humaneval_temporal_w256" \
PYTHON_BIN="$(which python)" \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=512 \
GPU_MEMORY_UTILIZATION=0.80 \
EXTRA_ARGS="--enforce-eager" \
scripts/run_pecs_control_case.sh pecs
```

The eager `pecs` server:

- loaded all 32 layer predictors
- initialized KV cache successfully
- started the OpenAI-compatible API server
- served a real `/v1/chat/completions` request successfully

### Immediate Code Follow-Up

The next PECS fix should move predictor path resolution and lazy file-system
checks out of the compiled forward path.

Concretely:

- do not call `Path.exists()` from `pre_route()`
- pre-resolve checkpoint availability earlier in model/runtime initialization
- keep compiled forward execution free of Python file I/O

## Final Outcome From The First Session

The first real runtime validation ended with three concrete results:

- `baseline` worked on a clean `2x A100 80GB` host
- `pecs` worked end to end in eager mode and served a real request
- `pecs` did not run cleanly under full `torch.compile` tracing because
  `pre_route()` still contains Python-heavy control logic

The initialization issues were narrowed and fixed in stages:

- predictor checkpoint discovery moved out of forward
- predictor module construction moved out of forward
- predictor device/dtype preparation moved out of forward
- PECS predictor preparation switched to a generic model-module traversal

After those fixes, the remaining compiled-mode failure was inside the PECS
proposal logic itself rather than lazy setup. Specifically, the proposal path
still performs Python-side ranking and list conversion such as:

- `.tolist()`
- `zip(...)`
- Python tuple/list/set merging

### Practical Short-Term Fix

The pragmatic fix for the current branch is to treat the PECS control hooks as
non-Dynamo code:

- `PecsLayerRuntime.pre_route`
- `PecsLayerRuntime.post_route`

These hooks can be marked with `@torch.compiler.disable`, which preserves the
compiled model path while keeping PECS setup, proposal ranking, and statistics
updates outside the traced graph.

This does **not** prove that PECS is compile-native. It proves that:

- the integration works
- the predictor artifacts are valid
- the PECS control logic can run correctly alongside a compiled wrapper when
  explicitly excluded from tracing

### Local Validation Strategy

Do not rerun the full vLLM server locally just to validate this control-path
change.

Use a CPU-only fake harness instead:

- set `VLLM_TARGET_DEVICE=cpu`
- import `PecsLayerRuntime` directly from the worktree
- build a tiny frozen checkpoint
- call `prepare_predictor(...)`
- run a small `torch.compile(..., backend=\"eager\")` wrapper around
  `pre_route()` / `post_route()`

That is enough to validate the PECS control-path contract without needing the
CUDA extension or a full model load.
