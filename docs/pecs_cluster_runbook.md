# PECS Cluster Runbook

This runbook exists to avoid repeating the setup failures that wasted paid
cluster time.

If you are specifically using Vast.ai for the first PECS runtime attempt, start
with [docs/pecs_vast_first_test.md](pecs_vast_first_test.md) before using the
full matrix below.

## Non-Negotiable Rules

- Do not copy the source tree to a cluster without `.git`.
- Do not run a plain `pip install -e .` after Torch is pinned.
- Prefer the precompiled / Python-only install path unless a PECS change
  actually touches C++ or CUDA code.
- Verify `python -m vllm.entrypoints.openai.api_server --help` before any
  serving run.

## Recommended Remote Bring-Up

### 1. Package a real checkout locally

From the PECS vLLM worktree:

```bash
cd /home/ms/projects/cis8000/_vllm_pecs_worktree
scripts/create_pecs_git_bundle.sh /tmp/vllm-pecs.bundle HEAD
```

Copy the bundle to the rented machine, then clone from it there:

```bash
git clone /path/to/vllm-pecs.bundle /workspace/vllm-pecs
cd /workspace/vllm-pecs
```

### 2. Bootstrap the environment

Preferred path:

```bash
cd /workspace/vllm-pecs
scripts/bootstrap_pecs_cluster_env.sh
```

This uses:

- `uv`
- a local `.venv`
- `VLLM_USE_PRECOMPILED=1`
- `uv pip install --editable . --torch-backend=auto`

Fallback path if the precompiled install is not viable:

```bash
INSTALL_MODE=source_no_deps scripts/bootstrap_pecs_cluster_env.sh
```

That path pins Torch first, then installs vLLM with `--no-deps`.

### 3. Validate the PECS runtime unit test

```bash
RUN_PECS_TEST=1 INSTALL_TEST_DEPS=1 scripts/bootstrap_pecs_cluster_env.sh
```

This is slower, but it validates the PECS runtime layer before serving.

For the smallest useful paid-runtime sequence on Vast, use:

```bash
scripts/run_pecs_vast_first_test.sh /workspace/pecs_first_test
```

## Serving Matrix

The PECS branch currently exposes a control-path hook plus observability:

- frozen per-layer predictor proposals
- confirmed expert cache
- EPLB-triggered flushes
- PECS candidate-set logging
- a prefetch-offloader path that stages PECS-selected expert slices before MoE
  execution

The entrypoint for case launches is:

```bash
scripts/run_pecs_control_case.sh
```

Supported cases:

- `baseline`
- `pecs`
- `pecs_dbo`
- `pecs_eplb`
- `pecs_dbo_eplb`

## Full Workload Matrix

Use the matrix runner to start each case, wait for health, run a benchmark
command, and collect logs in separate directories.

Example:

```bash
cd /workspace/vllm-pecs

export MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"
export PECS_PREDICTOR_PATH="/workspace/artifacts/checkpoints_real_mixed_domain_temporal_w256"
export PYTHON_BIN="/workspace/vllm-pecs/.venv/bin/python"
export BENCHMARK_CMD_TEMPLATE='
  .venv/bin/python -m vllm.entrypoints.cli.main bench serve \
    --backend openai-chat \
    --endpoint http://{host}:{port}/v1 \
    --dataset-name sonnet \
    --num-prompts 128
'

scripts/run_pecs_cluster_matrix.sh /workspace/pecs_matrix_runs
```

Replace the benchmark command with the real workload you want to test. The
template supports:

- `{host}`
- `{port}`
- `{case}`

## What to Inspect in Logs

With PECS enabled, the server logs should now report:

- `PECS confirmed hit`
- `PECS proposal hit`
- `PECS proposal exact`
- `PECS combined hit`
- `PECS avg candidates`
- `PECS flushes`

Sanity checks:

- PECS metrics appear only in PECS-enabled cases.
- `proposal hit` and `combined hit` are non-zero.
- `avg candidates` stays bounded and does not explode under EPLB.
- `flushes` rises when EPLB is enabled and remaps experts.
- predictor load failures remain zero.
