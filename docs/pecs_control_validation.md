# PECS Control-Path Validation

This branch exposes PECS control-path state in normal vLLM scheduler logs.

The current integration is for control-path validation, not TPOT claims:
- frozen per-layer predictor inference is live
- confirmed/proposed cache state is live
- EPLB-triggered PECS flushes are live
- scheduler logs now print aggregate PECS rates
- expert prefetch/staging is **not** yet wired into the kernel/transport path

## Logged PECS fields

When `--enable-pecs` is active and stats logging is enabled, vLLM logs:

- `PECS confirmed hit`
- `PECS proposal hit`
- `PECS proposal exact`
- `PECS combined hit`
- `PECS flushes`
- `PECS predictor load failures` (only if non-zero)

These are aggregate runtime values across PECS-enabled MoE layers.

## Recommended first matrix

Run one case at a time with [scripts/run_pecs_control_case.sh](../scripts/run_pecs_control_case.sh):

- `baseline`
- `pecs`
- `pecs_dbo`
- `pecs_eplb`
- `pecs_dbo_eplb`

Example:

```bash
cd /home/ms/projects/cis8000/_vllm_pecs_worktree

MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
PECS_PREDICTOR_PATH="/home/ms/projects/cis8000/artifacts/checkpoints_real_mixed_domain_temporal_w256" \
PORT=8000 \
scripts/run_pecs_control_case.sh pecs
```

With DBO:

```bash
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
PECS_PREDICTOR_PATH="/home/ms/projects/cis8000/artifacts/checkpoints_real_mixed_domain_temporal_w256" \
PORT=8001 \
scripts/run_pecs_control_case.sh pecs_dbo
```

With EPLB:

```bash
MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
PECS_PREDICTOR_PATH="/home/ms/projects/cis8000/artifacts/checkpoints_real_mixed_domain_temporal_w256" \
TENSOR_PARALLEL_SIZE=2 \
PORT=8002 \
scripts/run_pecs_control_case.sh pecs_eplb
```

## What to validate first

- PECS metrics appear only in PECS-enabled runs.
- `proposal hit` and `combined hit` are non-zero and stable.
- `proposal exact` is lower than hit rate but meaningfully above zero.
- `flushes` increases when EPLB is active and experts remap.
- predictor load failures stay at zero.

If these checks look sane, the next step is the lower-level dispatch/prefetch path for TPOT experiments.
