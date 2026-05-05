# PECS Vast Validation Notes - 2026-05-05

## Goal

Validate cheaply whether the patched PECS runtime engages during real compiled
serving on a rented Vast.ai host. The control question was:

> Does PECS produce non-zero live candidate generation and hit metrics in a
> compiled vLLM serving run?

## Host

- Instance: 2x NVIDIA A100-SXM4-80GB
- `/workspace` free space at start: about 196 GiB
- Initial GPU state: both GPUs visible, 0 MiB used, no running GPU processes
- Checkout: `/workspace/vllm-pecs`
- Branch: `pecs-prototype-work`
- Predictor path:
  `/workspace/vllm-pecs/pecs_predictors/checkpoints_real_humaneval_temporal_w256`

## Install And Test Path

Used the Python-only precompiled editable install:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

Focused unit slice after fixes:

```bash
VLLM_TARGET_DEVICE=cpu python -m pytest -q \
  tests/model_executor/layers/fused_moe/test_pecs_runtime.py \
  tests/test_config.py \
  -k 'pecs or enable_pecs'
```

Result:

```text
7 passed, 131 deselected, 18 warnings
```

## What Failed Initially

Baseline serving became healthy and returned HTTP 200.

Compiled PECS serving also became healthy and returned HTTP 200, but all PECS
engagement metrics stayed at zero:

```text
PECS confirmed hit: 0.0%, PECS proposal hit: 0.0%, PECS proposal exact: 0.0%,
PECS combined hit: 0.0%, PECS avg candidates: 0.00 logical / 0.00 physical
```

An eager PECS fallback also stayed at zero before the fixes. That ruled out a
pure CUDA graph replay issue and pointed at runtime wiring / metrics exposure.

## Debug Findings

1. PECS predictor quality was not the immediate problem.
   Predictors loaded cleanly from all 32 layer checkpoint files.

2. The original PECS Python control hooks needed compile boundaries.
   The dynamic Python-side methods must stay outside Torch compilation:
   `PecsLayerRuntime.stage_prefetch`, `PecsLayerRuntime.capture`,
   `FusedMoE.maybe_stage_pecs_prefetch`, and
   `FusedMoE._capture_pecs_logical_ids`.

3. CUDA graph replay is incompatible with the current PECS runtime path.
   The guard must force `cudagraph_mode=NONE` for PECS, not only PIECEWISE
   modes.

4. Router capture callback binding had to preserve any existing callback.
   Otherwise, later binding could overwrite a callback instead of chaining it.

5. Placing PECS staging in `FusedMoE.forward` as a separate custom op did not
   produce live stage calls. The serving logs still showed:

   ```text
   PECS stage calls: 0, PECS stage disabled/capture/empty: 0/0/0
   ```

6. The effective GPU MoE execution path already enters the
   `vllm.moe_forward` / `vllm.moe_forward_shared` Python custom-op bodies.
   PECS staging belongs there, immediately before `forward_dispatch`.

7. Mixtral did not expose PECS stats through `MixtralForCausalLM`.
   The generic `MixtureOfExperts.get_pecs_stats()` default returns `{}`, so the
   engine logger could report zeros even when layer stats were present unless
   Mixtral overrides the accessor.

8. Warmup/profile execution can intentionally disable PECS. The observed final
   run had non-zero `stage_disabled_calls`, which is acceptable as long as
   normal serving has non-zero stage calls and candidates.

## Final Fix Summary

- Added `@torch.compiler.disable` to PECS layer-owned capture/stage wrappers.
- Forced CUDA graph mode to `NONE` when PECS is enabled.
- Preserved previous router capture callbacks when binding PECS capture.
- Moved PECS staging into `_moe_forward` and `_moe_forward_shared`.
- Added `MixtralForCausalLM.get_pecs_stats()` so engine logging can aggregate
  per-layer PECS runtime stats.
- Added PECS diagnostic counters for stage calls and early exits.
- Updated the focused PECS runtime test to cover the custom-op hook placement.

## Successful Compiled Validation

Command shape:

```bash
env TMPDIR=/workspace/tmp TEMP=/workspace/tmp TMP=/workspace/tmp \
  HF_HOME=/workspace/hf_home \
  HUGGINGFACE_HUB_CACHE=/workspace/hf_home/hub \
  RESULTS_DIR=/workspace/pecs_results \
  MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 \
  TENSOR_PARALLEL_SIZE=2 \
  MAX_MODEL_LEN=512 \
  GPU_MEMORY_UTILIZATION=0.80 \
  PYTHON_BIN=/venv/main/bin/python \
  PORT=8001 \
  PECS_PREDICTOR_PATH=/workspace/vllm-pecs/pecs_predictors/checkpoints_real_humaneval_temporal_w256 \
  scripts/run_pecs_control_case.sh pecs
```

Single request:

```bash
curl -sS -m 120 http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mistralai/Mixtral-8x7B-Instruct-v0.1","prompt":"Write one sentence about Paris.","max_tokens":32,"temperature":0}'
```

Result:

- Server healthy
- Request returned HTTP 200
- Predictor load failures stayed at 0
- PECS metrics were non-zero:

```text
PECS confirmed hit: 47.0%, PECS proposal hit: 86.6%,
PECS proposal exact: 33.2%, PECS combined hit: 91.6%,
PECS avg candidates: 3.50 logical / 3.50 physical,
PECS flushes: 0, PECS stage calls: 2240,
PECS stage disabled/capture/empty: 192/0/0
```

## Conclusion

The patched PECS runtime now engages in a real compiled serving run. Candidate
generation is non-zero, proposal and combined hit metrics are non-zero, and
predictor loading remains clean. The next step can be a very small performance
comparison, but not a full benchmark matrix until this patch is reviewed and
kept stable.

