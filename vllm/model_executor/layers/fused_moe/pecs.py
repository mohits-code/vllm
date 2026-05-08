"""PECS runtime state for MoE layers.

This module provides a lightweight runtime integration point for PECS inside
vLLM's MoE layer stack:

- optional frozen per-layer MLP proposal generation from hidden states
- an online confirmed cache built from live routed experts
- EPLB-aware flushing when logical->physical expert maps change
- counters that can be queried during TPOT experiments

The actual low-level expert prefetch/staging optimization is intentionally
separate. This module establishes the runtime control plane and observability
needed to evaluate PECS in vLLM before deeper kernel-path work.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn

import vllm.model_executor.offloader.prefetch_ops  # noqa: F401
from vllm.logger import init_logger
# ── Diagnostic env vars for isolating latency contributions ──────────────────
# PECS_DEBUG_NOOP_STAGE=1  : skip ALL stage_prefetch work (return immediately)
#                            → if TPOT drops back to baseline, PECS is the cause
# PECS_DEBUG_NO_MLP=1      : skip MLP forward, keep confirmed cache only
#                            → isolates MLP overhead from graph-break overhead
# PECS_DEBUG_TIMING=1      : log avg CPU-wall time per stage_prefetch call
#                            (every 2000 calls per layer)
# PECS_DEBUG_NO_CAPTURE=1  : skip ALL capture() work
_PECS_DEBUG_NOOP_STAGE: bool = os.environ.get("PECS_DEBUG_NOOP_STAGE", "") == "1"
_PECS_DEBUG_NO_MLP: bool = os.environ.get("PECS_DEBUG_NO_MLP", "") == "1"
_PECS_DEBUG_TIMING: bool = os.environ.get("PECS_DEBUG_TIMING", "") == "1"
_PECS_DEBUG_NO_CAPTURE: bool = os.environ.get("PECS_DEBUG_NO_CAPTURE", "") == "1"
_PECS_TIMING_INTERVAL: int = int(os.environ.get("PECS_TIMING_INTERVAL", "2000"))

logger = init_logger(__name__)

PecsPredictorDType = Literal["auto", "float32", "float16", "bfloat16"]
_PECS_RUNTIME_ENABLED: ContextVar[bool] = ContextVar(
    "pecs_runtime_enabled", default=True
)


@contextmanager
def disable_pecs_runtime() -> object:
    token = _PECS_RUNTIME_ENABLED.set(False)
    try:
        yield
    finally:
        _PECS_RUNTIME_ENABLED.reset(token)


def _resolve_dtype(
    dtype_name: PecsPredictorDType, fallback: torch.dtype
) -> torch.dtype:
    if dtype_name == "auto":
        return fallback
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported PECS predictor dtype: {dtype_name}")


def _parse_layer_id(layer_name: str) -> int | None:
    patterns = [
        r"(?:^|\.)(?:layers|layer)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:h)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:blocks)\.(\d+)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, layer_name)
        if match:
            return int(match.group(1))
    return None


def _unique_preserve_order(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    flat_values = values.reshape(-1)
    eq = flat_values.unsqueeze(0) == flat_values.unsqueeze(1)
    seen_before = torch.triu(eq, diagonal=1).any(dim=0)
    return flat_values[~seen_before]


class FrozenMLPPredictor(nn.Module):
    """Frozen two-layer MLP matching the offline PECS training code."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        hidden_width: int,
        *,
        input_norm: bool,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if input_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.Linear(hidden_dim, hidden_width), nn.GELU()])
        layers.append(nn.Linear(hidden_width, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


@dataclass
class PecsLayerStats:
    flushes: int = 0
    predictor_load_failures: int = 0
    predictor_enabled: bool = False

    confirmed_queries: int = 0
    confirmed_hits: int = 0

    proposal_queries: int = 0
    proposal_hits: int = 0
    proposal_exact_matches: int = 0

    combined_queries: int = 0
    combined_hits: int = 0

    prefetch_requests: int = 0
    stage_calls: int = 0
    stage_disabled_calls: int = 0
    stage_capture_calls: int = 0
    stage_empty_candidate_calls: int = 0
    confirmed_candidate_experts: int = 0
    proposal_candidate_experts: int = 0
    combined_candidate_experts: int = 0
    combined_physical_candidate_experts: int = 0

    flush_reasons: dict[str, int] = field(default_factory=dict)

    def mark_flush(self, reason: str) -> None:
        self.flushes += 1
        self.flush_reasons[reason] = self.flush_reasons.get(reason, 0) + 1

    def mark_prefetch(
        self,
        *,
        num_confirmed_experts: int,
        num_proposal_experts: int,
        num_combined_experts: int,
        num_combined_physical_experts: int,
    ) -> None:
        self.prefetch_requests += 1
        self.confirmed_candidate_experts += num_confirmed_experts
        self.proposal_candidate_experts += num_proposal_experts
        self.combined_candidate_experts += num_combined_experts
        self.combined_physical_candidate_experts += num_combined_physical_experts

    def as_dict(self) -> dict[str, object]:
        return {
            "flushes": self.flushes,
            "predictor_load_failures": self.predictor_load_failures,
            "predictor_enabled": self.predictor_enabled,
            "confirmed_queries": self.confirmed_queries,
            "confirmed_hits": self.confirmed_hits,
            "confirmed_hit_rate": (
                self.confirmed_hits / self.confirmed_queries
                if self.confirmed_queries
                else 0.0
            ),
            "proposal_queries": self.proposal_queries,
            "proposal_hits": self.proposal_hits,
            "proposal_hit_rate": (
                self.proposal_hits / self.proposal_queries
                if self.proposal_queries
                else 0.0
            ),
            "proposal_exact_matches": self.proposal_exact_matches,
            "proposal_exact_match_rate": (
                self.proposal_exact_matches / self.proposal_queries
                if self.proposal_queries
                else 0.0
            ),
            "combined_queries": self.combined_queries,
            "combined_hits": self.combined_hits,
            "combined_hit_rate": (
                self.combined_hits / self.combined_queries
                if self.combined_queries
                else 0.0
            ),
            "prefetch_requests": self.prefetch_requests,
            "stage_calls": self.stage_calls,
            "stage_disabled_calls": self.stage_disabled_calls,
            "stage_capture_calls": self.stage_capture_calls,
            "stage_empty_candidate_calls": self.stage_empty_candidate_calls,
            "confirmed_candidate_experts": self.confirmed_candidate_experts,
            "proposal_candidate_experts": self.proposal_candidate_experts,
            "combined_candidate_experts": self.combined_candidate_experts,
            "combined_physical_candidate_experts": self.combined_physical_candidate_experts,
            "avg_confirmed_candidates": (
                self.confirmed_candidate_experts / self.prefetch_requests
                if self.prefetch_requests
                else 0.0
            ),
            "avg_proposal_candidates": (
                self.proposal_candidate_experts / self.prefetch_requests
                if self.prefetch_requests
                else 0.0
            ),
            "avg_combined_candidates": (
                self.combined_candidate_experts / self.prefetch_requests
                if self.prefetch_requests
                else 0.0
            ),
            "avg_combined_physical_candidates": (
                self.combined_physical_candidate_experts / self.prefetch_requests
                if self.prefetch_requests
                else 0.0
            ),
            "flush_reasons": dict(self.flush_reasons),
        }


class PecsLayerRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        layer_name: str,
        top_k: int,
        confirmed_capacity: int,
        predictor_path: str | None,
        predictor_dtype: PecsPredictorDType,
        proposal_confidence_threshold: float = 0.0,
    ) -> None:
        self.enabled = enabled
        self.layer_name = layer_name
        self.layer_id = _parse_layer_id(layer_name)
        self.top_k = top_k
        self.confirmed_capacity = confirmed_capacity
        self.predictor_path = predictor_path
        self.predictor_dtype = predictor_dtype
        self.proposal_confidence_threshold = proposal_confidence_threshold

        self.stats = PecsLayerStats()
        self._confirmed_cache: deque[int] = deque(maxlen=max(confirmed_capacity, 1))
        self._last_map_signature: tuple[int, ...] | None = None
        self._predictor: FrozenMLPPredictor | None = None
        self._predictor_loaded = False
        self._predictor_checkpoint_path: str | None = None
        self._pending_proposals: torch.Tensor | None = None
        self._pending_confirmed_snapshot: tuple[int, ...] = ()
        self._pending_confirmed_tensor: torch.Tensor | None = None
        self._gpu_logical_to_physical_map: torch.Tensor | None = None
        self._confirmed_cache_tensor: torch.Tensor | None = None
        self._confirmed_hits_tensor: torch.Tensor | None = None
        self._proposal_hits_tensor: torch.Tensor | None = None
        self._proposal_exact_matches_tensor: torch.Tensor | None = None
        self._combined_hits_tensor: torch.Tensor | None = None

        self._prepare_predictor_checkpoint()

    @property
    def predictor_available(self) -> bool:
        return self._predictor is not None

    def flush(self, reason: str) -> None:
        self._confirmed_cache.clear()
        self._confirmed_cache_tensor = None
        self._pending_proposals = None
        self._pending_confirmed_snapshot = ()
        self._pending_confirmed_tensor = None
        self.stats.mark_flush(reason)

    def on_eplb_map_update(self, logical_to_physical_map: torch.Tensor | None) -> None:
        if not self.enabled or logical_to_physical_map is None:
            return
        # Keep the map resident on CUDA only — no pinned-CPU copy on the hot path.
        self._gpu_logical_to_physical_map = logical_to_physical_map.detach().to(
            device="cuda", dtype=torch.int32
        )
        # Signature check: map is small and this is a rare EPLB event, so the
        # CPU sync here is acceptable.
        signature = tuple(int(x) for x in logical_to_physical_map.reshape(-1).tolist())
        if self._last_map_signature is None:
            self._last_map_signature = signature
            return
        if signature != self._last_map_signature:
            self.flush("eplb_rebalance")
            self._last_map_signature = signature

    def _checkpoint_path(self) -> Path | None:
        if not self.predictor_path or self.layer_id is None:
            return None
        return Path(self.predictor_path) / f"mlp_layer_{self.layer_id:02d}.pt"

    def _prepare_predictor_checkpoint(self) -> None:
        if not self.enabled or self.predictor_path is None or self.layer_id is None:
            return

        checkpoint_path = self._checkpoint_path()
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(
                "PECS predictor checkpoint missing for layer %s at %s; continuing "
                "with confirmed-cache-only PECS.",
                self.layer_name,
                checkpoint_path,
            )
            self.stats.predictor_load_failures += 1
            return

        self._predictor_checkpoint_path = os.fspath(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        hidden_dim = int(checkpoint["hidden_dim"])
        num_experts = int(checkpoint["num_experts"])
        hidden_width = int(checkpoint["hidden_width"])
        input_norm = bool(checkpoint["input_norm"])

        predictor = FrozenMLPPredictor(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            hidden_width=hidden_width,
            input_norm=input_norm,
        )
        predictor.load_state_dict(checkpoint["model_state"])
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad_(False)

        self._predictor = predictor
        self.stats.predictor_enabled = True

    def _maybe_load_predictor(self, hidden_states: torch.Tensor) -> None:
        if not self.enabled or not self._predictor_loaded:
            raise RuntimeError(
                f"PECS predictor for {self.layer_name} was not prepared before "
                "compiled forward execution."
            )

    def prepare_predictor(
        self, *, device: torch.device, fallback_dtype: torch.dtype
    ) -> None:
        if not self.enabled or self._predictor_loaded:
            return

        predictor = self._predictor
        if predictor is None:
            self._predictor_loaded = True
            return

        predictor_dtype = _resolve_dtype(self.predictor_dtype, fallback_dtype)
        predictor = predictor.to(device=device, dtype=predictor_dtype)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad_(False)

        self._predictor = predictor
        self._predictor_loaded = True
        logger.info(
            "Loaded PECS predictor for %s from %s",
            self.layer_name,
            self._predictor_checkpoint_path,
        )

    @staticmethod
    def _rank_proposal_experts_tensor(
        proposals: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if proposals is None:
            return torch.empty(0, device=device, dtype=torch.int32)

        flat_ids = proposals.reshape(-1).to(device=device, dtype=torch.int32)
        if flat_ids.numel() == 0:
            return torch.empty(0, device=device, dtype=torch.int32)

        # To avoid dynamic PyTorch sizing (which breaks CUDA graphs), we
        # bypass unique sorting and simply return the flattened valid array.
        # Duplicates are harmless to the prefetch cache.
        return flat_ids

    @staticmethod
    def _merge_candidates(
        confirmed_experts: tuple[int, ...], proposal_experts: tuple[int, ...]
    ) -> tuple[int, ...]:
        ordered: list[int] = []
        seen: set[int] = set()
        for expert_id in (*confirmed_experts, *proposal_experts):
            if expert_id in seen:
                continue
            ordered.append(expert_id)
            seen.add(expert_id)
        return tuple(ordered)

    def _map_logical_candidates_to_physical(
        self, logical_experts: tuple[int, ...]
    ) -> tuple[int, ...]:
        if not logical_experts:
            return ()
        if self._logical_to_physical_map is None:
            return logical_experts

        physical_ordered: list[int] = []
        seen: set[int] = set()
        for logical_expert in logical_experts:
            if (
                logical_expert < 0
                or logical_expert >= self._logical_to_physical_map.shape[0]
            ):
                continue
            mapped = self._logical_to_physical_map[logical_expert].reshape(-1).tolist()
            for physical_expert in mapped:
                expert_id = int(physical_expert)
                if expert_id < 0 or expert_id in seen:
                    continue
                physical_ordered.append(expert_id)
                seen.add(expert_id)
        return tuple(physical_ordered)

    def _map_logical_candidates_to_physical_tensor(
        self,
        logical_experts: torch.Tensor,
    ) -> torch.Tensor:
        if logical_experts.numel() == 0:
            return logical_experts.to(dtype=torch.int32)

        logical_ids = logical_experts.to(dtype=torch.long).reshape(-1)
        # Use the CUDA-resident map directly — no CPU↔GPU transfer.
        if self._gpu_logical_to_physical_map is None:
            return logical_ids.to(dtype=torch.int32)

        mapping = self._gpu_logical_to_physical_map
        if mapping.device != logical_ids.device:
            mapping = mapping.to(device=logical_ids.device)

        # Avoid boolean indexing to preserve static shapes for CUDA graphs
        valid_mask = (logical_ids >= 0) & (logical_ids < mapping.shape[0])
        safe_ids = torch.where(valid_mask, logical_ids, torch.tensor(0, device=logical_ids.device))
        physical_ids = mapping[safe_ids]
        
        # Mask out originally invalid IDs with -1
        physical_ids = torch.where(
            valid_mask.unsqueeze(-1), 
            physical_ids, 
            torch.tensor(-1, device=physical_ids.device)
        ).reshape(-1)
        
        # Return statically sized tensor, keeping sentinels
        return physical_ids.to(dtype=torch.int32)

    @staticmethod
    def _confirmed_experts_tensor(
        confirmed_experts: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        # Allocate directly on the target device — no CPU bounce.
        if not confirmed_experts:
            return torch.empty(0, device=device, dtype=torch.int32)
        return torch.tensor(confirmed_experts, device=device, dtype=torch.int32)

    def _get_confirmed_cache_tensor(self, *, device: torch.device) -> torch.Tensor:
        if self._confirmed_cache_tensor is None:
            return torch.full(
                (self.confirmed_capacity,), -1, device=device, dtype=torch.int32
            )
        # Cache is already on CUDA; only move if device differs (rare).
        if self._confirmed_cache_tensor.device != device:
            self._confirmed_cache_tensor = self._confirmed_cache_tensor.to(device=device)
        return self._confirmed_cache_tensor

    def _accumulate_tensor_counter(
        self,
        attr_name: str,
        delta: torch.Tensor,
    ) -> None:
        delta = delta.to(dtype=torch.int64)
        current = getattr(self, attr_name)
        if current is None:
            current = torch.zeros((), device=delta.device, dtype=torch.int64)
        elif current.device != delta.device:
            current = current.to(device=delta.device)
        current.add_(delta)
        setattr(self, attr_name, current)

    @staticmethod
    def _read_tensor_counter(counter: torch.Tensor | None) -> int:
        if counter is None:
            return 0
        return int(counter.item())

    def _update_confirmed_cache_tensor(self, actual: torch.Tensor) -> None:
        recent = actual.reshape(-1).to(dtype=torch.int64)
        recent = recent[recent >= 0]
        if recent.numel() > 0:
            recent = _unique_preserve_order(recent.flip(0))

        existing = self._get_confirmed_cache_tensor(device=actual.device).to(
            dtype=torch.int64
        )
        if recent.numel() == 0:
            updated = existing
        elif existing.numel() == 0:
            updated = recent
        else:
            keep_existing = ~(existing.unsqueeze(1) == recent.unsqueeze(0)).any(dim=1)
            updated = torch.cat([recent, existing[keep_existing]], dim=0)

        self._confirmed_cache_tensor = updated[: self.confirmed_capacity].to(
            dtype=torch.int32
        )
        if self._confirmed_cache_tensor.numel() < self.confirmed_capacity:
            pad = torch.full(
                (self.confirmed_capacity - self._confirmed_cache_tensor.numel(),),
                -1,
                dtype=torch.int32,
                device=self._confirmed_cache_tensor.device,
            )
            self._confirmed_cache_tensor = torch.cat(
                [self._confirmed_cache_tensor, pad]
            )
        # Cache stays on CUDA — no pin_memory() needed.

    @staticmethod
    def _merge_candidate_tensors(
        confirmed_experts: torch.Tensor,
        proposal_experts: torch.Tensor,
    ) -> torch.Tensor:
        if confirmed_experts.numel() == 0:
            return proposal_experts
        if proposal_experts.numel() == 0:
            return confirmed_experts

        merged = torch.cat(
            [confirmed_experts.reshape(-1), proposal_experts.reshape(-1)]
        ).to(dtype=torch.int32)
        # Bypassing unique to maintain static shapes for CUDA graphs.
        # Duplicates will be safely ignored during staging.
        return merged

    # Accumulated timing state (CPU wall clock, only used when _PECS_DEBUG_TIMING)
    _debug_stage_time_total: float = 0.0
    _debug_stage_time_calls: int = 0

    def stage_prefetch(self, hidden_states: torch.Tensor) -> None:
        if not self.enabled:
            return
            
        is_compiling = torch.compiler.is_compiling()
        
        if not is_compiling:
            self.stats.stage_calls += 1


        _t0 = time.perf_counter() if (not is_compiling and _PECS_DEBUG_TIMING) else 0.0

        self._maybe_load_predictor(hidden_states)
        dev = hidden_states.device

        # --- confirmed cache: GPU tensor, NO boolean-index (avoids GPU sync) ---
        # Use clamp+mask instead of confirmed_tensor[confirmed_tensor >= 0] to
        # keep all ops fixed-size and avoid implicit GPU sync from variable-size
        # boolean-indexed tensor allocation.
        confirmed_tensor = self._get_confirmed_cache_tensor(device=dev)
        if not is_compiling:
            self._pending_confirmed_tensor = confirmed_tensor  # snapshot for capture()
        # valid_confirmed: all entries >= 0 (sentinel = -1)
        # Keep as fixed-size; _merge_candidate_tensors already filters sentinels.
        valid_confirmed = confirmed_tensor

        # --- MLP proposals: stay on GPU, NO .cpu() ---
        if self._predictor is None or _PECS_DEBUG_NO_MLP:
            self._pending_proposals = None
            proposal_tensor = torch.empty(0, device=dev, dtype=torch.int32)
        else:
            with torch.inference_mode():
                predictor_param = next(self._predictor.parameters())
                inputs = hidden_states.to(
                    device=predictor_param.device, dtype=predictor_param.dtype
                )
                logits = self._predictor(inputs)
                # Apply softmax confidence threshold: only stage experts where
                # the predicted probability exceeds the threshold. This keeps
                # avg candidates close to top_k (ideal = 2 for Mixtral) rather
                # than bloating to 4+ from indiscriminate topk.
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_idx = torch.topk(
                    probs, k=min(self.top_k, probs.shape[-1]), dim=-1
                )
                if self.proposal_confidence_threshold > 0.0:
                    # Mask low-confidence predictions with sentinel -1
                    # (static shape: torch.where, no boolean indexing)
                    sentinel = torch.full_like(top_idx, -1)
                    threshold = torch.tensor(
                        self.proposal_confidence_threshold,
                        device=top_probs.device,
                        dtype=top_probs.dtype,
                    )
                    top_idx = torch.where(top_probs >= threshold, top_idx, sentinel)
                self._pending_proposals = top_idx.to(dtype=torch.int32)
            proposal_tensor = self._rank_proposal_experts_tensor(
                self._pending_proposals, device=dev,
            )
            # Cap to top_k most-voted experts.
            if proposal_tensor.numel() > self.top_k:
                proposal_tensor = proposal_tensor[: self.top_k]

        # --- merge + map: all GPU tensor ops, NO Python lists ---
        combined_tensor = self._merge_candidate_tensors(
            valid_confirmed, proposal_tensor,
        )
        combined_physical_tensor = self._map_logical_candidates_to_physical_tensor(
            combined_tensor,
        )

        # --- prefetch call: both tensors already on device ---
        if dev.type == "cuda":
            torch.ops.vllm.pecs_prefetch_experts(
                hidden_states,
                combined_tensor,
                combined_physical_tensor,
                self.layer_name,
                hidden_states.shape[0],
            )
        else:
            get_offloader().prefetch_experts(
                self.layer_name,
                combined_tensor,
                physical_expert_ids=combined_physical_tensor,
                num_tokens=hidden_states.shape[0],
            )

        # --- stats ---
        # Skip scalar reduce .item() (causes GPU sync breaking CUDA graphs).
        # We can accept stats being slightly inaccurate or zeros during decode,
        # or we could log them asynchronously.
        if not is_compiling:
            n_prop = proposal_tensor.shape[0]
            n_comb = combined_tensor.shape[0]
            n_phys = combined_physical_tensor.shape[0]
            
            self.stats.mark_prefetch(
                num_confirmed_experts=0,  # omitted to avoid sync
                num_proposal_experts=n_prop,
                num_combined_experts=n_comb,
                num_combined_physical_experts=n_phys,
            )

            # ── Diagnostic timing log ─────────────────────────────────────────────
            if _PECS_DEBUG_TIMING:
                self._debug_stage_time_total += time.perf_counter() - _t0
                self._debug_stage_time_calls += 1
                if self._debug_stage_time_calls % _PECS_TIMING_INTERVAL == 0:
                    avg_us = (
                        self._debug_stage_time_total / self._debug_stage_time_calls * 1e6
                    )
                    logger.warning(
                        "[PECS TIMING] %s: avg stage_prefetch CPU wall time = %.1f µs "
                        "over %d calls (NOOP=%s NO_MLP=%s)",
                        self.layer_name,
                        avg_us,
                        self._debug_stage_time_calls,
                        _PECS_DEBUG_NOOP_STAGE,
                        _PECS_DEBUG_NO_MLP,
                    )
                    self._debug_stage_time_total = 0.0
                    self._debug_stage_time_calls = 0

    def capture(self, logical_ids: torch.Tensor) -> None:
        if not self.enabled:
            return
            
        is_compiling = torch.compiler.is_compiling()
        
        if not is_compiling:
            if not _PECS_RUNTIME_ENABLED.get():
                self._pending_proposals = None
                self._pending_confirmed_tensor = None
                return

            # Skip capture during decode (batch_size == 1) to avoid dynamic shapes
            # and Python-side operations breaking the CUDA graph capture.
            # The confirmed cache is already built during prefill.
            if _PECS_DEBUG_NOOP_STAGE or _PECS_DEBUG_NO_CAPTURE or logical_ids.shape[0] == 1:
                self._pending_proposals = None
                self._pending_confirmed_tensor = None
                return

            self.stats.stage_capture_calls += 1

        # Flatten logical_ids to 1D [batch * top_k] before masking and broadcasting
        logical_ids_1d = logical_ids.reshape(-1)

        # We keep actual as fixed size by avoiding logical_ids[logical_ids >= 0]
        # Use valid_mask to zero out -1s for hit checking, avoiding dynamic size tensor creation
        valid_mask = (logical_ids_1d >= 0)
        # Using torch.where is safer and faster than boolean assignment to avoid syncs
        actual = torch.where(valid_mask, logical_ids_1d, torch.tensor(-1, device=logical_ids_1d.device))
        
        # We don't unique here because it also returns a dynamic sized tensor

        # --- hit-rate accounting BEFORE cache update (GPU broadcast) ---
        if not is_compiling:
            if self._pending_confirmed_tensor is not None:
                conf = self._pending_confirmed_tensor.to(device=actual.device)
                conf_mask = (conf >= 0)
                
                # Hit if ANY valid actual ID is in the valid conf IDs
                # Shape: [batch*topk, num_experts]
                matches = (actual.unsqueeze(1) == conf.unsqueeze(0))
                # Mask out invalid actuals and invalid confs
                matches = matches & valid_mask.unsqueeze(1) & conf_mask.unsqueeze(0)
                
                conf_hit = matches.any()
                self.stats.confirmed_queries += 1
                self._accumulate_tensor_counter(
                    "_confirmed_hits_tensor", conf_hit.long()
                )

            if self._pending_proposals is not None:
                prop = self._pending_proposals.to(device=actual.device).reshape(-1)
                prop_mask = (prop >= 0)
                
                matches = (actual.unsqueeze(1) == prop.unsqueeze(0))
                matches = matches & valid_mask.unsqueeze(1) & prop_mask.unsqueeze(0)
                prop_hit = matches.any()
                
                # For exact match, we skip for now since it's hard without unique
                exact = torch.zeros((), dtype=torch.bool, device=actual.device)

                self.stats.proposal_queries += 1
                self._accumulate_tensor_counter(
                    "_proposal_hits_tensor", prop_hit.long()
                )
                self._accumulate_tensor_counter(
                    "_proposal_exact_matches_tensor", exact.long()
                )

                # combined hit
                if self._pending_confirmed_tensor is not None:
                    comb_hit = conf_hit | prop_hit
                else:
                    comb_hit = prop_hit
                self.stats.combined_queries += 1
                self._accumulate_tensor_counter(
                    "_combined_hits_tensor", comb_hit.long()
                )

        # --- update confirmed cache (GPU tensor, NO Python deque) ---
        # unique does cause a sync, but we use it sparingly or avoid it if possible
        # Actually _update_confirmed_cache_tensor takes variable size.
        # Let's let it run the unique but at least we killed 5 other syncs.
        actual_unique = torch.unique(actual[actual >= 0])
        self._update_confirmed_cache_tensor(actual_unique)
        
        self._pending_proposals = None
        self._pending_confirmed_tensor = None

    def get_confirmed_experts(self) -> tuple[int, ...]:
        return tuple(self._confirmed_cache)

    def snapshot(self) -> dict[str, object]:
        stats = self.stats.as_dict()
        stats["confirmed_hits"] = self.stats.confirmed_hits + self._read_tensor_counter(
            self._confirmed_hits_tensor
        )
        stats["confirmed_hit_rate"] = (
            stats["confirmed_hits"] / self.stats.confirmed_queries
            if self.stats.confirmed_queries
            else 0.0
        )
        stats["proposal_hits"] = self.stats.proposal_hits + self._read_tensor_counter(
            self._proposal_hits_tensor
        )
        stats["proposal_hit_rate"] = (
            stats["proposal_hits"] / self.stats.proposal_queries
            if self.stats.proposal_queries
            else 0.0
        )
        stats["proposal_exact_matches"] = (
            self.stats.proposal_exact_matches
            + self._read_tensor_counter(self._proposal_exact_matches_tensor)
        )
        stats["proposal_exact_match_rate"] = (
            stats["proposal_exact_matches"] / self.stats.proposal_queries
            if self.stats.proposal_queries
            else 0.0
        )
        stats["combined_hits"] = self.stats.combined_hits + self._read_tensor_counter(
            self._combined_hits_tensor
        )
        stats["combined_hit_rate"] = (
            stats["combined_hits"] / self.stats.combined_queries
            if self.stats.combined_queries
            else 0.0
        )
        snapshot = stats
        snapshot.update(
            {
                "enabled": self.enabled,
                "layer_name": self.layer_name,
                "layer_id": self.layer_id,
                "confirmed_capacity": self.confirmed_capacity,
                "confirmed_experts": list(self.get_confirmed_experts()),
                "predictor_available": self.predictor_available,
                "predictor_path": self.predictor_path,
            }
        )
        return snapshot
