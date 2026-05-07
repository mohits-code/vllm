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
from vllm.model_executor.offloader.base import get_offloader

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
    ) -> None:
        self.enabled = enabled
        self.layer_name = layer_name
        self.layer_id = _parse_layer_id(layer_name)
        self.top_k = top_k
        self.confirmed_capacity = confirmed_capacity
        self.predictor_path = predictor_path
        self.predictor_dtype = predictor_dtype

        self.stats = PecsLayerStats()
        self._confirmed_cache: deque[int] = deque(maxlen=max(confirmed_capacity, 1))
        self._last_map_signature: tuple[int, ...] | None = None
        self._predictor: FrozenMLPPredictor | None = None
        self._predictor_loaded = False
        self._predictor_checkpoint_path: str | None = None
        self._pending_proposals: torch.Tensor | None = None
        self._pending_confirmed_snapshot: tuple[int, ...] = ()
        self._pending_confirmed_tensor: torch.Tensor | None = None
        self._logical_to_physical_map: torch.Tensor | None = None
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
        self._logical_to_physical_map = (
            logical_to_physical_map.detach()
            .to(dtype=torch.int32, device="cpu")
            .pin_memory()
        )
        self._gpu_logical_to_physical_map = self._logical_to_physical_map.to(
            device="cuda"
        )
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
    def _rank_proposal_experts(proposals: torch.Tensor | None) -> tuple[int, ...]:
        if proposals is None or proposals.numel() == 0:
            return ()

        flat_ids = proposals.reshape(-1).cpu().to(dtype=torch.int64)
        unique_ids, counts = torch.unique(flat_ids, return_counts=True, sorted=True)
        ranked = sorted(
            zip(unique_ids.tolist(), counts.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(int(expert_id) for expert_id, _ in ranked)

    @staticmethod
    def _rank_proposal_experts_tensor(
        proposals: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if proposals is None:
            return torch.empty(0, device=device, dtype=torch.int32)

        flat_ids = proposals.reshape(-1).to(device=device, dtype=torch.int64)
        if flat_ids.numel() == 0:
            return torch.empty(0, device=device, dtype=torch.int32)

        unique_ids, counts = torch.unique(flat_ids, return_counts=True, sorted=True)
        sort_keys = unique_ids.to(dtype=torch.int64) - counts.to(dtype=torch.int64) * (
            unique_ids.numel() + 1
        )
        ranked_indices = torch.argsort(sort_keys, stable=True)
        return unique_ids[ranked_indices].to(dtype=torch.int32)

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
        if self._logical_to_physical_map is None:
            return logical_ids.to(dtype=torch.int32)

        mapping = self._logical_to_physical_map.to(device=logical_ids.device)
        valid = (logical_ids >= 0) & (logical_ids < mapping.shape[0])
        logical_ids = logical_ids[valid]
        if logical_ids.numel() == 0:
            return torch.empty(0, device=mapping.device, dtype=torch.int32)

        physical_ids = mapping[logical_ids].reshape(-1).to(dtype=torch.int64)
        physical_ids = physical_ids[physical_ids >= 0]
        if physical_ids.numel() == 0:
            return torch.empty(0, device=mapping.device, dtype=torch.int32)
        return _unique_preserve_order(physical_ids).to(dtype=torch.int32)

    @staticmethod
    def _confirmed_experts_tensor(
        confirmed_experts: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not confirmed_experts:
            res = torch.empty(0, device="cpu", dtype=torch.int32)
        else:
            res = torch.tensor(confirmed_experts, device="cpu", dtype=torch.int32)
        if device.type == "cuda":
            res = res.pin_memory()
        return res.to(device=device)

    def _get_confirmed_cache_tensor(self, *, device: torch.device) -> torch.Tensor:
        if self._confirmed_cache_tensor is None:
            return torch.full(
                (self.confirmed_capacity,), -1, device=device, dtype=torch.int32
            )
        return self._confirmed_cache_tensor.to(device=device, dtype=torch.int32)

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
        if self._confirmed_cache_tensor.device.type == "cpu":
            self._confirmed_cache_tensor = self._confirmed_cache_tensor.pin_memory()

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
        ).to(dtype=torch.int64)
        return _unique_preserve_order(merged).to(dtype=torch.int32)

    @torch.compiler.disable
    def stage_prefetch(self, hidden_states: torch.Tensor) -> None:
        if not self.enabled:
            return
        self.stats.stage_calls += 1
        if not _PECS_RUNTIME_ENABLED.get():
            self.stats.stage_disabled_calls += 1
            self._pending_proposals = None
            self._pending_confirmed_snapshot = tuple(self._confirmed_cache)
            return

        self._maybe_load_predictor(hidden_states)
        self._pending_confirmed_snapshot = tuple(self._confirmed_cache)
        confirmed_experts = self._pending_confirmed_snapshot

        if self._predictor is None:
            self._pending_proposals = None
            proposal_experts: tuple[int, ...] = ()
        else:
            with torch.inference_mode():
                predictor_param = next(self._predictor.parameters())
                inputs = hidden_states.to(
                    device=predictor_param.device, dtype=predictor_param.dtype
                )
                logits = self._predictor(inputs)
                self._pending_proposals = torch.topk(
                    logits, k=min(self.top_k, logits.shape[-1]), dim=-1
                ).indices.to(device=hidden_states.device, dtype=torch.int32)
            
            proposals_cpu = self._pending_proposals.cpu()
            proposal_experts = self._rank_proposal_experts(proposals_cpu)

        combined_experts = self._merge_candidates(confirmed_experts, proposal_experts)
        combined_physical_experts = self._map_logical_candidates_to_physical(combined_experts)

        if hidden_states.device.type == "cuda":
            if not combined_experts:
                combined_expert_tensor = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            else:
                combined_expert_tensor = torch.tensor(combined_experts, dtype=torch.int32, device=hidden_states.device)
                
            if not combined_physical_experts:
                combined_physical_expert_tensor = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            else:
                combined_physical_expert_tensor = torch.tensor(combined_physical_experts, dtype=torch.int32, device=hidden_states.device)

            torch.ops.vllm.pecs_prefetch_experts(
                hidden_states,
                combined_expert_tensor,
                combined_physical_expert_tensor,
                self.layer_name,
                int(hidden_states.shape[0]),
            )
        else:
            combined_expert_tensor = torch.tensor(combined_experts, dtype=torch.int32, device=hidden_states.device)
            combined_physical_expert_tensor = torch.tensor(combined_physical_experts, dtype=torch.int32, device=hidden_states.device)
            get_offloader().prefetch_experts(
                self.layer_name,
                combined_expert_tensor,
                physical_expert_ids=combined_physical_expert_tensor,
                num_tokens=int(hidden_states.shape[0]),
            )

        self.stats.mark_prefetch(
            num_confirmed_experts=len(confirmed_experts),
            num_proposal_experts=len(proposal_experts),
            num_combined_experts=len(combined_experts),
            num_combined_physical_experts=len(combined_physical_experts),
        )

    @torch.compiler.disable
    def capture(self, logical_ids: torch.Tensor) -> None:
        if not self.enabled:
            return
        if not _PECS_RUNTIME_ENABLED.get():
            self._pending_proposals = None
            self._pending_confirmed_snapshot = tuple(self._confirmed_cache)
            return

        self.stats.stage_capture_calls += 1

        # One blocking GPU->CPU copy per capture call (unavoidable to update the
        # confirmed cache deque). We call .tolist() on a small int32 tensor of
        # unique expert IDs (at most 8 values for Mixtral), not on hidden states.
        unique_experts = torch.unique(logical_ids[logical_ids >= 0])
        actual_list: list[int] = unique_experts.tolist()
        actual_set = set(actual_list)

        # ---- update confirmed cache deque (LRU-style: most recent at front) ----
        for expert in actual_list:
            try:
                self._confirmed_cache.remove(expert)
            except ValueError:
                pass
            self._confirmed_cache.appendleft(expert)

        # ---- hit-rate accounting (CPU-only, uses snapshots from stage_prefetch) ----
        confirmed_snapshot = set(self._pending_confirmed_snapshot or ())

        self.stats.confirmed_queries += 1
        if actual_set & confirmed_snapshot:
            self.stats.confirmed_hits += 1

        if self._pending_proposals is not None:
            # _pending_proposals is already on CPU (pulled in stage_prefetch)
            proposal_ids_cpu = self._pending_proposals
            if proposal_ids_cpu.device.type != "cpu":
                proposal_ids_cpu = proposal_ids_cpu.cpu()
            proposal_set = set(proposal_ids_cpu.reshape(-1).tolist())
            proposal_set.discard(-1)

            self.stats.proposal_queries += 1
            if actual_set & proposal_set:
                self.stats.proposal_hits += 1
            if actual_set == proposal_set and len(actual_set) > 0:
                self.stats.proposal_exact_matches += 1

            combined_set = confirmed_snapshot | proposal_set
            self.stats.combined_queries += 1
            if actual_set & combined_set:
                self.stats.combined_hits += 1

        self._pending_proposals = None
        self._pending_confirmed_snapshot = tuple(self._confirmed_cache)

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
