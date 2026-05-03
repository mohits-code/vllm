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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

PecsPredictorDType = Literal["auto", "float32", "float16", "bfloat16"]


def _resolve_dtype(dtype_name: PecsPredictorDType, fallback: torch.dtype) -> torch.dtype:
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


@dataclass(frozen=True)
class PecsPrefetchPlan:
    layer_name: str
    layer_id: int | None
    num_tokens: int
    confirmed_experts: tuple[int, ...]
    proposal_experts: tuple[int, ...]
    combined_experts: tuple[int, ...]
    combined_physical_experts: tuple[int, ...]
    proposal_ids: torch.Tensor | None = None


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
    confirmed_candidate_experts: int = 0
    proposal_candidate_experts: int = 0
    combined_candidate_experts: int = 0
    combined_physical_candidate_experts: int = 0

    flush_reasons: dict[str, int] = field(default_factory=dict)

    def mark_flush(self, reason: str) -> None:
        self.flushes += 1
        self.flush_reasons[reason] = self.flush_reasons.get(reason, 0) + 1

    def mark_prefetch(self, plan: PecsPrefetchPlan) -> None:
        self.prefetch_requests += 1
        self.confirmed_candidate_experts += len(plan.confirmed_experts)
        self.proposal_candidate_experts += len(plan.proposal_experts)
        self.combined_candidate_experts += len(plan.combined_experts)
        self.combined_physical_candidate_experts += len(plan.combined_physical_experts)

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
        self._predictor_checkpoint: dict[str, object] | None = None
        self._pending_proposals: torch.Tensor | None = None
        self._pending_confirmed_snapshot: tuple[int, ...] = ()
        self._logical_to_physical_map: torch.Tensor | None = None

        self._prepare_predictor_checkpoint()

    @property
    def predictor_available(self) -> bool:
        return self._predictor is not None

    def flush(self, reason: str) -> None:
        self._confirmed_cache.clear()
        self._pending_proposals = None
        self._pending_confirmed_snapshot = ()
        self.stats.mark_flush(reason)

    def on_eplb_map_update(self, logical_to_physical_map: torch.Tensor | None) -> None:
        if not self.enabled or logical_to_physical_map is None:
            return
        self._logical_to_physical_map = logical_to_physical_map.detach().to(
            dtype=torch.int32, device="cpu"
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
        self._predictor_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    def _maybe_load_predictor(self, hidden_states: torch.Tensor) -> None:
        if (
            not self.enabled
            or self._predictor_loaded
            or self.predictor_path is None
            or self.layer_id is None
        ):
            return

        self._predictor_loaded = True
        checkpoint = self._predictor_checkpoint
        if checkpoint is None:
            return

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
        predictor_dtype = _resolve_dtype(self.predictor_dtype, hidden_states.dtype)
        predictor = predictor.to(device=hidden_states.device, dtype=predictor_dtype)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad_(False)

        self._predictor = predictor
        self.stats.predictor_enabled = True
        logger.info(
            "Loaded PECS predictor for %s from %s",
            self.layer_name,
            self._predictor_checkpoint_path,
        )

    @staticmethod
    def _rank_proposal_experts(proposals: torch.Tensor | None) -> tuple[int, ...]:
        if proposals is None or proposals.numel() == 0:
            return ()

        flat_ids = proposals.reshape(-1).to(device="cpu", dtype=torch.int64)
        unique_ids, counts = torch.unique(flat_ids, return_counts=True, sorted=True)
        ranked = sorted(
            zip(unique_ids.tolist(), counts.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        return tuple(int(expert_id) for expert_id, _ in ranked)

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
            if logical_expert < 0 or logical_expert >= self._logical_to_physical_map.shape[0]:
                continue
            mapped = self._logical_to_physical_map[logical_expert].reshape(-1).tolist()
            for physical_expert in mapped:
                expert_id = int(physical_expert)
                if expert_id < 0 or expert_id in seen:
                    continue
                physical_ordered.append(expert_id)
                seen.add(expert_id)
        return tuple(physical_ordered)

    def pre_route(self, hidden_states: torch.Tensor) -> PecsPrefetchPlan | None:
        if not self.enabled:
            return None

        self._maybe_load_predictor(hidden_states)
        self._pending_confirmed_snapshot = tuple(self._confirmed_cache)

        if self._predictor is None:
            self._pending_proposals = None
            proposal_experts: tuple[int, ...] = ()
            combined_experts = self._pending_confirmed_snapshot
            return PecsPrefetchPlan(
                layer_name=self.layer_name,
                layer_id=self.layer_id,
                num_tokens=int(hidden_states.shape[0]),
                confirmed_experts=self._pending_confirmed_snapshot,
                proposal_experts=proposal_experts,
                combined_experts=combined_experts,
                combined_physical_experts=self._map_logical_candidates_to_physical(
                    combined_experts
                ),
                proposal_ids=None,
            )

        with torch.inference_mode():
            predictor_param = next(self._predictor.parameters())
            inputs = hidden_states.to(
                device=predictor_param.device, dtype=predictor_param.dtype
            )
            logits = self._predictor(inputs)
            self._pending_proposals = torch.topk(
                logits, k=min(self.top_k, logits.shape[-1]), dim=-1
            ).indices.to(device=hidden_states.device, dtype=torch.int32)
        proposal_experts = self._rank_proposal_experts(self._pending_proposals)
        combined_experts = self._merge_candidates(
            self._pending_confirmed_snapshot, proposal_experts
        )
        return PecsPrefetchPlan(
            layer_name=self.layer_name,
            layer_id=self.layer_id,
            num_tokens=int(hidden_states.shape[0]),
            confirmed_experts=self._pending_confirmed_snapshot,
            proposal_experts=proposal_experts,
            combined_experts=combined_experts,
            combined_physical_experts=self._map_logical_candidates_to_physical(
                combined_experts
            ),
            proposal_ids=self._pending_proposals,
        )

    @staticmethod
    def _has_overlap(actual: torch.Tensor, predicted: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        if isinstance(predicted, tuple):
            if not predicted:
                return torch.zeros(actual.shape[0], device=actual.device, dtype=torch.bool)
            predicted_tensor = torch.tensor(
                predicted, device=actual.device, dtype=actual.dtype
            ).unsqueeze(0)
        else:
            predicted_tensor = predicted.to(device=actual.device, dtype=actual.dtype)
        return (actual.unsqueeze(-1) == predicted_tensor.unsqueeze(-2)).any(dim=(-1, -2))

    def post_route(self, logical_ids: torch.Tensor) -> None:
        if not self.enabled:
            return

        actual = logical_ids.to(dtype=torch.int32)
        num_tokens = int(actual.shape[0])

        confirmed_overlap = self._has_overlap(actual, self._pending_confirmed_snapshot)
        self.stats.confirmed_queries += num_tokens
        self.stats.confirmed_hits += int(confirmed_overlap.sum().item())

        if self._pending_proposals is not None:
            proposal_overlap = self._has_overlap(actual, self._pending_proposals)
            proposal_exact = (
                torch.sort(actual, dim=-1).values
                == torch.sort(self._pending_proposals.to(actual.device), dim=-1).values
            ).all(dim=-1)
            self.stats.proposal_queries += num_tokens
            self.stats.proposal_hits += int(proposal_overlap.sum().item())
            self.stats.proposal_exact_matches += int(proposal_exact.sum().item())

            combined_hits = confirmed_overlap | proposal_overlap
            self.stats.combined_queries += num_tokens
            self.stats.combined_hits += int(combined_hits.sum().item())

        for expert_id in actual.reshape(-1).tolist():
            expert = int(expert_id)
            if expert < 0:
                continue
            try:
                self._confirmed_cache.remove(expert)
            except ValueError:
                pass
            self._confirmed_cache.appendleft(expert)

        self._pending_proposals = None
        self._pending_confirmed_snapshot = tuple(self._confirmed_cache)

    def get_confirmed_experts(self) -> tuple[int, ...]:
        return tuple(self._confirmed_cache)

    def mark_prefetch(self, plan: PecsPrefetchPlan) -> None:
        if self.enabled:
            self.stats.mark_prefetch(plan)

    def snapshot(self) -> dict[str, object]:
        snapshot = self.stats.as_dict()
        snapshot.update(
            {
                "enabled": self.enabled,
                "layer_name": self.layer_name,
                "layer_id": self.layer_id,
                "confirmed_capacity": self.confirmed_capacity,
                "confirmed_experts": list(self._confirmed_cache),
                "predictor_available": self.predictor_available,
                "predictor_path": self.predictor_path,
            }
        )
        return snapshot
