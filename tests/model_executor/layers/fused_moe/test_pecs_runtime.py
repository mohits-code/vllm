# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.pecs import (
    FrozenMLPPredictor,
    PecsLayerRuntime,
)


def _write_checkpoint(checkpoint_dir: Path, layer_id: int = 0) -> None:
    predictor = FrozenMLPPredictor(
        hidden_dim=4,
        num_experts=3,
        hidden_width=4,
        input_norm=False,
    )
    with torch.no_grad():
        predictor.net[0].weight.zero_()
        predictor.net[0].bias.zero_()
        predictor.net[2].weight.zero_()
        predictor.net[2].bias.copy_(torch.tensor([3.0, 2.0, -1.0]))

    torch.save(
        {
            "layer_id": layer_id,
            "model_state": predictor.state_dict(),
            "hidden_dim": 4,
            "num_experts": 3,
            "hidden_width": 4,
            "input_norm": False,
        },
        checkpoint_dir / f"mlp_layer_{layer_id:02d}.pt",
    )


def _build_runtime(checkpoint_dir: Path) -> PecsLayerRuntime:
    pecs = PecsLayerRuntime(
        enabled=True,
        layer_name="model.layers.0.block_sparse_moe",
        top_k=2,
        confirmed_capacity=2,
        predictor_path=str(checkpoint_dir),
        predictor_dtype="float32",
    )
    pecs.prepare_predictor(device=torch.device("cpu"), fallback_dtype=torch.float32)
    return pecs


def test_pecs_runtime_loads_predictor_and_tracks_hits(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)

    hidden_states = torch.randn(2, 4)
    plan = pecs.pre_route(hidden_states)
    assert plan is not None
    assert plan.proposal_ids is not None
    assert tuple(plan.proposal_ids.shape) == (2, 2)
    assert plan.proposal_experts == (0, 1)
    assert plan.combined_experts == (0, 1)
    assert plan.combined_physical_experts == (0, 1)
    pecs.mark_prefetch(plan)

    actual = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    pecs.post_route(actual)
    stats = pecs.snapshot()
    assert stats["predictor_available"] is True
    assert stats["proposal_queries"] == 2
    assert stats["proposal_hits"] == 2
    assert stats["proposal_exact_matches"] == 2
    assert stats["combined_hits"] == 2
    assert stats["prefetch_requests"] == 1
    assert stats["avg_combined_candidates"] == 2.0
    assert stats["confirmed_experts"] == [1, 0]


def test_pecs_runtime_flushes_on_eplb_remap(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)

    pecs.pre_route(torch.randn(1, 4))
    pecs.post_route(torch.tensor([[0, 1]], dtype=torch.int32))
    assert pecs.get_confirmed_experts() == (1, 0)

    initial_map = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    changed_map = torch.tensor([[1, 0, 2]], dtype=torch.int32)
    pecs.on_eplb_map_update(initial_map)
    pecs.on_eplb_map_update(changed_map)

    stats = pecs.snapshot()
    assert stats["flushes"] == 1
    assert stats["flush_reasons"] == {"eplb_rebalance": 1}
    assert pecs.get_confirmed_experts() == ()


def test_pecs_runtime_maps_combined_candidates_through_eplb(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)

    initial_map = torch.tensor([[4, 5], [1, 3], [2, 0]], dtype=torch.int32)
    pecs.on_eplb_map_update(initial_map)

    plan = pecs.pre_route(torch.randn(1, 4))
    assert plan is not None
    assert plan.proposal_experts == (0, 1)
    assert plan.combined_physical_experts == (4, 5, 1, 3)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_pecs_runtime_can_run_behind_compiled_wrapper(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)

    class _Wrapper(torch.nn.Module):
        def __init__(self, runtime: PecsLayerRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            plan = self.runtime.pre_route(hidden_states)
            assert plan is not None
            assert plan.proposal_ids is not None
            return plan.proposal_ids.to(dtype=torch.float32)

    wrapper = _Wrapper(pecs)
    compiled = torch.compile(wrapper, backend="eager")

    proposal_ids = compiled(torch.randn(2, 4))
    assert tuple(proposal_ids.shape) == (2, 2)
