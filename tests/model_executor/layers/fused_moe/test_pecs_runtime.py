# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
import pytest
from torch.library import opcheck

import vllm.model_executor.layers.fused_moe.runner.moe_runner_base as moe_runner_base
from vllm.model_executor.layers.fused_moe.pecs import (
    FrozenMLPPredictor,
    PecsLayerRuntime,
    disable_pecs_runtime,
)
from vllm.model_executor.offloader.base import BaseOffloader, get_offloader, set_offloader


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


class _RecordingOffloader(BaseOffloader):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def wrap_modules(self, modules_generator):
        return list(modules_generator)

    def prefetch_experts(
        self,
        layer_name: str,
        expert_ids: tuple[int, ...] | torch.Tensor,
        *,
        physical_expert_ids: tuple[int, ...] | torch.Tensor | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if isinstance(expert_ids, torch.Tensor):
            expert_ids = tuple(int(x) for x in expert_ids.reshape(-1).tolist())
        if isinstance(physical_expert_ids, torch.Tensor):
            physical_expert_ids = tuple(
                int(x) for x in physical_expert_ids.reshape(-1).tolist()
            )
        self.calls.append(
            {
                "layer_name": layer_name,
                "expert_ids": expert_ids,
                "physical_expert_ids": physical_expert_ids,
                "num_tokens": num_tokens,
            }
        )


def test_pecs_runtime_stages_prefetch_and_tracks_hits(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)
    recording_offloader = _RecordingOffloader()
    original_offloader = get_offloader()
    set_offloader(recording_offloader)
    try:
        hidden_states = torch.randn(2, 4)
        pecs.stage_prefetch(hidden_states)

        assert recording_offloader.calls == [
            {
                "layer_name": "model.layers.0.block_sparse_moe",
                "expert_ids": (0, 1),
                "physical_expert_ids": (0, 1),
                "num_tokens": 2,
            }
        ]

        actual = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        pecs.capture(actual)
    finally:
        set_offloader(original_offloader)

    stats = pecs.snapshot()
    assert stats["predictor_available"] is True
    assert stats["proposal_queries"] == 2
    assert stats["proposal_hits"] == 2
    assert stats["proposal_exact_matches"] == 2
    assert stats["combined_hits"] == 2
    assert stats["prefetch_requests"] == 1
    assert stats["avg_combined_candidates"] == 2.0
    assert stats["confirmed_experts"] == [1, 0]


def test_pecs_runtime_disabled_context_skips_prefetch_and_capture(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)
    recording_offloader = _RecordingOffloader()
    original_offloader = get_offloader()
    set_offloader(recording_offloader)
    try:
        hidden_states = torch.randn(2, 4)
        with disable_pecs_runtime():
            pecs.stage_prefetch(hidden_states)
            pecs.capture(torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))
    finally:
        set_offloader(original_offloader)

    assert recording_offloader.calls == []
    stats = pecs.snapshot()
    assert stats["proposal_queries"] == 0
    assert stats["confirmed_queries"] == 0
    assert stats["prefetch_requests"] == 0
    assert stats["confirmed_experts"] == []


def test_pecs_runtime_flushes_on_eplb_remap(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)

    pecs = _build_runtime(tmp_path)

    pecs.stage_prefetch(torch.randn(1, 4))
    pecs.capture(torch.tensor([[0, 1]], dtype=torch.int32))
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
    recording_offloader = _RecordingOffloader()
    original_offloader = get_offloader()
    set_offloader(recording_offloader)
    try:
        initial_map = torch.tensor([[4, 5], [1, 3], [2, 0]], dtype=torch.int32)
        pecs.on_eplb_map_update(initial_map)

        pecs.stage_prefetch(torch.randn(1, 4))
    finally:
        set_offloader(original_offloader)

    assert recording_offloader.calls == [
        {
            "layer_name": "model.layers.0.block_sparse_moe",
            "expert_ids": (0, 1),
            "physical_expert_ids": (4, 5, 1, 3),
            "num_tokens": 1,
        }
    ]


def test_moe_forward_custom_op_invokes_layer_owned_pecs_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DummyRunner:
        def __init__(self) -> None:
            self.calls = 0

        def forward_dispatch(
            self,
            layer: object,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            del layer, router_logits, shared_experts_input
            self.calls += 1
            return hidden_states

    class _DummyLayer:
        def __init__(self) -> None:
            self.runner = _DummyRunner()
            self.prefetch_inputs: list[torch.Tensor] = []

        def maybe_stage_pecs_prefetch(self, hidden_states: torch.Tensor) -> None:
            self.prefetch_inputs.append(hidden_states.clone())

    layer = _DummyLayer()
    monkeypatch.setattr(
        moe_runner_base,
        "get_layer_from_name",
        lambda layer_name: layer,
    )
    hidden_states = torch.randn(3, 4)
    router_logits = torch.randn(3, 8)

    output = moe_runner_base._moe_forward(
        hidden_states,
        router_logits,
        None,
        "model.layers.0.block_sparse_moe",
    )

    assert torch.equal(output, hidden_states)
    assert layer.runner.calls == 1
    assert len(layer.prefetch_inputs) == 1
    assert torch.equal(layer.prefetch_inputs[0], hidden_states)


def test_pecs_prefetch_custom_op_opcheck() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_states = torch.randn(2, 4, device=device)
    logical_expert_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
    physical_expert_ids = torch.tensor([0, 1], device=device, dtype=torch.int32)

    opcheck(
        torch.ops.vllm.pecs_prefetch_experts,
        (
            hidden_states,
            logical_expert_ids,
            physical_expert_ids,
            "model.layers.0.block_sparse_moe",
            2,
        ),
    )
