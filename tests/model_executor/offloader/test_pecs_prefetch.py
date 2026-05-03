# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    _PecsLayerBinding,
    _copy_selected_expert_slices,
    _normalize_expert_ids,
)


def test_normalize_expert_ids_filters_invalid_and_duplicates():
    assert _normalize_expert_ids((2, 2, -1, 1, 7, 0), num_experts=3) == (2, 1, 0)


def test_copy_selected_expert_slices_only_updates_requested_rows():
    src = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    dst = torch.full_like(src, -1.0)

    copied = _copy_selected_expert_slices(src, dst, (2, 0))

    assert copied == 2
    assert torch.equal(
        dst,
        torch.tensor(
            [
                [1.0, 10.0],
                [-1.0, -1.0],
                [3.0, 30.0],
            ]
        ),
    )


def test_prefetch_experts_maps_global_to_local_ids_before_staging():
    staged_calls: list[tuple[str, tuple[int, ...], int]] = []

    class FakeModuleOffloader:
        def stage_expert_slices(
            self,
            *,
            param_prefix: str,
            local_expert_ids: tuple[int, ...],
            num_experts: int,
        ) -> int:
            staged_calls.append((param_prefix, local_expert_ids, num_experts))
            return 2

    layer = SimpleNamespace(
        local_num_experts=3,
        _map_global_expert_id_to_local_expert_id=lambda expert_id: {
            8: 2,
            4: 1,
            9: -1,
        }.get(expert_id, -1),
    )

    offloader = PrefetchOffloader.__new__(PrefetchOffloader)
    offloader._pecs_layer_bindings = {
        "model.layers.0.block_sparse_moe": _PecsLayerBinding(
            layer_name="model.layers.0.block_sparse_moe",
            param_prefix="block_sparse_moe",
            module_offloader=FakeModuleOffloader(),
            moe_layer=layer,
        )
    }

    PrefetchOffloader.prefetch_experts(
        offloader,
        "model.layers.0.block_sparse_moe",
        expert_ids=(0,),
        physical_expert_ids=(8, 4, 8, 9),
    )

    assert staged_calls == [("block_sparse_moe", (2, 1), 3)]
