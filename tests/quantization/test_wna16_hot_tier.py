# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.model_executor.layers.fused_moe.experts import hot_tier


def test_static_hot_tier_loads_rankings(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rankings.json"
    path.write_text(json.dumps({"0": [2, 0, 1], "1": [1, 2, 0]}))
    monkeypatch.setenv("VLLM_WNA16_HOT_TIER_SIZE", "2")
    monkeypatch.setenv("VLLM_WNA16_HOT_TIER_FILE", str(path))
    hot_tier._RANKINGS.clear()

    assert hot_tier.tier_size() == 2
    assert hot_tier._load_rankings() == {0: [2, 0, 1], 1: [1, 2, 0]}

    hot_tier._RANKINGS.clear()


def test_static_hot_tier_extracts_layer_index() -> None:
    assert hot_tier._layer_index("model.layers.47.mlp.experts") == 47
    assert hot_tier._layer_index("model.mlp.experts") is None
