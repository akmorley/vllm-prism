# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash context K/V for quantized drafters: per-layer qkv_proj path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.platforms import current_platform


class _QKV(nn.Module):
    """Stand-in for QKVParallelLinear: returns (output, bias) like vLLM layers."""

    def __init__(self, weight, bias, quantized):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        # Any non-unquantized method selects the per-layer path.
        self.quant_method = object() if quantized else UnquantizedLinearMethod()

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias), None


def _model(quantized, has_bias, layers=3, hidden=64, nkv=2, head_dim=16, nq=4):
    torch.manual_seed(0)
    q_size, kv = nq * head_dim, nkv * head_dim
    attn = []
    for _ in range(layers):
        weight = torch.randn(q_size + 2 * kv, hidden, device="cuda") / hidden**0.5
        bias = torch.randn(q_size + 2 * kv, device="cuda") if has_bias else None
        attn.append(
            SimpleNamespace(
                qkv_proj=_QKV(weight, bias, quantized),
                q_size=q_size,
                k_norm=SimpleNamespace(
                    weight=nn.Parameter(torch.rand(head_dim, device="cuda") + 0.5)
                ),
            )
        )
    model = SimpleNamespace(
        hidden_norm=SimpleNamespace(
            weight=nn.Parameter(torch.rand(hidden, device="cuda") + 0.5)
        ),
        _rms_norm_eps=1e-6,
    )
    DFlashQwen3Model._build_context_kv_buffers(model, attn, has_bias)
    return model, (layers, nkv, head_dim, hidden)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="This test requires CUDA")
@pytest.mark.parametrize("has_bias", [False, True])
def test_quantized_drafter_context_kv_matches_fused(has_bias):
    fused, (layers, nkv, head_dim, hidden) = _model(False, has_bias)
    per_layer, _ = _model(True, has_bias)
    assert fused._context_qkv_layers is None and fused._fused_kv_weight is not None
    assert per_layer._context_qkv_layers is not None
    assert per_layer._fused_kv_weight is None
    torch.testing.assert_close(per_layer._k_norm_weights, fused._k_norm_weights)

    states = torch.randn(7, hidden, device="cuda")
    expected = DFlashQwen3Model._project_context_kv(
        fused, states, 7, layers, nkv, head_dim
    )
    actual = DFlashQwen3Model._project_context_kv(
        per_layer, states, 7, layers, nkv, head_dim
    )
    for a, e in zip(actual, expected):
        assert a.shape == (layers, 7, nkv, head_dim) and a.is_contiguous()
        torch.testing.assert_close(a, e)
