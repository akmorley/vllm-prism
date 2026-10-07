# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-token-head scale views must not write to the KV cache storage.

In hybrid models the attention cache tensor aliases pages owned by mamba
groups, and the views are created lazily on the first forward, after earlier
mamba layers have written their state for the step.
"""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def test_scale_views_do_not_modify_storage():
    num_blocks, nkv, block_size, head_size = 3, 4, 8, 256
    content = 2 * (head_size + 4)  # [K | K scale | V | V scale] as int8
    # Logical (blocks, heads, slots, content) with NHD physical strides.
    storage = torch.randint(-128, 127, (num_blocks, block_size, nkv, content), dtype=torch.int8)
    kv_cache = storage.permute(0, 2, 1, 3)
    before = storage.clone()

    impl = SimpleNamespace(_k_scale_cache=None, _v_scale_cache=None)
    TritonAttentionImpl._ensure_scale_caches(impl, kv_cache)

    assert torch.equal(storage, before)
    assert impl._k_scale_cache.shape == (num_blocks, block_size, nkv)
    # The views address the scale bytes after each head's K and V data.
    impl._k_scale_cache[1, 2, 3] = 2.0
    impl._v_scale_cache[1, 2, 3] = 3.0
    k_bytes = storage[1, 2, 3, head_size : head_size + 4].view(torch.float32)
    v_bytes = storage[1, 2, 3, content - 4 :].view(torch.float32)
    assert k_bytes.item() == 2.0 and v_bytes.item() == 3.0
