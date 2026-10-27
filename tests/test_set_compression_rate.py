"""Tests for the in-training compression-rate setter and the underlying
_chunked_attention_pool's shape behaviour under cr changes, plus the
equivalence between the attention path (with zero pool_queries) and the legacy
mean-pool path. Also covers K>1 multi-query (Perceiver-style) pool.

We intentionally avoid building a full CVLM (would need an HF download); the
setter is tested as an unbound method on a tiny dummy object, and the pool
function is tested directly against synthetic tensors. CPU is fine.
"""
import pytest
import torch


def test_chunked_attention_pool_v_scales_with_cr():
    """V_max scales as ceil(L_real / cr) when the source mask is all ones."""
    from modeling import _chunked_attention_pool

    B, L, H = 2, 64, 8
    hidden = torch.randn(B, L, H)
    mask = torch.ones(B, L, dtype=torch.long)

    pooled1, vmask1 = _chunked_attention_pool(hidden, mask, compression_rate=1, max_vision_len=64)
    assert pooled1.shape == (B, 64, H)
    assert vmask1.sum().item() == B * 64

    pooled4, vmask4 = _chunked_attention_pool(hidden, mask, compression_rate=4, max_vision_len=64)
    assert pooled4.shape == (B, 16, H)
    assert vmask4.sum().item() == B * 16

    pooled8, vmask8 = _chunked_attention_pool(hidden, mask, compression_rate=8, max_vision_len=64)
    assert pooled8.shape == (B, 8, H)
    assert vmask8.sum().item() == B * 8


def test_zero_pool_queries_matches_mean_pool():
    """At pool_queries=0 the attention path must reproduce mean-pool exactly,
    including for the partial-last-chunk case (uniform softmax over real chunk
    tokens equals the mean of those tokens). Covers both 1D [H] and 2D [1,H]
    inputs (the kwarg auto-promotes 1D)."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(0)
    B, L, H = 3, 19, 16   # L=19 with cr=4 -> last chunk has 3 real + 1 pad
    hidden = torch.randn(B, L, H, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.long)

    pooled_mean, vmask_mean = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=None,
    )
    for q in (
        torch.zeros(H, dtype=torch.float64),       # legacy 1D
        torch.zeros(1, H, dtype=torch.float64),    # K=1 explicit
    ):
        pooled_attn, vmask_attn = _chunked_attention_pool(
            hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=q,
        )
        assert torch.equal(vmask_mean, vmask_attn)
        real = vmask_mean.bool().unsqueeze(-1)
        diff = (pooled_mean - pooled_attn).where(real, torch.zeros_like(pooled_mean))
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-12), \
            f"shape={tuple(q.shape)} max abs diff = {diff.abs().max().item()}"


def test_nonzero_pool_queries_diverges_from_mean_pool():
    """Sanity: with a non-zero pool_queries the attention output is generically
    different from the mean — guards against accidentally short-circuiting to
    mean-pool when pool_queries is provided."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(1)
    B, L, H = 2, 16, 8
    hidden = torch.randn(B, L, H, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.long)
    q = torch.randn(H, dtype=torch.float64)

    pooled_mean, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=None,
    )
    pooled_attn, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=q,
    )
    assert not torch.allclose(pooled_mean, pooled_attn)


def test_zero_multi_pool_queries_matches_mean_pool():
    """Zero-init K=4 latents must still reduce to mean-pool: each of the K
    softmaxes is uniform, so all K outputs equal the mean of the chunk, and the
    mean-of-K is the mean too. Strict drop-in for any K."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(2)
    B, L, H = 2, 17, 12  # L=17, cr=4 -> last chunk has 1 real + 3 pad
    hidden = torch.randn(B, L, H, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.long)

    pooled_mean, vmask_mean = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=None,
    )
    q = torch.zeros(4, H, dtype=torch.float64)
    pooled_multi, vmask_multi = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=q,
    )
    assert torch.equal(vmask_mean, vmask_multi)
    real = vmask_mean.bool().unsqueeze(-1)
    diff = (pooled_mean - pooled_multi).where(real, torch.zeros_like(pooled_mean))
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-12), \
        f"max abs diff = {diff.abs().max().item()}"


def test_nonzero_multi_pool_queries_diverges_from_mean_pool():
    """K>1 with random non-zero queries should not collapse to mean-pool."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(3)
    B, L, H = 2, 16, 8
    hidden = torch.randn(B, L, H, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.long)
    q = torch.randn(4, H, dtype=torch.float64)

    pooled_mean, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=None,
    )
    pooled_multi, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=q,
    )
    assert not torch.allclose(pooled_mean, pooled_multi)


def test_multi_pool_queries_output_shape_one_per_chunk():
    """K>1 must still emit exactly one vision token per chunk (K outputs are
    mean-reduced). Output geometry must match K=1 / mean-pool."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(4)
    B, L, H = 2, 32, 8
    hidden = torch.randn(B, L, H, dtype=torch.float64)
    mask = torch.ones(B, L, dtype=torch.long)

    for K in (1, 2, 4, 8):
        q = torch.randn(K, H, dtype=torch.float64)
        pooled, vmask = _chunked_attention_pool(
            hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=q,
        )
        # 32 / cr=4 = 8 chunks => V=8 vision tokens per sample.
        assert pooled.shape == (B, 8, H), f"K={K} got {pooled.shape}"
        assert vmask.sum().item() == B * 8


def test_set_compression_rate_validates_inputs():
    from modeling import CVLM

    class _Dummy:
        compression_rate = 4

    dummy = _Dummy()
    CVLM.set_compression_rate(dummy, 1)
    assert dummy.compression_rate == 1
    CVLM.set_compression_rate(dummy, 8)
    assert dummy.compression_rate == 8
    with pytest.raises(ValueError):
        CVLM.set_compression_rate(dummy, 0)
    with pytest.raises(ValueError):
        CVLM.set_compression_rate(dummy, -1)


def test_set_compression_rate_idempotent():
    """Calling the setter with the same value twice is a no-op of the same value."""
    from modeling import CVLM

    class _Dummy:
        compression_rate = 4

    dummy = _Dummy()
    CVLM.set_compression_rate(dummy, 4)
    CVLM.set_compression_rate(dummy, 4)
    assert dummy.compression_rate == 4


def test_projector_state_dict_keys_and_forward_shape():
    """Projector = pre-RMSNorm + 2-layer GELU MLP; state dict is checkpoint-stable."""
    from modeling import Projector

    projector = Projector(8, 16, use_input_rmsnorm=True)
    keys = set(projector.state_dict())
    assert "input_norm.weight" in keys
    assert "up_proj.weight" in keys
    assert "down_proj.weight" in keys
    assert not any(k.startswith("hidden_layers.") for k in keys)

    x = torch.randn(2, 5, 8)
    y = projector(x)
    assert y.shape == (2, 5, 16)


def test_projector_legacy_no_input_norm_keys():
    """Legacy checkpoints: only up_proj / down_proj (no input_norm.*)."""
    from modeling import Projector

    projector = Projector(8, 16, use_input_rmsnorm=False)
    keys = set(projector.state_dict())
    assert "up_proj.weight" in keys
    assert "down_proj.weight" in keys
    assert not any(k.startswith("input_norm.") for k in keys)
    x = torch.randn(2, 5, 8)
    assert projector(x).shape == (2, 5, 16)
