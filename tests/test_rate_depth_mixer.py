"""Tests for the Rate-Conditional Depth mixer (modeling.py) and the balanced
MIXED-CR per-step sampler (train_cvlm.py).

No HF downloads: the mixer forward is exercised on a tiny config-init ViTModel
(ViTConfig built directly), the depth rule / parsers on plain dicts, and the
CVLM method via an unbound-dummy call (same pattern as test_set_compression_rate).
CPU only.

Campaign boundary arms covered explicitly: the all-0 map (byte-for-byte
identical to the no-ViT path) and the all-12 map (full-depth ANCHORED arm).
"""
from collections import Counter

import pytest
import torch

# ---------------------------------------------------------------------------
# mixer_depth_map parser
# ---------------------------------------------------------------------------


def test_parse_mixer_depth_map_basic():
    from modeling import _parse_mixer_depth_map

    assert _parse_mixer_depth_map("1:0,2:1,4:3,8:6,16:12") == {1: 0, 2: 1, 4: 3, 8: 6, 16: 12}


def test_parse_mixer_depth_map_boundary_arms():
    """Campaign arms: all-0 (novit, pooling-only) and all-12 (fixed12-ANCHORED)."""
    from modeling import _parse_mixer_depth_map

    assert _parse_mixer_depth_map("1:0,2:0,4:0,8:0,16:0") == {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    all12 = _parse_mixer_depth_map("1:12,2:12,4:12,8:12,16:12", max_depth=12)
    assert all12 == {1: 12, 2: 12, 4: 12, 8: 12, 16: 12}
    # depth == max_depth ("use every layer") is legal; depth > max_depth is not.
    with pytest.raises(ValueError):
        _parse_mixer_depth_map("1:12,2:12,4:12,8:12,16:12", max_depth=11)


def test_parse_mixer_depth_map_empty_returns_none():
    from modeling import _parse_mixer_depth_map

    assert _parse_mixer_depth_map("") is None
    assert _parse_mixer_depth_map("   ") is None
    assert _parse_mixer_depth_map(None) is None


def test_parse_mixer_depth_map_strips_whitespace():
    from modeling import _parse_mixer_depth_map

    assert _parse_mixer_depth_map(" 1:0 , 2:1 ") == {1: 0, 2: 1}


@pytest.mark.parametrize(
    "bad",
    [
        "1",  # no colon
        "1:",  # empty depth
        "a:2",  # non-integer cr
        "1:b",  # non-integer depth
        "0:3",  # cr < 1
        "2:-1",  # negative depth
        "1:2,1:3",  # duplicate cr key
    ],
)
def test_parse_mixer_depth_map_malformed_raises(bad):
    from modeling import _parse_mixer_depth_map

    with pytest.raises(ValueError) as exc:
        _parse_mixer_depth_map(bad)
    assert str(exc.value)


# ---------------------------------------------------------------------------
# depth lookup: exact key / nearest lower / smallest
# ---------------------------------------------------------------------------


def test_mixer_depth_for_cr_exact_and_nearest_lower():
    from modeling import _mixer_depth_for_cr

    m = {1: 0, 2: 1, 4: 3, 8: 6, 16: 12}
    # Exact keys.
    for cr, d in m.items():
        assert _mixer_depth_for_cr(m, cr) == d
    # Unseen cr -> nearest LOWER key.
    assert _mixer_depth_for_cr(m, 3) == 1  # from cr=2
    assert _mixer_depth_for_cr(m, 5) == 3  # from cr=4
    assert _mixer_depth_for_cr(m, 6) == 3
    assert _mixer_depth_for_cr(m, 12) == 6  # from cr=8
    # The sbatch's post-eval case: cr=32 with a map up to 16 -> depth of 16.
    assert _mixer_depth_for_cr(m, 32) == 12


def test_mixer_depth_for_cr_below_smallest_key():
    from modeling import _mixer_depth_for_cr

    m = {4: 2, 8: 5}
    # No lower key -> smallest key's depth.
    assert _mixer_depth_for_cr(m, 1) == 2
    assert _mixer_depth_for_cr(m, 2) == 2
    assert _mixer_depth_for_cr(m, 6) == 2  # nearest lower = 4
    assert _mixer_depth_for_cr(m, 100) == 5


def test_mixer_depth_for_cr_boundary_arm_maps():
    from modeling import _mixer_depth_for_cr

    all0 = {1: 0, 2: 0, 4: 0, 8: 0, 16: 0}
    all12 = {1: 12, 2: 12, 4: 12, 8: 12, 16: 12}
    for cr in (1, 2, 3, 4, 8, 16, 32, 100):
        assert _mixer_depth_for_cr(all0, cr) == 0
        assert _mixer_depth_for_cr(all12, cr) == 12


def test_mixer_depth_for_method_on_dummy():
    """Method form used by _encode_vision, without building a full CVLM."""
    from modeling import CVLM

    class _Dummy:
        mixer_depth_map = {1: 0, 2: 1, 16: 12}

    assert CVLM._mixer_depth_for(_Dummy(), 1) == 0
    assert CVLM._mixer_depth_for(_Dummy(), 2) == 1
    assert CVLM._mixer_depth_for(_Dummy(), 32) == 12

    class _Off:
        mixer_depth_map = None

    with pytest.raises(RuntimeError):
        CVLM._mixer_depth_for(_Off(), 4)


# ---------------------------------------------------------------------------
# mixer forward: identity / gating / checkpointing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_vit_layers():
    """3 real ViTLayer blocks from a tiny config-init ViTModel (no download)."""
    from transformers import ViTConfig, ViTModel

    torch.manual_seed(0)
    cfg = ViTConfig(
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=16,
        patch_size=8,
    )
    model = ViTModel(cfg)
    return torch.nn.ModuleList(list(model.encoder.layer))


def test_mixer_depth0_is_novit_path_byte_for_byte(tiny_vit_layers):
    """depth=0 returns the INPUT TENSOR ITSELF — exactly the no-ViT branch
    (vit_output = vit_input, no final layernorm)."""
    from modeling import _mixer_forward

    torch.manual_seed(1)
    x = torch.randn(2, 5, 32)
    gates = torch.zeros(len(tiny_vit_layers))
    out = _mixer_forward(x, tiny_vit_layers, gates, depth=0)
    assert out is x  # same object => trivially byte-identical
    assert torch.equal(out, x)


def test_mixer_zero_gates_identity_at_positive_depth(tiny_vit_layers):
    """g=0: h + 0*(Layer(h)-h) == h bitwise, for any depth up to the full stack
    (covers the all-12-style full-depth arm at its zero-gated starting point)."""
    from modeling import _mixer_forward

    torch.manual_seed(2)
    x = torch.randn(2, 5, 32)
    gates = torch.zeros(len(tiny_vit_layers))
    for depth in (1, 2, len(tiny_vit_layers)):
        out = _mixer_forward(x, tiny_vit_layers, gates, depth=depth)
        assert torch.equal(out, x), f"depth={depth}: zero-gated mixer must be an exact identity"


def test_mixer_nonzero_gates_diverge(tiny_vit_layers):
    """Guards against short-circuiting: opened gates must change the output."""
    from modeling import _mixer_forward

    torch.manual_seed(3)
    x = torch.randn(2, 5, 32)
    gates = torch.full((len(tiny_vit_layers),), 0.5)
    out = _mixer_forward(x, tiny_vit_layers, gates, depth=len(tiny_vit_layers))
    assert not torch.allclose(out, x)


def test_mixer_partial_depth_uses_only_first_layers(tiny_vit_layers):
    """depth=d must be insensitive to gates/layers beyond index d-1."""
    from modeling import _mixer_forward

    torch.manual_seed(4)
    x = torch.randn(2, 5, 32)
    g1 = torch.tensor([0.7, 0.0, 0.0])
    g2 = torch.tensor([0.7, 123.0, -55.0])  # gates past depth must not matter
    out1 = _mixer_forward(x, tiny_vit_layers, g1, depth=1)
    out2 = _mixer_forward(x, tiny_vit_layers, g2, depth=1)
    assert torch.equal(out1, out2)


def test_mixer_checkpoint_matches_plain_forward(tiny_vit_layers):
    """Manual torch.utils.checkpoint path must reproduce outputs AND input
    grads of the plain path (non-reentrant recompute, CPU-deterministic)."""
    from modeling import _mixer_forward

    torch.manual_seed(5)
    gates = torch.nn.Parameter(torch.full((len(tiny_vit_layers),), 0.3))

    x1 = torch.randn(2, 5, 32, requires_grad=True)
    out_ref = _mixer_forward(x1, tiny_vit_layers, gates, depth=3, use_checkpoint=False)
    out_ref.sum().backward()

    x2 = x1.detach().clone().requires_grad_(True)
    out_ck = _mixer_forward(x2, tiny_vit_layers, gates, depth=3, use_checkpoint=True)
    out_ck.sum().backward()

    assert torch.allclose(out_ref, out_ck, atol=0, rtol=0)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6, rtol=1e-5)


def test_mixer_handles_tuple_returning_layers():
    """Older transformers ViTLayer returns (hidden, ...) tuples; newer returns a
    bare Tensor. Both must be unwrapped — naive `layer(h)[0]` on a Tensor would
    silently slice the batch dimension."""
    from modeling import _mixer_forward

    class _TupleLayer(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = torch.nn.Linear(dim, dim)

        def forward(self, h):
            return (self.lin(h),)

    torch.manual_seed(6)
    layers = torch.nn.ModuleList([_TupleLayer(8)])
    x = torch.randn(2, 3, 8)
    gates = torch.ones(1)
    out = _mixer_forward(x, layers, gates, depth=1)
    expected = x + 1.0 * (layers[0].lin(x) - x)
    assert torch.allclose(out, expected)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# _encode_vision integration: mixer branch vs no-ViT branch on the real CVLM code
# ---------------------------------------------------------------------------


def _build_stub_cvlm(tiny_vit_layers):
    """CVLM instance with heavy __init__ bypassed: only the attributes
    _encode_vision touches are installed (tiny stub encoder, real Projectors,
    real mixer layers). Lets us run the ACTUAL production forward branch
    without any HF download."""
    from types import SimpleNamespace

    from modeling import CVLM, Projector

    class _StubEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, 16)

        def forward(self, input_ids=None, attention_mask=None):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    torch.manual_seed(10)
    m = CVLM.__new__(CVLM)
    torch.nn.Module.__init__(m)
    m.model_args = SimpleNamespace(unfreeze_encoder_top_k=0)
    m.text_encoder = _StubEncoder()
    m.pool_queries = torch.nn.Parameter(torch.zeros(2, 16))
    m.compression_rate = 4
    m.max_vision_len = 8
    m.text_projector = Projector(16, 32)
    m.vision_pos_embed = torch.nn.Embedding(8, 32)
    m.vision_projector = Projector(32, 24)
    m.vision_encoder = None
    m.mixer_layers = tiny_vit_layers
    m.mixer_gates = torch.nn.Parameter(torch.zeros(len(tiny_vit_layers)))
    m.mixer_depth_map = {4: 0}
    m._mixer_grad_ckpt = False
    m.eval()
    return m


def test_encode_vision_mixer_depth0_equals_novit_branch(tiny_vit_layers):
    """End-to-end through the real _encode_vision: the mixer at depth 0 must be
    BYTE-identical to the no-ViT branch (all shared modules are the very same
    instances, so any difference would come from the mixer path itself)."""
    m = _build_stub_cvlm(tiny_vit_layers)
    torch.manual_seed(11)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    m.mixer_depth_map = {4: 0}  # depth 0 at cr=4
    out_mixer, vmask_mixer = m._encode_vision(ids, mask)

    m.mixer_layers = None  # switch the SAME instance to the no-ViT branch
    out_novit, vmask_novit = m._encode_vision(ids, mask)

    assert torch.equal(vmask_mixer, vmask_novit)
    assert torch.equal(out_mixer, out_novit)

    # Zero gates at FULL depth must also collapse to the no-ViT path bitwise.
    m.mixer_layers = tiny_vit_layers
    m.mixer_depth_map = {4: len(tiny_vit_layers)}
    out_zero_gate, vmask_zero_gate = m._encode_vision(ids, mask)
    assert torch.equal(vmask_zero_gate, vmask_novit)
    assert torch.equal(out_zero_gate, out_novit)


def test_encode_vision_mixer_open_gates_diverge_from_novit(tiny_vit_layers):
    """With opened gates the mixer must actually transform the sequence."""
    m = _build_stub_cvlm(tiny_vit_layers)
    torch.manual_seed(12)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    m.mixer_layers = None
    out_novit, _ = m._encode_vision(ids, mask)

    m.mixer_layers = tiny_vit_layers
    m.mixer_depth_map = {4: len(tiny_vit_layers)}
    with torch.no_grad():
        m.mixer_gates.fill_(0.5)
    out_mixer, _ = m._encode_vision(ids, mask)
    assert not torch.allclose(out_mixer, out_novit)


# ---------------------------------------------------------------------------
# mixed_cr_set parser + balanced deterministic sampler
# ---------------------------------------------------------------------------


def test_parse_mixed_cr_set_basic():
    from train_cvlm import _parse_mixed_cr_set

    assert _parse_mixed_cr_set("1 2 4 8 16") == [1, 2, 4, 8, 16]
    assert _parse_mixed_cr_set("  16   8 ") == [16, 8]
    assert _parse_mixed_cr_set("") is None
    assert _parse_mixed_cr_set("   ") is None
    assert _parse_mixed_cr_set(None) is None


@pytest.mark.parametrize("bad", ["1 2 foo", "0", "-1 2", "4 4"])
def test_parse_mixed_cr_set_malformed_raises(bad):
    from train_cvlm import _parse_mixed_cr_set

    with pytest.raises(ValueError):
        _parse_mixed_cr_set(bad)


def test_mixed_cr_sampler_deterministic_across_instances():
    """Guarantee (a): identical sequence across arms/runs for a given seed."""
    from train_cvlm import MixedCrSampler

    s1 = MixedCrSampler([1, 2, 4, 8, 16], 42 + 30000)
    s2 = MixedCrSampler([1, 2, 4, 8, 16], 42 + 30000)
    seq1 = [s1.cr_for_step(k) for k in range(1, 501)]
    seq2 = [s2.cr_for_step(k) for k in range(1, 501)]
    assert seq1 == seq2


def test_mixed_cr_sampler_balanced_blocks():
    """Guarantee (b): every consecutive block of len(set) steps is a permutation
    of the full set, so any prefix divisible by len(set) has EQUAL per-cr
    update counts."""
    from train_cvlm import MixedCrSampler

    cr_set = [1, 2, 4, 8, 16]
    s = MixedCrSampler(cr_set, 123)
    n = 1000
    seq = [s.cr_for_step(k) for k in range(1, n + 1)]
    for i in range(0, n, 5):
        assert sorted(seq[i : i + 5]) == sorted(cr_set), f"block at {i} is not a permutation"
    for m in range(5, n + 1, 5):
        counts = Counter(seq[:m])
        assert all(counts[c] == m // 5 for c in cr_set), f"prefix {m} is unbalanced: {counts}"


def test_mixed_cr_sampler_resume_reconstruction():
    """Resume contract: a FRESH sampler asked directly for step G reproduces the
    uninterrupted run's value at G (pure function of seed + global_step; nothing
    is stored in checkpoints)."""
    from train_cvlm import MixedCrSampler

    seed = 7 + 30000
    ref = MixedCrSampler([1, 2, 4, 8, 16], seed)
    seq = [ref.cr_for_step(k) for k in range(1, 301)]

    fresh = MixedCrSampler([1, 2, 4, 8, 16], seed)
    assert fresh.cr_for_step(237) == seq[236]  # mid-run resume, no prior calls
    assert [fresh.cr_for_step(k) for k in range(1, 301)] == seq


def test_mixed_cr_sampler_input_order_insensitive():
    """The set is sorted before shuffling, so --mixed_cr_set formatting order
    cannot desynchronise arms."""
    from train_cvlm import MixedCrSampler

    s1 = MixedCrSampler([1, 2, 4, 8, 16], 99)
    s2 = MixedCrSampler([16, 8, 1, 4, 2], 99)
    assert [s1.cr_for_step(k) for k in range(1, 101)] == [s2.cr_for_step(k) for k in range(1, 101)]


def test_mixed_cr_sampler_seed_changes_sequence():
    from train_cvlm import MixedCrSampler

    a = MixedCrSampler([1, 2, 4, 8, 16], 30042)
    b = MixedCrSampler([1, 2, 4, 8, 16], 30043)
    assert [a.cr_for_step(k) for k in range(1, 101)] != [b.cr_for_step(k) for k in range(1, 101)]


def test_mixed_cr_sampler_rejects_bad_inputs():
    from train_cvlm import MixedCrSampler

    with pytest.raises(ValueError):
        MixedCrSampler([], 1)
    s = MixedCrSampler([1, 2], 1)
    with pytest.raises(ValueError):
        s.cr_for_step(0)
