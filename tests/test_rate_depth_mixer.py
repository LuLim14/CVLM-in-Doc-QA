"""Tests for the Rate-Conditional Depth mixer (modeling.py), its DIAGNOSTIC
instrumentation (contribution ratio, forced-zero gates, gate LR group) and the
balanced MIXED-CR per-step sampler (train_cvlm.py).

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
# CONTRIBUTION RATIO: ||g_i*(Layer_i(h)-h)||_F / ||h||_F per applied layer
# ---------------------------------------------------------------------------


class _ScaleLayer(torch.nn.Module):
    """out = k*h, so (out - h) == (k-1)*h and the ratio is analytic: |g*(k-1)|."""

    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, h):
        return self.k * h


def test_mixer_contrib_depth0_is_empty(tiny_vit_layers):
    from modeling import _mixer_forward

    x = torch.randn(2, 5, 32)
    gates = torch.zeros(len(tiny_vit_layers))
    out, contrib = _mixer_forward(x, tiny_vit_layers, gates, depth=0, return_contrib=True)
    assert out is x
    assert contrib.shape == (0,)
    assert float(contrib.sum()) == 0.0


def test_mixer_contrib_zero_gates_is_zero(tiny_vit_layers):
    """g=0 => the update is exactly 0 => contribution is exactly 0 at every layer."""
    from modeling import _mixer_forward

    torch.manual_seed(20)
    x = torch.randn(2, 5, 32)
    gates = torch.zeros(len(tiny_vit_layers))
    depth = len(tiny_vit_layers)
    out, contrib = _mixer_forward(x, tiny_vit_layers, gates, depth=depth, return_contrib=True)
    assert contrib.shape == (depth,)
    assert torch.equal(contrib, torch.zeros(depth))
    assert float(contrib.sum()) == 0.0
    assert torch.equal(out, x)  # instrumentation must not perturb the forward


def test_mixer_contrib_identity_layer_is_zero():
    """gate=1 but Layer == identity => (Layer(h) - h) == 0 => contribution 0.

    This is the case that makes the ratio strictly better than the raw gate
    value: a wide-open gate on a layer that does nothing contributes nothing.
    """
    from modeling import _mixer_forward

    torch.manual_seed(21)
    layers = torch.nn.ModuleList([_ScaleLayer(1.0), _ScaleLayer(1.0)])
    x = torch.randn(3, 4, 8)
    gates = torch.ones(2)
    out, contrib = _mixer_forward(x, layers, gates, depth=2, return_contrib=True)
    assert torch.equal(contrib, torch.zeros(2))
    assert torch.equal(out, x)


def test_mixer_contrib_matches_closed_form_scale_layer():
    """out=2h => delta = g*h => ratio == g exactly, at every layer and every h."""
    from modeling import _mixer_forward

    torch.manual_seed(22)
    layers = torch.nn.ModuleList([_ScaleLayer(2.0), _ScaleLayer(2.0), _ScaleLayer(2.0)])
    x = torch.randn(2, 6, 8)
    gates = torch.tensor([0.25, 0.5, 1.5])
    out, contrib = _mixer_forward(x, layers, gates, depth=3, return_contrib=True)
    assert torch.allclose(contrib, gates, atol=1e-6)
    # h_{i+1} = (1+g_i) h_i, so the output is the product of (1+g_i) times x.
    scale = (1 + 0.25) * (1 + 0.5) * (1 + 1.5)
    assert torch.allclose(out, x * scale, atol=1e-5)


def test_mixer_contrib_matches_manual_computation():
    """Arbitrary (linear) layer: the returned ratios equal a hand-rolled
    sequential float32 computation, denominator = the layer's INPUT h."""
    from modeling import _mixer_forward

    torch.manual_seed(23)
    layers = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
    x = torch.randn(2, 5, 8)
    gates = torch.tensor([0.3, -0.7])

    _, contrib = _mixer_forward(x, layers, gates, depth=2, return_contrib=True)

    h = x
    expected = []
    for i in range(2):
        delta = gates[i] * (layers[i](h) - h)
        expected.append(
            torch.linalg.vector_norm(delta.detach().float()) / torch.linalg.vector_norm(h.detach().float())
        )
        h = h + delta
    assert torch.allclose(contrib, torch.stack(expected), atol=1e-6)


def test_mixer_contrib_is_detached_and_output_unchanged(tiny_vit_layers):
    """The diagnostic must be free of autograd: no grad_fn on the ratios, and
    the returned hidden state bit-identical to the uninstrumented path."""
    from modeling import _mixer_forward

    torch.manual_seed(24)
    gates = torch.nn.Parameter(torch.full((len(tiny_vit_layers),), 0.4))
    x = torch.randn(2, 5, 32, requires_grad=True)

    out_plain = _mixer_forward(x, tiny_vit_layers, gates, depth=3)
    out_instr, contrib = _mixer_forward(x, tiny_vit_layers, gates, depth=3, return_contrib=True)

    assert torch.equal(out_plain, out_instr)
    assert not contrib.requires_grad
    assert contrib.grad_fn is None
    assert contrib.dtype == torch.float32
    # Gradients still flow through the instrumented path.
    out_instr.sum().backward()
    assert gates.grad is not None and torch.isfinite(gates.grad).all()


def test_mixer_contrib_zero_norm_input_is_zero_not_nan():
    from modeling import _mixer_forward

    layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    x = torch.zeros(1, 3, 4)
    gates = torch.ones(1)
    _, contrib = _mixer_forward(x, layers, gates, depth=1, return_contrib=True)
    assert torch.isfinite(contrib).all()
    assert float(contrib[0]) == 0.0


def test_mixer_contrib_bf16_hidden_state_gives_finite_float32(tiny_vit_layers):
    """Production dtype is bf16 with fp32 gates: the ratio must still come back
    as a finite float32 scalar per layer (norms accumulate in fp32)."""
    from modeling import _mixer_forward

    torch.manual_seed(30)
    layers = torch.nn.ModuleList([torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)]).to(torch.bfloat16)
    x = torch.randn(2, 5, 16).to(torch.bfloat16)
    gates = torch.tensor([0.3, 0.7])  # fp32, cast at the use site
    out, contrib = _mixer_forward(x, layers, gates, depth=2, return_contrib=True)
    assert out.dtype == torch.bfloat16
    assert contrib.dtype == torch.float32
    assert torch.isfinite(contrib).all() and (contrib > 0).all()


def test_mixer_contrib_matches_under_gradient_checkpointing(tiny_vit_layers):
    from modeling import _mixer_forward

    torch.manual_seed(31)
    gates = torch.nn.Parameter(torch.full((len(tiny_vit_layers),), 0.3))
    x = torch.randn(2, 5, 32, requires_grad=True)
    _, c_plain = _mixer_forward(x, tiny_vit_layers, gates, depth=3, return_contrib=True)
    _, c_ckpt = _mixer_forward(x, tiny_vit_layers, gates, depth=3, use_checkpoint=True, return_contrib=True)
    assert torch.allclose(c_plain, c_ckpt, atol=1e-6)


def test_encode_vision_publishes_contrib(tiny_vit_layers):
    """Production path: _encode_vision stashes per-layer ratios sized to the
    ACTIVE depth; zero gates give an all-zero vector, open gates a positive one."""
    m = _build_stub_cvlm(tiny_vit_layers)
    torch.manual_seed(25)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    m.mixer_depth_map = {4: len(tiny_vit_layers)}
    m._encode_vision(ids, mask)
    assert m._last_mixer_contrib is not None
    assert m._last_mixer_contrib.shape == (len(tiny_vit_layers),)
    assert float(m._last_mixer_contrib.sum()) == 0.0  # gates are zero-init

    with torch.no_grad():
        m.mixer_gates.fill_(0.5)
    m._encode_vision(ids, mask)
    assert float(m._last_mixer_contrib.sum()) > 0.0

    # Depth 0: no layer applied => empty vector => total 0.
    m.mixer_depth_map = {4: 0}
    m._encode_vision(ids, mask)
    assert m._last_mixer_contrib.shape == (0,)


def test_encode_vision_computes_nothing_when_mixer_disabled(tiny_vit_layers):
    m = _build_stub_cvlm(tiny_vit_layers)
    m.mixer_layers = None
    m._last_mixer_contrib = None
    torch.manual_seed(26)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    m._encode_vision(ids, mask)
    assert m._last_mixer_contrib is None


def test_forward_publishes_contrib_keys(tiny_vit_layers):
    """The exact contract train_cvlm.py consumes: forward()'s output dict carries
    `mixer_contrib` (per-layer) and `mixer_contrib_total` (sum) when the mixer is
    on, and NEITHER key when it is off."""
    from types import SimpleNamespace

    import torch.nn.functional as F

    class _StubDecoder(torch.nn.Module):
        def __init__(self, d=24, vocab=13):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, d)
            self.head = torch.nn.Linear(d, vocab)
            self.config = SimpleNamespace(hidden_size=d)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds=None, attention_mask=None, labels=None, use_cache=False):
            logits = self.head(inputs_embeds)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(29)
    m = _build_stub_cvlm(tiny_vit_layers)
    m.decoder = _StubDecoder()
    m.mixed_fidelity_p = 0.0
    m.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    m.mixer_depth_map = {4: len(tiny_vit_layers)}
    with torch.no_grad():
        m.mixer_gates.fill_(0.4)
    m.train()

    kw = dict(
        source_input_ids=torch.randint(0, 50, (2, 16)),
        source_attention_mask=torch.ones(2, 16, dtype=torch.long),
        prompt_ids=torch.randint(0, 13, (2, 3)),
        answer_ids=torch.randint(0, 13, (2, 4)),
        answer_labels=torch.randint(0, 13, (2, 4)),
        prompt_mask=torch.ones(2, 3, dtype=torch.long),
        answer_mask=torch.ones(2, 4, dtype=torch.long),
    )
    out = m(**kw)
    assert "mixer_contrib" in out and "mixer_contrib_total" in out
    assert out["mixer_contrib"].shape == (len(tiny_vit_layers),)
    assert float(out["mixer_contrib_total"]) == pytest.approx(float(out["mixer_contrib"].sum()))
    assert float(out["mixer_contrib_total"]) > 0.0
    assert not out["mixer_contrib"].requires_grad

    # Mixer off => the keys must be absent (nothing computed at all).
    m2 = _build_stub_cvlm(tiny_vit_layers)
    m2.decoder = _StubDecoder()
    m2.mixed_fidelity_p = 0.0
    m2.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    m2.mixer_layers = None
    m2._last_mixer_contrib = None
    m2.train()
    out2 = m2(**kw)
    assert "mixer_contrib" not in out2 and "mixer_contrib_total" not in out2


# ---------------------------------------------------------------------------
# eval_cvlm --force_zero_gates
# ---------------------------------------------------------------------------


def test_force_zero_gates_restores_depth0_equivalence(tiny_vit_layers):
    """Zeroing the gates of a 'trained' model must reproduce the no-ViT path
    bit-for-bit, leaving every other weight untouched."""
    from eval_cvlm import force_zero_mixer_gates

    m = _build_stub_cvlm(tiny_vit_layers)
    torch.manual_seed(27)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    m.mixer_layers = None
    out_novit, _ = m._encode_vision(ids, mask)

    m.mixer_layers = tiny_vit_layers
    m.mixer_depth_map = {4: len(tiny_vit_layers)}
    with torch.no_grad():
        m.mixer_gates.copy_(torch.tensor([0.5, -0.3, 0.8]))
    out_open, _ = m._encode_vision(ids, mask)
    assert not torch.allclose(out_open, out_novit)

    n = force_zero_mixer_gates(m)
    assert n == len(tiny_vit_layers)
    assert torch.equal(m.mixer_gates.detach(), torch.zeros(len(tiny_vit_layers)))
    out_zeroed, _ = m._encode_vision(ids, mask)
    assert torch.equal(out_zeroed, out_novit)


def test_force_zero_gates_without_mixer_raises():
    from eval_cvlm import force_zero_mixer_gates

    class _NoMixer(torch.nn.Module):
        pass

    with pytest.raises(ValueError) as exc:
        force_zero_mixer_gates(_NoMixer())
    assert "mixer" in str(exc.value).lower()
    with pytest.raises(ValueError):
        force_zero_mixer_gates(None)


# ---------------------------------------------------------------------------
# --gate_lr_mult param groups + optimizer-resume guard
# ---------------------------------------------------------------------------


class _GatedStub(torch.nn.Module):
    def __init__(self, n_gates=3, with_gates=True):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)  # 2 tensors
        self.extra = torch.nn.Parameter(torch.zeros(2))
        self.mixer_gates = torch.nn.Parameter(torch.zeros(n_gates)) if with_gates else None
        self.frozen = torch.nn.Parameter(torch.zeros(3))
        self.frozen.requires_grad = False


def test_build_param_groups_default_is_single_legacy_group():
    from train_cvlm import _build_param_groups

    m = _GatedStub()
    groups = _build_param_groups(m, lr=1e-4, gate_lr_mult=1.0)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-4
    n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    assert len(groups[0]["params"]) == n_trainable  # frozen tensor excluded
    assert any(p is m.mixer_gates for p in groups[0]["params"])


def test_build_param_groups_gate_group_has_scaled_lr():
    from train_cvlm import _build_param_groups

    m = _GatedStub()
    groups = _build_param_groups(m, lr=1e-4, gate_lr_mult=100.0)
    assert len(groups) == 2
    base, gate = groups
    assert base["lr"] == 1e-4
    assert gate["lr"] == pytest.approx(1e-2)
    assert len(gate["params"]) == 1
    assert gate["params"][0] is m.mixer_gates
    assert all(p is not m.mixer_gates for p in base["params"])
    n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    assert len(base["params"]) + len(gate["params"]) == n_trainable


def test_build_param_groups_rejects_bad_config():
    from train_cvlm import _build_param_groups

    with pytest.raises(ValueError):
        _build_param_groups(_GatedStub(), lr=1e-4, gate_lr_mult=0.0)
    with pytest.raises(ValueError):
        _build_param_groups(_GatedStub(), lr=1e-4, gate_lr_mult=-1.0)
    # mult != 1 without a mixer is a misconfiguration, not a silent no-op.
    with pytest.raises(ValueError) as exc:
        _build_param_groups(_GatedStub(with_gates=False), lr=1e-4, gate_lr_mult=10.0)
    assert "mixer" in str(exc.value).lower()


def test_gate_group_survives_adamw_and_cosine_schedule():
    """The x-mult must hold through AdamW construction and LR decay, and the
    gate group must actually move faster than the base group."""
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from train_cvlm import _build_param_groups

    torch.manual_seed(28)
    m = _GatedStub()
    groups = _build_param_groups(m, lr=1e-3, gate_lr_mult=100.0)
    opt = torch.optim.AdamW(groups, lr=1e-3, weight_decay=0.0)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 1e-1]
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)

    # One update with identical gradients. Adam normalises the step, so the
    # displacement ratio is not exactly the LR ratio, but the gate must move
    # far further than a base parameter fed the same gradient.
    for p in (m.extra, m.mixer_gates):
        p.grad = torch.ones_like(p)
    before_extra = m.extra.detach().clone()
    before_gate = m.mixer_gates.detach().clone()
    opt.step()
    d_extra = float((m.extra.detach() - before_extra).abs().mean())
    d_gate = float((m.mixer_gates.detach() - before_gate).abs().mean())
    assert d_gate > d_extra * 10

    for _ in range(3):
        sched.step()
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[1] == pytest.approx(lrs[0] * 100.0)


def test_check_optimizer_groups_match_detects_layout_change():
    from train_utils import check_optimizer_groups_match

    m = _GatedStub()
    one = torch.optim.AdamW([{"params": [p for p in m.parameters() if p.requires_grad], "lr": 1e-4}])
    two = torch.optim.AdamW(
        [
            {"params": [p for p in m.parameters() if p.requires_grad and p is not m.mixer_gates], "lr": 1e-4},
            {"params": [m.mixer_gates], "lr": 1e-2},
        ]
    )
    # Same layout: no complaint.
    check_optimizer_groups_match(one, one.state_dict())
    check_optimizer_groups_match(two, two.state_dict())
    # Layout changed across resume: explicit error naming the culprit flag.
    with pytest.raises(RuntimeError) as exc:
        check_optimizer_groups_match(two, one.state_dict(), trainer_path="trainer_step_5.pt")
    msg = str(exc.value)
    assert "gate_lr_mult" in msg and "trainer_step_5.pt" in msg
    with pytest.raises(RuntimeError):
        check_optimizer_groups_match(one, two.state_dict())


def test_check_optimizer_groups_match_detects_size_change():
    from train_utils import check_optimizer_groups_match

    m = _GatedStub()
    full = torch.optim.AdamW([{"params": [p for p in m.parameters() if p.requires_grad], "lr": 1e-4}])
    partial = torch.optim.AdamW([{"params": [m.extra], "lr": 1e-4}])
    with pytest.raises(RuntimeError) as exc:
        check_optimizer_groups_match(full, partial.state_dict())
    assert "param group 0" in str(exc.value)


def test_load_cvlm_checkpoint_raises_on_group_mismatch(tmp_path):
    """End-to-end resume guard: a checkpoint written by a 1-group run must not
    load silently into a 2-group (--gate_lr_mult) optimizer."""
    from safetensors.torch import save_file
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from train_utils import load_cvlm_checkpoint

    m = _GatedStub()
    opt1 = torch.optim.AdamW([{"params": [p for p in m.parameters() if p.requires_grad], "lr": 1e-4}])
    sch1 = CosineAnnealingLR(opt1, T_max=10)
    model_path = str(tmp_path / "model_step_5.safetensors")
    trainer_path = str(tmp_path / "trainer_step_5.pt")
    save_file({k: v.detach().contiguous() for k, v in m.state_dict().items()}, model_path)
    torch.save(
        {
            "optimizer": opt1.state_dict(),
            "scheduler": sch1.state_dict(),
            "next_start_epoch": 2,
            "global_step": 5,
        },
        trainer_path,
    )

    # Same layout resumes fine.
    m2 = _GatedStub()
    opt_ok = torch.optim.AdamW([{"params": [p for p in m2.parameters() if p.requires_grad], "lr": 1e-4}])
    sch_ok = CosineAnnealingLR(opt_ok, T_max=10)
    nxt, gstep = load_cvlm_checkpoint(model_path, trainer_path, m2, opt_ok, sch_ok, torch.device("cpu"))
    assert (nxt, gstep) == (2, 5)

    # Split layout must fail loudly.
    m3 = _GatedStub()
    opt_bad = torch.optim.AdamW(
        [
            {"params": [p for p in m3.parameters() if p.requires_grad and p is not m3.mixer_gates], "lr": 1e-4},
            {"params": [m3.mixer_gates], "lr": 1e-2},
        ]
    )
    sch_bad = CosineAnnealingLR(opt_bad, T_max=10)
    with pytest.raises(RuntimeError) as exc:
        load_cvlm_checkpoint(model_path, trainer_path, m3, opt_bad, sch_bad, torch.device("cpu"))
    assert "gate_lr_mult" in str(exc.value)


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
