"""Tests for the two advisor-mandated ablation arms.

P0 — MeanPool+Project (``--mean_pool`` / ``MEAN_POOL=1``)
    The strong arm uses a LEARNABLE attention pool with K latent queries. The
    queries are zero-initialised, and at exactly zero the softmax is uniform so
    the pool *starts* as a mean — but they move during training (measured: norm
    5.25 at the end of a run), so an attention-pool run is NOT a mean-pool run.
    This arm pins the pool to the mean for the whole trajectory: the pool is an
    EXACT masked mean, ``pool_queries`` is frozen at zero, gets no gradient, is
    excluded from the optimizer and from the saved trainable state.

P1 — Pool+1L (``--vit_num_layers 1`` with ``--vit_init random``)
    The existing comparison is 0 layers vs 12, from which nothing can be said
    about the shape of the depth curve. The middle point must differ from the
    12-layer arm in DEPTH ONLY, so it is the same plain ViT stack truncated to
    its first block — same block type, same config-init draw, same final
    LayerNorm, same padding-mask setting.
    It is deliberately NOT ``--mixer_depth_map "1:1,..."``: the mixer is
    zero-gated, so its "depth 1" is the identity at init and the arm would
    measure gate dynamics instead of depth. Those tests are here too, as the
    evidence for that decision.

No HF downloads: the pool and the truncation are tested directly, and the
production ``CVLM._encode_vision`` is driven through the same stub pattern as
tests/test_rate_depth_mixer.py. CPU only.
"""
import json
from types import SimpleNamespace

import pytest
import torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _manual_masked_mean(hidden, mask, cr, max_vision_len):
    """Reference masked mean, written independently of the implementation.

    Loops chunks in Python and averages the REAL token slice of the original
    ``hidden`` (``hidden[b, start:end].mean(0)``) — no padding, no rescaling, no
    view/reshape tricks. This is the thing the arm claims the pool is equal to.
    """
    B, L, H = hidden.shape
    lens = mask.sum(dim=1).tolist()
    v_list = [min((int(n) + cr - 1) // cr, max_vision_len) for n in lens]
    v_max = max(max(v_list), 1)
    out = torch.zeros(B, v_max, H, dtype=hidden.dtype)
    v_mask = torch.zeros(B, v_max, dtype=torch.long)
    for b in range(B):
        n_real = int(lens[b])
        v_i = v_list[b]
        if n_real == 0 or v_i == 0:
            continue
        n_use = min(n_real, v_i * cr)
        for v in range(v_i):
            start = v * cr
            end = min((v + 1) * cr, n_use)
            out[b, v] = hidden[b, start:end].mean(dim=0)
            v_mask[b, v] = 1
    return out, v_mask


def _ragged_batch(batch, length, hidden, dtype):
    """Random [B, L, H] + a mask with a DIFFERENT amount of padding per row."""
    x = torch.randn(batch, length, hidden, dtype=dtype)
    mask = torch.ones(batch, length, dtype=torch.long)
    mask[1, length - 5:] = 0
    mask[2, length - 1:] = 0
    # Padded positions must not carry content (the collator zero-pads too).
    return x * mask.unsqueeze(-1).to(dtype), mask


def _tiny_vit_config(n_layers=3, dim=32):
    from transformers import ViTConfig

    return ViTConfig(
        hidden_size=dim,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=16,
        patch_size=8,
    )


def _tiny_vit_layers(n_layers=3, dim=32, seed=0):
    from transformers import ViTModel

    torch.manual_seed(seed)
    model = ViTModel(_tiny_vit_config(n_layers, dim))
    return torch.nn.ModuleList(list(model.encoder.layer))


def _build_stub_cvlm(mean_pool=None, mixer_layers=None, mixer_depth_map=None, pool_latents=4):
    """CVLM with the heavy __init__ bypassed (no HF download).

    Only the attributes ``_encode_vision`` touches are installed, so the test
    exercises the ACTUAL production branch. ``mean_pool=None`` installs no
    attribute at all — that is the pre-change model surface, and it must keep
    working through the ``getattr`` default.
    """
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
    m.pool_queries = torch.nn.Parameter(torch.zeros(pool_latents, 16))
    m.compression_rate = 4
    m.max_vision_len = 8
    m.text_projector = Projector(16, 32)
    m.vision_pos_embed = torch.nn.Embedding(8, 32)
    m.vision_projector = Projector(32, 24)
    m.vision_encoder = None
    m.mixer_layers = mixer_layers
    m.mixer_gates = (
        torch.nn.Parameter(torch.zeros(len(mixer_layers))) if mixer_layers is not None else None
    )
    m.mixer_depth_map = mixer_depth_map
    m._mixer_grad_ckpt = False
    if mean_pool is not None:
        m.mean_pool = bool(mean_pool)
    m.eval()
    return m


# ===========================================================================
# P0 — MeanPool+Project
# ===========================================================================

# ---------------------------------------------------------------------------
# output == explicit masked mean, for FULL and PARTIAL chunks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("cr", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("length", [17, 19, 33, 64])
def test_mean_pool_branch_is_bitwise_masked_mean(dtype, cr, length):
    """``pool_queries=None`` (the --mean_pool path) == manual masked mean, BITWISE.

    torch.equal, not allclose: the arm's claim is that the pool IS the mean, so
    a tolerance would leave the claim unverified. Covers ragged padding across
    the batch, partial last chunks, bf16 (the training dtype) and a non
    power-of-two cr — exactly the cases where the older
    ``mean(dim=1) * cr / true_last`` rescale was mathematically right but
    numerically off by an ULP.
    """
    from modeling import _chunked_attention_pool

    torch.manual_seed(hash((cr, length)) % (2**31))
    hidden, mask = _ragged_batch(3, length, 24, dtype)

    pooled, v_mask = _chunked_attention_pool(
        hidden, mask, compression_rate=cr, max_vision_len=1024, pool_queries=None
    )
    ref, ref_mask = _manual_masked_mean(hidden, mask, cr, 1024)

    assert torch.equal(v_mask, ref_mask)
    assert torch.equal(pooled, ref), f"max |diff| = {(pooled - ref).abs().max().item()}"


def test_mean_pool_full_and_partial_chunks_explicitly():
    """The two cases stated separately, on a hand-checkable example.

    L_real = 13 with cr = 4 -> three FULL chunks (tokens 0-3, 4-7, 8-11) and one
    PARTIAL chunk holding a single real token (12). Each is compared with the
    mean of exactly its own real tokens.
    """
    from modeling import _chunked_attention_pool

    torch.manual_seed(21)
    hidden = torch.randn(1, 20, 8)
    mask = torch.zeros(1, 20, dtype=torch.long)
    mask[0, :13] = 1
    hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)

    pooled, v_mask = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=64, pool_queries=None
    )
    assert v_mask.sum().item() == 4  # ceil(13/4)

    for v, (start, end) in enumerate([(0, 4), (4, 8), (8, 12), (12, 13)]):
        expected = hidden[0, start:end].mean(dim=0)
        assert torch.equal(pooled[0, v], expected), f"chunk {v} ({start}:{end})"


def test_mean_pool_ignores_padded_tokens_entirely():
    """Padding must carry ZERO weight: perturbing padded positions (which a
    collator may leave as garbage) must not move a single output bit."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(7)
    hidden = torch.randn(2, 20, 16)
    mask = torch.ones(2, 20, dtype=torch.long)
    mask[0, 13:] = 0  # 13 real tokens -> cr=4 leaves a 1-token partial chunk
    mask[1, 18:] = 0
    clean = hidden * mask.unsqueeze(-1).to(hidden.dtype)

    dirty = clean.clone()
    dirty[0, 13:] = 1e3
    dirty[1, 18:] = -1e3

    out_clean, m_clean = _chunked_attention_pool(
        clean, mask, compression_rate=4, max_vision_len=64, pool_queries=None
    )
    out_dirty, m_dirty = _chunked_attention_pool(
        dirty, mask, compression_rate=4, max_vision_len=64, pool_queries=None
    )
    assert torch.equal(m_clean, m_dirty)
    assert torch.equal(out_clean, out_dirty)


def test_encode_vision_mean_pool_ignores_pool_queries():
    """Through the REAL _encode_vision: with --mean_pool the queries are inert.

    Same instance, same shared modules; only pool_queries changes. A mean-pool
    model must be bit-identical with zero and with wildly non-zero queries,
    while the attention-pool model must NOT be — otherwise the flag is not
    actually routing to the parameter-free branch.
    """
    torch.manual_seed(11)
    ids = torch.randint(0, 50, (2, 19))
    mask = torch.ones(2, 19, dtype=torch.long)
    mask[1, 15:] = 0

    m = _build_stub_cvlm(mean_pool=True)
    out_zero, vmask_zero = m._encode_vision(ids, mask)
    with torch.no_grad():
        m.pool_queries.normal_(mean=0.0, std=3.0)
    out_loud, vmask_loud = m._encode_vision(ids, mask)
    assert torch.equal(vmask_zero, vmask_loud)
    assert torch.equal(out_zero, out_loud), "mean_pool=True still depends on pool_queries"

    # Control: the same non-zero queries DO change the attention-pool arm.
    m.mean_pool = False
    out_attn, _ = m._encode_vision(ids, mask)
    assert not torch.equal(out_attn, out_loud)


# ---------------------------------------------------------------------------
# pool_queries: no gradient, not in the optimizer, stays exactly zero
# ---------------------------------------------------------------------------


def test_mean_pool_gives_pool_queries_no_gradient():
    """Backprop through _encode_vision must leave pool_queries.grad as None."""
    m = _build_stub_cvlm(mean_pool=True)
    m.pool_queries.requires_grad_(False)  # what CVLM.__init__ does for this arm
    torch.manual_seed(12)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    out, _ = m._encode_vision(ids, mask)
    out.sum().backward()

    assert m.pool_queries.grad is None
    # Sanity: the rest of the compression path still trains.
    assert m.text_projector.up_proj.weight.grad is not None


def test_frozen_pool_queries_are_absent_from_optimizer_param_groups():
    """The REAL param-group builder must not hand the frozen queries to AdamW."""
    from train_cvlm import _build_param_groups

    m = _build_stub_cvlm(mean_pool=True)
    m.pool_queries.requires_grad_(False)

    groups = _build_param_groups(m, lr=1e-2, gate_lr_mult=1.0)
    in_groups = [p for g in groups for p in g["params"]]
    assert all(p is not m.pool_queries for p in in_groups), (
        "frozen pool_queries leaked into an optimizer param group"
    )
    assert in_groups, "every other trainable parameter must still be optimized"


def test_frozen_pool_queries_survive_an_optimizer_step_unchanged():
    """End-to-end with the REAL param-group builder + AdamW: after a full
    forward/backward/step the frozen queries are still exactly zero."""
    from train_cvlm import _build_param_groups

    m = _build_stub_cvlm(mean_pool=True)
    m.pool_queries.requires_grad_(False)
    optimizer = torch.optim.AdamW(_build_param_groups(m, lr=1e-2, gate_lr_mult=1.0), lr=1e-2)

    torch.manual_seed(13)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    before = m.pool_queries.detach().clone()

    out, _ = m._encode_vision(ids, mask)
    out.sum().backward()
    optimizer.step()

    assert torch.equal(m.pool_queries.detach(), before)
    assert torch.equal(m.pool_queries.detach(), torch.zeros_like(before))
    assert m.pool_queries.grad is None


def test_attention_pool_queries_do_move_without_the_flag():
    """The reason this arm exists: WITHOUT --mean_pool the queries are trainable
    and an optimizer step moves them off zero."""
    from train_cvlm import _build_param_groups

    m = _build_stub_cvlm(mean_pool=False)
    groups = _build_param_groups(m, lr=1e-2, gate_lr_mult=1.0)
    assert any(any(p is m.pool_queries for p in g["params"]) for g in groups)

    optimizer = torch.optim.AdamW(groups, lr=1e-2)
    torch.manual_seed(14)
    ids = torch.randint(0, 50, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)

    out, _ = m._encode_vision(ids, mask)
    out.sum().backward()
    assert m.pool_queries.grad is not None
    optimizer.step()
    assert not torch.equal(m.pool_queries.detach(), torch.zeros_like(m.pool_queries))


# ---------------------------------------------------------------------------
# checkpoint round-trip + the eval-side architecture guard
# ---------------------------------------------------------------------------


def test_mean_pool_checkpoint_omits_pool_queries_and_still_loads():
    """A mean-pool checkpoint has no pool_queries key (frozen params are not
    saved), and loading it into a mean-pool model must be accepted — the model
    rebuilds the zeros itself."""
    from train_utils import state_dict_for_safetensors, validate_partial_checkpoint_load

    m = _build_stub_cvlm(mean_pool=True)
    m.pool_queries.requires_grad_(False)
    saved = state_dict_for_safetensors(m)
    assert "pool_queries" not in saved
    assert any(k.startswith("text_projector.") for k in saved)

    fresh = _build_stub_cvlm(mean_pool=True)
    fresh.pool_queries.requires_grad_(False)
    missing, unexpected = fresh.load_state_dict(saved, strict=False)
    validate_partial_checkpoint_load(
        fresh, missing, unexpected, "mean_pool.safetensors",
        allowed_missing_trainable={"pool_queries"},
    )
    assert "pool_queries" in missing  # rebuilt, not restored
    assert not unexpected
    assert torch.equal(fresh.pool_queries.detach(), torch.zeros_like(fresh.pool_queries))
    assert torch.equal(fresh.text_projector.up_proj.weight, m.text_projector.up_proj.weight)


def _write_manifest(tmp_path, architecture):
    (tmp_path / "initialization_manifest.json").write_text(
        json.dumps({"schema_version": 1, "architecture": architecture}), encoding="utf-8"
    )
    ckpt = tmp_path / "model_step_100.safetensors"
    ckpt.write_bytes(b"")  # the guard runs before any load
    return str(ckpt)


def test_eval_without_mean_pool_on_a_mean_checkpoint_is_rejected(tmp_path):
    """THE silent-swap guard. A mean-pool checkpoint loads cleanly into an
    attention-pool model (pool_queries is an allowed missing key), so nothing
    downstream would notice. The manifest check must refuse it up front."""
    from eval_cvlm import verify_architecture_against_manifest

    ckpt = _write_manifest(tmp_path, {"no_vit": True, "mean_pool": True})

    with pytest.raises(RuntimeError) as exc:
        verify_architecture_against_manifest(ckpt, SimpleNamespace(mean_pool=False, vit_num_layers=0))
    assert "mean_pool" in str(exc.value)

    # ...and the correct invocation passes.
    architecture = verify_architecture_against_manifest(
        ckpt, SimpleNamespace(mean_pool=True, vit_num_layers=0)
    )
    assert architecture["mean_pool"] is True


def test_eval_with_mean_pool_on_an_attention_checkpoint_is_rejected(tmp_path):
    """The mirror image: --mean_pool on a normal checkpoint is just as wrong."""
    from eval_cvlm import verify_architecture_against_manifest

    ckpt = _write_manifest(tmp_path, {"no_vit": True})  # legacy: key absent => False

    with pytest.raises(RuntimeError):
        verify_architecture_against_manifest(ckpt, SimpleNamespace(mean_pool=True, vit_num_layers=0))
    assert (
        verify_architecture_against_manifest(
            ckpt, SimpleNamespace(mean_pool=False, vit_num_layers=0)
        )
        is not None
    )


def test_eval_vit_num_layers_mismatch_is_rejected(tmp_path):
    """Same guard for the depth arm: 1 layer trained, 12 evaluated is silent
    otherwise (strict=False just reports the missing blocks as 'unexpected')."""
    from eval_cvlm import verify_architecture_against_manifest

    ckpt = _write_manifest(tmp_path, {"vit_init": "random", "vit_num_layers": 1})

    with pytest.raises(RuntimeError) as exc:
        verify_architecture_against_manifest(ckpt, SimpleNamespace(mean_pool=False, vit_num_layers=0))
    assert "vit_num_layers" in str(exc.value)
    assert (
        verify_architecture_against_manifest(
            ckpt, SimpleNamespace(mean_pool=False, vit_num_layers=1)
        )
        is not None
    )


def test_architecture_guard_is_a_noop_without_a_manifest(tmp_path):
    """Pre-manifest runs and ad-hoc checkpoints must keep evaluating."""
    from eval_cvlm import verify_architecture_against_manifest

    ckpt = tmp_path / "model_step_1.safetensors"
    ckpt.write_bytes(b"")
    assert (
        verify_architecture_against_manifest(
            str(ckpt), SimpleNamespace(mean_pool=True, vit_num_layers=3)
        )
        is None
    )
    assert (
        verify_architecture_against_manifest("", SimpleNamespace(mean_pool=True, vit_num_layers=0))
        is None
    )


# ---------------------------------------------------------------------------
# shared-init comparability with the paired seed-43 Pool+Project arm
# ---------------------------------------------------------------------------


def test_mean_pool_does_not_change_shared_init_hashes():
    """The arm must be comparable with Pool+Project at the same seed: freezing
    pool_queries changes requires_grad, NOT bytes, and the flag consumes no RNG,
    so every recorded init hash (projectors, pos-embed, pool_queries) is
    identical between the two arms."""
    from train_cvlm import _module_sha256, _tensor_sha256

    attention_arm = _build_stub_cvlm(mean_pool=False)
    mean_arm = _build_stub_cvlm(mean_pool=True)
    mean_arm.pool_queries.requires_grad_(False)

    for name in ("text_projector", "vision_projector", "vision_pos_embed"):
        assert _module_sha256(getattr(attention_arm, name)) == _module_sha256(
            getattr(mean_arm, name)
        ), name
    assert _tensor_sha256(attention_arm.pool_queries) == _tensor_sha256(mean_arm.pool_queries)
    assert attention_arm.pool_queries.requires_grad
    assert not mean_arm.pool_queries.requires_grad


def test_mean_pool_recorded_in_manifest_only_when_enabled():
    """Back-compat: the key is written ONLY when the arm is on, so every
    pre-existing run stays resumable (the manifest 'architecture' dict is
    compared as a whole), while a wrong-flag resume still fails."""
    from train_cvlm import _initialization_manifest

    model = _build_stub_cvlm(mean_pool=True)
    model.decoder = SimpleNamespace(config=SimpleNamespace(_commit_hash="c"))
    model.text_encoder.config = SimpleNamespace(_commit_hash="c")
    model.model_args = SimpleNamespace(
        model_name_or_path="d", text_encoder_name="t", vision_encoder_name="v"
    )

    def _args(**over):
        base = dict(
            seed=43, vit_init="pretrained", freeze_vit=False, no_vit=True,
            mixer_depth_map="", mask_vit_padding=False, vit_apply_final_layernorm=True,
            compact_decoder_positions=False, num_pool_latents=4, mean_pool=False,
            vit_num_layers=0, model_name_or_path="d", text_encoder_name="t",
            vision_encoder_name="v",
        )
        base.update(over)
        return SimpleNamespace(**base)

    off = _initialization_manifest(model, _args())["architecture"]
    on = _initialization_manifest(model, _args(mean_pool=True))["architecture"]
    depth = _initialization_manifest(model, _args(vit_num_layers=1))["architecture"]

    assert "mean_pool" not in off and "vit_num_layers" not in off
    assert on["mean_pool"] is True
    assert depth["vit_num_layers"] == 1
    assert off != on and off != depth


# ---------------------------------------------------------------------------
# flag OFF -> previous behaviour, bit for bit
# ---------------------------------------------------------------------------


def test_missing_mean_pool_attribute_behaves_like_false():
    """Old model objects (and old stubs) have no .mean_pool attribute at all.
    The getattr default must keep them on the attention path bit for bit."""
    ids = torch.randint(0, 50, (2, 18))
    mask = torch.ones(2, 18, dtype=torch.long)
    mask[0, 14:] = 0

    legacy = _build_stub_cvlm(mean_pool=None)  # attribute absent
    explicit = _build_stub_cvlm(mean_pool=False)
    assert not hasattr(legacy, "mean_pool")
    with torch.no_grad():
        legacy.pool_queries.normal_(std=2.0, generator=torch.Generator().manual_seed(3))
        explicit.pool_queries.copy_(legacy.pool_queries)

    out_legacy, vmask_legacy = legacy._encode_vision(ids, mask)
    out_explicit, vmask_explicit = explicit._encode_vision(ids, mask)
    assert torch.equal(vmask_legacy, vmask_explicit)
    assert torch.equal(out_legacy, out_explicit)


def test_flag_off_keeps_the_attention_pool_path():
    """With the flag off the pool must still be the softmax-weighted attention
    pool: non-zero queries change the output, zero queries reproduce the mean
    (the documented zero-init drop-in property)."""
    from modeling import _chunked_attention_pool

    torch.manual_seed(15)
    hidden, mask = _ragged_batch(3, 19, 16, torch.float64)

    mean_out, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8, pool_queries=None
    )
    zero_q, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8,
        pool_queries=torch.zeros(4, 16, dtype=torch.float64),
    )
    loud_q, _ = _chunked_attention_pool(
        hidden, mask, compression_rate=4, max_vision_len=8,
        pool_queries=torch.randn(4, 16, dtype=torch.float64),
    )
    assert torch.allclose(mean_out, zero_q, atol=1e-12)
    assert not torch.allclose(mean_out, loud_q)


def test_model_arguments_defaults_keep_both_arms_off():
    from modeling import ModelArguments

    args = ModelArguments()
    assert args.mean_pool is False
    assert args.vit_num_layers == 0


def test_train_and_eval_both_expose_the_new_flags():
    """The flags MUST exist on both sides: an eval built without them silently
    evaluates a different architecture."""
    import inspect

    import eval_cvlm
    import train_cvlm

    for module in (train_cvlm, eval_cvlm):
        src = inspect.getsource(module.parse_args)
        for flag, env in (("--mean_pool", "MEAN_POOL"), ("--vit_num_layers", "VIT_NUM_LAYERS")):
            assert f'"{flag}"' in src, f"{module.__name__} has no {flag}"
            assert env in src, f"{module.__name__} does not read {env}"


# ===========================================================================
# P1 — Pool+1L via --vit_num_layers (plain ViT, NOT the zero-gated mixer)
# ===========================================================================


def _truncated_vit(n_layers_total, keep, seed=5):
    """Build a ViTModel and apply the PRODUCTION truncation helper to it.

    Deliberately calls modeling._truncate_vit_layers — the very function
    CVLM.__init__ uses — so these tests cannot drift from the shipped rule.
    """
    from transformers import ViTModel

    from modeling import _truncate_vit_layers

    torch.manual_seed(seed)
    model = ViTModel(_tiny_vit_config(n_layers_total))
    _truncate_vit_layers(model, keep)
    return model


def test_truncate_helper_validates_its_input():
    from transformers import ViTModel

    from modeling import _truncate_vit_layers

    model = ViTModel(_tiny_vit_config(3))
    assert _truncate_vit_layers(model, 0) == 3  # 0 = keep everything
    with pytest.raises(ValueError):
        _truncate_vit_layers(model, -1)
    with pytest.raises(ValueError):
        _truncate_vit_layers(model, 4)  # more layers than exist


def test_truncation_keeps_the_first_blocks_bit_for_bit():
    """The N-layer arm's blocks must be BITWISE the 12-layer arm's first N: the
    stack is built at full depth from the same RNG state and only then cut, so
    depth is the sole difference between the arms."""
    full = _truncated_vit(4, keep=0)
    cut = _truncated_vit(4, keep=1)

    assert len(cut.encoder.layer) == 1
    assert cut.config.num_hidden_layers == 1
    for key, value in cut.encoder.layer[0].state_dict().items():
        assert torch.equal(value, full.encoder.layer[0].state_dict()[key]), key
    # The final LayerNorm is untouched — the N-layer arm keeps it exactly as the
    # 12-layer arm has it.
    for key, value in cut.layernorm.state_dict().items():
        assert torch.equal(value, full.layernorm.state_dict()[key]), key


def test_truncation_does_not_shift_rng_for_later_modules():
    """The full stack is constructed either way, so anything initialised AFTER
    the ViT (projectors, pos-embed, pool queries) draws the same numbers. If the
    arm had been built from a shallower config instead, this would fail."""
    outs = []
    for keep in (0, 1):
        # _truncated_vit seeds internally, then builds the FULL stack and cuts.
        _truncated_vit(4, keep=keep, seed=5)
        outs.append(torch.randn(8))  # stands in for the next module's init draw
    assert torch.equal(outs[0], outs[1])


def test_truncation_removes_the_dropped_blocks_from_parameters():
    """Dropped blocks must leave parameters()/state_dict entirely: otherwise
    they sit unused in the optimizer, in checkpoints and (under DDP) trip
    find_unused_parameters=False."""
    cut = _truncated_vit(4, keep=1)
    names = {n for n, _ in cut.named_parameters()}
    assert any(n.startswith("encoder.layer.0.") for n in names)
    for dropped in ("encoder.layer.1.", "encoder.layer.2.", "encoder.layer.3."):
        assert not any(n.startswith(dropped) for n in names), dropped


def test_truncation_at_full_depth_is_a_noop():
    """vit_num_layers == the config depth must be bit-identical to 0 (= all)."""
    full = _truncated_vit(3, keep=0)
    same = _truncated_vit(3, keep=3)
    assert len(same.encoder.layer) == len(full.encoder.layer) == 3
    assert same.config.num_hidden_layers == 3
    sd_full, sd_same = full.state_dict(), same.state_dict()
    assert sd_full.keys() == sd_same.keys()
    for key in sd_full:
        assert torch.equal(sd_full[key], sd_same[key]), key


def test_one_layer_stack_forward_equals_the_first_block_alone():
    """A 1-layer plain stack applies exactly one ViTLayer — no gate, no
    interpolation, no identity shortcut. This is what makes it a genuine depth
    point rather than a gate-dynamics measurement."""
    cut = _truncated_vit(3, keep=1)
    torch.manual_seed(31)
    x = torch.randn(2, 5, 32)

    stack_out = cut.encoder(x).last_hidden_state
    block_out = cut.encoder.layer[0](x)
    if isinstance(block_out, tuple):
        block_out = block_out[0]

    assert torch.equal(stack_out, block_out)
    assert not torch.allclose(stack_out, x), "a real layer must move the representation"


def test_vit_num_layers_rejects_incompatible_flags():
    """vit_skip/mixer_depth_map own the depth already; combining them would
    silently pick one and report the other."""
    from modeling import ModelArguments

    assert ModelArguments(vit_num_layers=1).vit_num_layers == 1
    # The guard itself lives in CVLM.__init__; assert its exact conditions are
    # spelled out there (constructing a CVLM needs HF downloads).
    import inspect

    from modeling import CVLM

    src = inspect.getsource(CVLM.__init__)
    assert "vit_num_layers is incompatible with vit_skip" in src
    assert "vit_num_layers is incompatible with mixer_depth_map" in src
    assert "vit_num_layers must be >= 0" in inspect.getsource(CVLM.__init__)


def test_zero_gated_mixer_depth1_is_the_identity_at_init():
    """WHY the 1-layer arm is NOT --mixer_depth_map "1:1,...": at init the gated
    stack returns its input bitwise, i.e. it is the 0-layer arm. Such an arm
    measures whether the gate opens, not what one layer of depth costs."""
    from modeling import _mixer_forward

    layers = _tiny_vit_layers(n_layers=3)
    gates = torch.nn.Parameter(torch.zeros(3))
    torch.manual_seed(17)
    x = torch.randn(2, 5, 32)

    out = _mixer_forward(x, layers, gates, depth=1)
    assert torch.equal(out, x), "zero-gated depth-1 is indistinguishable from depth 0 at init"

    plain = layers[0](x)
    if isinstance(plain, tuple):
        plain = plain[0]
    assert not torch.allclose(plain, x), "the same block applied plainly is NOT the identity"


def test_mixer_depth_map_depth1_still_parses_if_ever_needed():
    """The depth-1 map remains valid input (nothing was broken by choosing the
    plain-ViT route), and the lookup gives 1 at every curriculum cr."""
    from modeling import _mixer_depth_for_cr, _parse_mixer_depth_map

    parsed = _parse_mixer_depth_map("1:1,2:1,4:1,8:1,16:1", max_depth=12)
    assert parsed == {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
    for cr in (1, 2, 3, 4, 5, 8, 16, 32):
        assert _mixer_depth_for_cr(parsed, cr) == 1, f"cr={cr}"


def test_encode_vision_plain_vit_runs_at_every_curriculum_cr():
    """Through the REAL _encode_vision + set_compression_rate: the truncated
    plain stack is applied at cr=1,2,4,8,16 and the vision geometry is the
    expected ceil(L/cr) at each rate."""
    m = _build_stub_cvlm(mean_pool=False)
    m.max_vision_len = 64
    m.vision_pos_embed = torch.nn.Embedding(64, 32)
    m.vision_encoder = _truncated_vit(3, keep=1)
    m.vit_apply_final_layernorm = True
    m.mask_vit_padding = False

    torch.manual_seed(18)
    ids = torch.randint(0, 50, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)

    for cr in (1, 2, 4, 8, 16):
        m.set_compression_rate(cr)
        out, vmask = m._encode_vision(ids, mask)
        assert out.size(1) == vmask.size(1) == (32 + cr - 1) // cr, f"cr={cr}"
        assert torch.isfinite(out).all(), f"cr={cr}"
    assert len(m.vision_encoder.encoder.layer) == 1
