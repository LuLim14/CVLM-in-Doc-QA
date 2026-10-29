"""fp32 master weights + the ViT bitwise-drift diagnostic.

The experiment these guard: training runs entirely in bf16, and an AdamW update
of size ~lr is BELOW half an ULP of a large (pretrained) bf16 weight, so it is
rounded away every step while the ~4x smaller random-init weights still move.
That makes "the pretrained visual prior hurts" indistinguishable from a pure
floating-point artifact. These tests pin (a) the legacy path stays byte-for-byte,
(b) the master-weight path actually lands sub-ULP updates, (c) resume keeps the
fp32 masters, (d) the diagnostic counts bitwise-unchanged weights correctly.
"""

from types import SimpleNamespace

import pytest
import torch


def _const_grad_steps(params, optimizer, steps: int, grad_value: float = 1.0):
    """Run ``steps`` optimizer updates with a constant gradient on ``params``."""
    for _ in range(steps):
        for param in params:
            param.grad = torch.full_like(param, grad_value)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[tensor.dtype]
    return tensor.detach().contiguous().view(view).clone()


# --------------------------------------------------------------------------
# 1. The ULP problem itself, and that the flag fixes it.
# --------------------------------------------------------------------------


def test_legacy_bf16_adamw_rounds_pretrained_scale_updates_away():
    """The confounder: at lr=1e-4 a ~1.0 weight cannot move, a ~0.02 weight can."""
    from train_utils import build_optimizer

    big = torch.nn.Parameter(torch.full((8,), 1.0, dtype=torch.bfloat16))
    small = torch.nn.Parameter(torch.full((8,), 0.02, dtype=torch.bfloat16))
    big_before, small_before = _bits(big), _bits(small)

    optimizer = build_optimizer(
        [{"params": [big, small], "lr": 1e-4}],
        lr=1e-4,
        betas=(0.9, 0.999),
        fp32_master_weights=False,
    )
    assert type(optimizer) is torch.optim.AdamW
    _const_grad_steps([big, small], optimizer, steps=50)

    # 50 steps x ~1e-4 never accumulate: each step restarts from the same bf16
    # value and rounds straight back to it.
    assert torch.equal(_bits(big), big_before), "pretrained-scale weight must be frozen by bf16 rounding"
    # The moments are bf16 too — the state itself is low precision.
    assert optimizer.state[big]["exp_avg"].dtype == torch.bfloat16
    # A random-init-scale weight has a 64x finer ULP and does move.
    assert not torch.equal(_bits(small), small_before)


def test_master_weights_land_sub_ulp_updates():
    from train_utils import MasterWeightAdamW, build_optimizer

    big = torch.nn.Parameter(torch.full((8,), 1.0, dtype=torch.bfloat16))
    optimizer = build_optimizer(
        [{"params": [big], "lr": 1e-4}],
        lr=1e-4,
        betas=(0.9, 0.999),
        fp32_master_weights=True,
    )
    assert isinstance(optimizer, MasterWeightAdamW)
    assert optimizer.mastered_numel == 8

    # A single step is still below half an ULP: the master moved, the bf16
    # parameter has not (yet).
    _const_grad_steps([big], optimizer, steps=1)
    master = optimizer.master_pairs[0][1]
    assert master.dtype == torch.float32
    assert float(master.detach()[0]) == pytest.approx(1.0 - 1e-4, abs=2e-6)
    assert float(big.detach()[0]) == 1.0

    _const_grad_steps([big], optimizer, steps=49)

    # 50 accumulated steps DO cross the rounding threshold.
    assert float(master.detach()[0]) == pytest.approx(1.0 - 50e-4, abs=1e-5)
    assert float(big.detach()[0]) < 1.0
    assert float(big.detach()[0]) == pytest.approx(0.995, abs=4e-3)
    assert big.dtype == torch.bfloat16, "parameters must stay bf16 for forward/backward"
    # fp32 masters => fp32 moments (the legacy bf16 exp_avg_sq stores squared
    # gradients at bf16's 0.39%-0.78% relative granularity).
    assert optimizer.state[master]["exp_avg"].dtype == torch.float32


def test_fp32_params_are_optimized_in_place_without_a_master_copy():
    from train_utils import MasterWeightAdamW

    gates = torch.nn.Parameter(torch.zeros(3))  # fp32, like mixer_gates
    weight = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    optimizer = MasterWeightAdamW([{"params": [gates, weight], "lr": 1e-3}], lr=1e-3)

    assert [id(p) for _, p in optimizer.master_pairs] != [id(gates)]
    assert len(optimizer.master_pairs) == 1
    assert optimizer.param_groups[0]["params"][0] is gates  # untouched pass-through

    gates.grad = torch.ones_like(gates)
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    assert float(gates.detach()[0]) == pytest.approx(-1e-3, abs=1e-6)


# --------------------------------------------------------------------------
# 2. Flag OFF is byte-for-byte the legacy trajectory.
# --------------------------------------------------------------------------


def _tiny_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Linear(6, 3))
    return model.to(torch.bfloat16)


def test_flag_off_matches_plain_adamw_bit_for_bit():
    from train_utils import build_optimizer

    reference = _tiny_model(0)
    candidate = _tiny_model(0)
    ref_opt = torch.optim.AdamW(
        [{"params": list(reference.parameters()), "lr": 1e-3}],
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    cand_opt = build_optimizer(
        [{"params": list(candidate.parameters()), "lr": 1e-3}],
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fp32_master_weights=False,
    )
    torch.manual_seed(7)
    batches = [torch.randn(4, 6).to(torch.bfloat16) for _ in range(6)]
    for batch in batches:
        for model, optimizer in ((reference, ref_opt), (candidate, cand_opt)):
            optimizer.zero_grad(set_to_none=True)
            model(batch).square().mean().backward()
            optimizer.step()

    for ref_p, cand_p in zip(reference.parameters(), candidate.parameters()):
        assert torch.equal(_bits(ref_p), _bits(cand_p))
    assert type(cand_opt) is torch.optim.AdamW


def test_master_weights_change_the_trajectory_but_keep_the_dtype():
    from train_utils import build_optimizer

    legacy = _tiny_model(0)
    mastered = _tiny_model(0)
    legacy_opt = build_optimizer(
        [{"params": list(legacy.parameters()), "lr": 1e-3}],
        lr=1e-3,
        betas=(0.9, 0.999),
        fp32_master_weights=False,
    )
    master_opt = build_optimizer(
        [{"params": list(mastered.parameters()), "lr": 1e-3}],
        lr=1e-3,
        betas=(0.9, 0.999),
        fp32_master_weights=True,
    )
    torch.manual_seed(7)
    batches = [torch.randn(4, 6).to(torch.bfloat16) for _ in range(8)]
    for batch in batches:
        for model, optimizer in ((legacy, legacy_opt), (mastered, master_opt)):
            optimizer.zero_grad(set_to_none=True)
            model(batch).square().mean().backward()
            optimizer.step()

    assert all(p.dtype == torch.bfloat16 for p in mastered.parameters())
    assert any(
        not torch.equal(_bits(a), _bits(b))
        for a, b in zip(legacy.parameters(), mastered.parameters())
    )
    # zero_grad must clear the MODEL grads too, not only the master ones.
    master_opt.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in mastered.parameters())


# --------------------------------------------------------------------------
# 2b. Scope: only the selected tensors change; everything else is legacy.
# --------------------------------------------------------------------------


class _ScopedModel(torch.nn.Module):
    """A ViT-ish stack (in scope) next to decoder/projector-ish tensors."""

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.vision_encoder = torch.nn.Sequential(torch.nn.Linear(6, 6), torch.nn.Linear(6, 6))
        self.decoder = torch.nn.Linear(6, 6)
        self.text_projector = torch.nn.Linear(6, 6)
        self.mixer_layers = None
        self.mixer_gates = None
        self.to(torch.bfloat16)
        with torch.no_grad():
            # Pretrained-scale magnitudes: their bf16 ULP swallows an lr=1e-4 step.
            for parameter in self.parameters():
                parameter.fill_(1.0)

    def loss(self, batch):
        hidden = self.vision_encoder(batch)
        return (self.decoder(hidden) + self.text_projector(hidden)).square().mean()


def _run_scoped(fp32_master_weights: bool, scope: str, steps: int = 40, foreach=None):
    from train_cvlm import _vision_scope_params
    from train_utils import build_optimizer

    model = _ScopedModel(seed=0)
    master_params = _vision_scope_params(model) if scope == "vision" else None
    kwargs = {} if foreach is None else {"foreach": foreach}
    optimizer = build_optimizer(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4}],
        lr=1e-4,
        betas=(0.9, 0.999),
        fp32_master_weights=fp32_master_weights,
        master_params=master_params,
        **kwargs,
    )
    torch.manual_seed(4)
    batches = [torch.randn(3, 6).to(torch.bfloat16) for _ in range(steps)]
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        model.loss(batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return model, optimizer


def test_vision_scope_selects_only_the_vit_stack():
    from train_cvlm import _vision_scope_params

    model = _ScopedModel(seed=0)
    selected = {id(p) for p in _vision_scope_params(model)}
    assert selected == {id(p) for p in model.vision_encoder.parameters()}
    assert id(model.decoder.weight) not in selected
    assert id(model.text_projector.weight) not in selected


@pytest.mark.parametrize("foreach", [None, True])
def test_vision_scope_leaves_non_vision_params_bit_identical(foreach):
    legacy, legacy_opt = _run_scoped(False, "all", foreach=foreach)
    scoped, scoped_opt = _run_scoped(True, "vision", foreach=foreach)

    assert len(scoped_opt.master_pairs) == len(list(scoped.vision_encoder.parameters()))

    for name in ("decoder", "text_projector"):
        for legacy_p, scoped_p in zip(
            getattr(legacy, name).parameters(), getattr(scoped, name).parameters()
        ):
            assert torch.equal(_bits(legacy_p), _bits(scoped_p)), (
                f"{name} must follow the legacy bf16 path bit-for-bit under scope=vision"
            )
    # ...and their optimizer state is still the legacy bf16 one.
    for parameter in scoped.decoder.parameters():
        assert scoped_opt.state[parameter]["exp_avg"].dtype == torch.bfloat16

    # The ViT stack, which legacy cannot move at all, does move.
    assert all(
        torch.equal(_bits(a), _bits(b))
        for a, b in zip(legacy.vision_encoder.parameters(), _ScopedModel(seed=0).vision_encoder.parameters())
    ), "sanity: the legacy run must leave the 1.0-magnitude ViT weights untouched"
    assert any(
        not torch.equal(_bits(a), _bits(b))
        for a, b in zip(legacy.vision_encoder.parameters(), scoped.vision_encoder.parameters())
    )
    for parameter, master in scoped_opt.master_pairs:
        assert scoped_opt.state[master]["exp_avg"].dtype == torch.float32
        assert parameter.dtype == torch.bfloat16


def test_scope_all_masters_everything_including_the_decoder():
    _, all_opt = _run_scoped(True, "all", steps=2)
    model = _ScopedModel(seed=0)
    assert all_opt.mastered_numel == sum(p.numel() for p in model.parameters())


def test_scope_mismatch_on_resume_is_rejected():
    from train_utils import MasterWeightAdamW

    _, scoped_opt = _run_scoped(True, "vision", steps=2)
    _, all_opt = _run_scoped(True, "all", steps=2)
    assert isinstance(scoped_opt, MasterWeightAdamW)
    with pytest.raises(RuntimeError, match="master SCOPE"):
        scoped_opt.load_state_dict(
            {**all_opt.state_dict(), "param_groups": scoped_opt.state_dict()["param_groups"]}
        )


# --------------------------------------------------------------------------
# 3. Resume keeps the fp32 masters.
# --------------------------------------------------------------------------


def test_resume_restores_master_copies_and_continues_identically(tmp_path):
    from train_utils import build_optimizer

    def make(seed: int):
        model = _tiny_model(seed)
        optimizer = build_optimizer(
            [{"params": list(model.parameters()), "lr": 1e-4}],
            lr=1e-4,
            betas=(0.9, 0.999),
            fp32_master_weights=True,
        )
        return model, optimizer

    torch.manual_seed(11)
    batches = [torch.randn(4, 6).to(torch.bfloat16) for _ in range(10)]

    uninterrupted, opt_a = make(0)
    for batch in batches:
        opt_a.zero_grad(set_to_none=True)
        uninterrupted(batch).square().mean().backward()
        opt_a.step()

    interrupted, opt_b = make(0)
    for batch in batches[:5]:
        opt_b.zero_grad(set_to_none=True)
        interrupted(batch).square().mean().backward()
        opt_b.step()

    # Round-trip through disk exactly like save_cvlm_checkpoint does.
    path = tmp_path / "trainer_step_5.pt"
    torch.save({"optimizer": opt_b.state_dict()}, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)["optimizer"]
    assert "fp32_master_weights" in saved
    assert len(saved["fp32_master_weights"]["params"]) == len(opt_b.master_pairs)
    # The masters carry information the bf16 checkpoint cannot represent.
    assert any(
        not torch.equal(master, master.to(torch.bfloat16).float())
        for _, master in opt_b.master_pairs
    )

    # Fresh process: weights come from the (rounded) safetensors, optimizer
    # state from the trainer file.
    resumed, opt_c = make(0)
    with torch.no_grad():
        for dst, src in zip(resumed.parameters(), interrupted.parameters()):
            dst.copy_(src)
    opt_c.load_state_dict(saved)

    for (_, restored), (_, original) in zip(opt_c.master_pairs, opt_b.master_pairs):
        assert torch.equal(restored.cpu(), original.cpu())
    assert opt_c.param_groups[0]["lr"] == opt_b.param_groups[0]["lr"]

    for batch in batches[5:]:
        opt_c.zero_grad(set_to_none=True)
        resumed(batch).square().mean().backward()
        opt_c.step()

    for a, b in zip(uninterrupted.parameters(), resumed.parameters()):
        assert torch.equal(_bits(a), _bits(b))


def test_end_to_end_checkpoint_roundtrip_through_train_utils(tmp_path):
    """The real save/load helpers, incl. the LR scheduler and the group check."""
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from train_utils import build_optimizer, load_cvlm_checkpoint, save_cvlm_checkpoint

    def make(seed: int):
        model = _tiny_model(seed)
        optimizer = build_optimizer(
            [{"params": list(model.parameters()), "lr": 1e-4}],
            lr=1e-4,
            betas=(0.9, 0.999),
            fp32_master_weights=True,
        )
        # CosineAnnealingLR requires a real torch Optimizer instance.
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0)
        return model, optimizer, scheduler

    torch.manual_seed(13)
    batches = [torch.randn(4, 6).to(torch.bfloat16) for _ in range(6)]

    reference, ref_opt, ref_sched = make(0)
    for batch in batches:
        ref_opt.zero_grad(set_to_none=True)
        reference(batch).square().mean().backward()
        ref_opt.step()
        ref_sched.step()

    partial, part_opt, part_sched = make(0)
    for batch in batches[:3]:
        part_opt.zero_grad(set_to_none=True)
        partial(batch).square().mean().backward()
        part_opt.step()
        part_sched.step()
    save_cvlm_checkpoint(
        str(tmp_path),
        partial,
        part_opt,
        part_sched,
        next_start_epoch=1,
        global_step=3,
        is_master=True,
        next_start_batch=4,
    )

    resumed, res_opt, res_sched = make(0)
    load_cvlm_checkpoint(
        str(tmp_path / "model_step_3.safetensors"),
        str(tmp_path / "trainer_step_3.pt"),
        resumed,
        res_opt,
        res_sched,
        torch.device("cpu"),
    )
    for (_, restored), (_, original) in zip(res_opt.master_pairs, part_opt.master_pairs):
        assert torch.equal(restored.detach(), original.detach())

    for batch in batches[3:]:
        res_opt.zero_grad(set_to_none=True)
        resumed(batch).square().mean().backward()
        res_opt.step()
        res_sched.step()

    for a, b in zip(reference.parameters(), resumed.parameters()):
        assert torch.equal(_bits(a), _bits(b))


def test_master_weight_resume_guard(capsys):
    from train_utils import build_optimizer, check_master_weights_match

    weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    master_opt = build_optimizer(
        [{"params": [weight], "lr": 1e-4}], lr=1e-4, betas=(0.9, 0.999), fp32_master_weights=True
    )
    plain_opt = build_optimizer(
        [{"params": [weight], "lr": 1e-4}], lr=1e-4, betas=(0.9, 0.999), fp32_master_weights=False
    )
    master_state = master_opt.state_dict()
    plain_state = plain_opt.state_dict()

    with pytest.raises(RuntimeError, match="fp32 master weights"):
        check_master_weights_match(plain_opt, master_state, "trainer_step_10.pt")
    check_master_weights_match(master_opt, plain_state, "trainer_step_10.pt")
    assert "WITHOUT fp32 master weights" in capsys.readouterr().out
    check_master_weights_match(master_opt, master_state, "trainer_step_10.pt")
    check_master_weights_match(plain_opt, plain_state, "trainer_step_10.pt")


def test_master_weight_count_mismatch_is_rejected():
    from train_utils import MasterWeightAdamW

    small = MasterWeightAdamW([{"params": [torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))]}], lr=1e-4)
    two = MasterWeightAdamW(
        [
            {
                "params": [
                    torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16)),
                    torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16)),
                ]
            }
        ],
        lr=1e-4,
    )
    with pytest.raises(RuntimeError, match="master-weight resume mismatch"):
        two.load_state_dict(
            {
                **small.state_dict(),
                "param_groups": two.state_dict()["param_groups"],
            }
        )


# --------------------------------------------------------------------------
# 4. The ViT bitwise / drift diagnostic.
# --------------------------------------------------------------------------


class _StubVit(torch.nn.Module):
    """Minimal stand-in exposing the attribute path VitDriftTracker walks."""

    def __init__(self, layers: int = 2, width: int = 4):
        super().__init__()
        layer_stack = torch.nn.ModuleList(
            [torch.nn.Linear(width, width, bias=False) for _ in range(layers)]
        )
        self.mixer_layers = None
        self.mixer_gates = None
        self.vision_encoder = SimpleNamespace(
            encoder=SimpleNamespace(layer=layer_stack),
            layernorm=torch.nn.LayerNorm(width),
        )
        # Keep the submodules registered so .parameters() works for the test.
        self._layers = layer_stack
        self._norm = self.vision_encoder.layernorm


def test_drift_tracker_counts_bitwise_unchanged_weights():
    from train_cvlm import VitDriftTracker

    torch.manual_seed(3)
    model = _StubVit(layers=2, width=4)
    tracker = VitDriftTracker(model)
    assert tracker.enabled
    assert tracker.branch == "vit"
    # 2 x (4x4) weights + LayerNorm weight + bias
    assert tracker.numel == 2 * 16 + 4 + 4

    first = tracker.measure()
    assert first["vit_frozen_frac"] == 1.0
    assert first["vit_drift"] == 0.0

    with torch.no_grad():
        model._layers[0].weight.view(-1)[:8] += 1.0
    second = tracker.measure()
    assert second["vit_frozen_frac"] == pytest.approx(1.0 - 8 / tracker.numel)
    assert second["vit_drift"] > 0.0

    # Nothing changed since the previous call -> back to a fully frozen window,
    # while drift from INIT stays where it was.
    third = tracker.measure()
    assert third["vit_frozen_frac"] == 1.0
    assert third["vit_drift"] == pytest.approx(second["vit_drift"])


def test_drift_tracker_reports_master_drift_and_hidden_residual():
    """The live bf16 weights understate what the optimizer has learned."""
    from train_cvlm import VitDriftTracker
    from train_utils import build_optimizer

    torch.manual_seed(17)
    model = _StubVit(layers=1, width=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
    model = model.to(torch.bfloat16)

    optimizer = build_optimizer(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4}],
        lr=1e-4,
        betas=(0.9, 0.999),
        fp32_master_weights=True,
    )
    tracker = VitDriftTracker(
        model,
        master_lookup={id(p): m for p, m in optimizer.master_pairs},
    )
    assert tracker.has_masters
    assert tracker.mastered_tensors == len(tracker.params)

    _const_grad_steps([p for p in model.parameters()], optimizer, steps=5)
    stats = tracker.measure()

    # bf16 has not budged yet, but the masters have — and the gap is reported.
    assert stats["vit_frozen_frac"] == 1.0
    assert stats["vit_drift"] == 0.0
    assert stats["vit_master_drift"] > 0.0
    assert stats["vit_master_resid"] > 0.0
    assert stats["vit_master_resid_max"] == pytest.approx(5e-4, rel=0.2)

    # Without masters the extra fields are absent (None), not silently zero.
    plain = VitDriftTracker(model)
    plain_stats = plain.measure()
    assert plain_stats["vit_master_drift"] is None
    assert plain_stats["vit_master_resid"] is None
    assert plain_stats["vit_master_resid_max"] is None


def test_drift_tracker_relative_l2_matches_manual_computation():
    from train_cvlm import VitDriftTracker

    torch.manual_seed(5)
    model = _StubVit(layers=1, width=4)
    tracker = VitDriftTracker(model)
    initial = [p.detach().clone() for _, p in tracker.params]
    with torch.no_grad():
        model._layers[0].weight.add_(0.25)
        model._norm.bias.add_(-0.5)
    stats = tracker.measure()

    delta_sq = sum(
        float(((p.detach() - ref).float() ** 2).sum())
        for (_, p), ref in zip(tracker.params, initial)
    )
    init_sq = sum(float((ref.float() ** 2).sum()) for ref in initial)
    assert stats["vit_drift"] == pytest.approx((delta_sq**0.5) / (init_sq**0.5), rel=1e-6)


def test_drift_tracker_anchors_init_to_mechanistic_init_file():
    from train_cvlm import VitDriftTracker

    torch.manual_seed(9)
    model = _StubVit(layers=1, width=4)
    # Emulate a resume: the live weights already moved away from step 0.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1)
    step_zero = {
        "vision_encoder_layers.0.weight": model._layers[0].weight.detach() - 0.1,
        "vision_encoder_final_layernorm.weight": model._norm.weight.detach() - 0.1,
        "vision_encoder_final_layernorm.bias": model._norm.bias.detach() - 0.1,
    }
    tracker = VitDriftTracker(model, init_state=step_zero)
    assert tracker.init_from_file == 3
    assert tracker.measure()["vit_drift"] > 0.0

    # Without the file the tracker anchors on the current weights instead.
    fresh = VitDriftTracker(model)
    assert fresh.init_from_file == 0
    assert fresh.measure()["vit_drift"] == 0.0


def test_drift_tracker_names_match_mechanistic_init_keys():
    """The drift anchor only works if both functions agree on parameter names."""
    from transformers import ViTConfig, ViTModel

    from train_cvlm import VitDriftTracker, _compression_path_initial_state

    config = ViTConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=8,
        patch_size=4,
    )

    class TinyCvlm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_encoder = ViTModel(config)
            self.mixer_layers = None
            self.mixer_gates = None
            self.text_projector = torch.nn.Linear(4, 4)
            self.vision_projector = torch.nn.Linear(4, 4)
            self.vision_pos_embed = torch.nn.Embedding(4, 4)
            self.pool_queries = torch.nn.Parameter(torch.zeros(1, 4))

    model = TinyCvlm()
    init_state = _compression_path_initial_state(model)
    tracker = VitDriftTracker(model, init_state=init_state)

    assert tracker.enabled
    assert tracker.init_from_file == len(tracker.params) > 0
    assert {name for name, _ in tracker.params} <= set(init_state)
    assert tracker.measure()["vit_drift"] == 0.0
    # Embeddings/pooler are unused by the compression path and stay untracked.
    assert not any(name.startswith("vision_encoder_layers.embeddings") for name, _ in tracker.params)


def test_drift_tracker_tracks_the_mixer_stack_and_gates():
    from train_cvlm import VitDriftTracker

    class StubMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer_layers = torch.nn.ModuleList([torch.nn.Linear(3, 3, bias=False)])
            self.mixer_gates = torch.nn.Parameter(torch.zeros(1))
            self.vision_encoder = None

    tracker = VitDriftTracker(StubMixer())
    assert tracker.branch == "mixer"
    assert tracker.numel == 9 + 1
    assert {name for name, _ in tracker.params} == {"mixer_layers.0.weight", "mixer_gates"}


def test_drift_tracker_disabled_without_trainable_vit():
    from train_cvlm import VitDriftTracker

    class StubNoVit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer_layers = None
            self.mixer_gates = None
            self.vision_encoder = None

    assert not VitDriftTracker(StubNoVit()).enabled

    frozen = _StubVit(layers=1, width=4)
    for parameter in frozen.parameters():
        parameter.requires_grad = False
    assert not VitDriftTracker(frozen).enabled


def test_bitwise_comparison_is_about_bits_not_float_equality():
    from train_cvlm import _bitwise_unchanged

    a = torch.tensor([0.0, -0.0, 1.0], dtype=torch.bfloat16)
    b = torch.tensor([0.0, 0.0, 1.0], dtype=torch.bfloat16)
    # -0.0 == 0.0 as floats, but the stored bits differ: an update landed.
    assert _bitwise_unchanged(a, b) == 2


# --------------------------------------------------------------------------
# 5. The decisive experiment in miniature: a real bf16 ViT stack at lr=1e-4.
# --------------------------------------------------------------------------


def _run_vit_steps(fp32_master_weights: bool, steps: int = 25):
    from transformers import ViTConfig, ViTModel

    from train_cvlm import VitDriftTracker
    from train_utils import build_optimizer

    config = ViTConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        image_size=8,
        patch_size=4,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )

    class TinyCvlm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.vision_encoder = ViTModel(config)
            self.mixer_layers = None
            self.mixer_gates = None

    model = TinyCvlm()
    with torch.no_grad():
        # Pretrained ViT weights are several times larger than a config init;
        # that magnitude is exactly what makes their bf16 ULP too coarse.
        for parameter in model.vision_encoder.encoder.layer.parameters():
            parameter.mul_(0.0).add_(1.0)
    model = model.to(torch.bfloat16)

    tracker = VitDriftTracker(model)
    optimizer = build_optimizer(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4}],
        lr=1e-4,
        betas=(0.9, 0.999),
        fp32_master_weights=fp32_master_weights,
    )
    torch.manual_seed(1)
    hidden = torch.randn(2, 5, 16).to(torch.bfloat16)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model.vision_encoder.encoder(hidden).last_hidden_state
        out.square().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return tracker.measure()


def test_bf16_vit_stack_is_frozen_by_rounding_and_master_weights_unfreeze_it():
    legacy = _run_vit_steps(fp32_master_weights=False)
    mastered = _run_vit_steps(fp32_master_weights=True)

    # Legacy: every ~1.0-magnitude weight is bit-for-bit where it started.
    assert legacy["vit_frozen_frac"] == 1.0
    assert legacy["vit_drift"] == 0.0
    # With fp32 masters the accumulated updates cross the rounding threshold.
    assert mastered["vit_frozen_frac"] < 1.0
    assert mastered["vit_drift"] > 0.0


# --------------------------------------------------------------------------
# 6. CLI / env wiring.
# --------------------------------------------------------------------------


def test_cli_flag_defaults_off_and_reads_env(monkeypatch):
    import train_cvlm

    monkeypatch.delenv("FP32_MASTER_WEIGHTS", raising=False)
    monkeypatch.setattr("sys.argv", ["train_cvlm.py", "--output_dir", "/tmp/x"])
    assert train_cvlm.parse_args().fp32_master_weights is False

    monkeypatch.setattr(
        "sys.argv",
        ["train_cvlm.py", "--output_dir", "/tmp/x", "--fp32_master_weights", "true"],
    )
    assert train_cvlm.parse_args().fp32_master_weights is True

    monkeypatch.setenv("FP32_MASTER_WEIGHTS", "1")
    monkeypatch.setattr("sys.argv", ["train_cvlm.py", "--output_dir", "/tmp/x"])
    assert train_cvlm.parse_args().fp32_master_weights is True

    monkeypatch.setattr(
        "sys.argv",
        ["train_cvlm.py", "--output_dir", "/tmp/x", "--fp32_master_weights", "0"],
    )
    assert train_cvlm.parse_args().fp32_master_weights is False


def test_cli_scope_defaults_to_all_and_reads_env(monkeypatch):
    import train_cvlm

    monkeypatch.delenv("FP32_MASTER_SCOPE", raising=False)
    monkeypatch.setattr("sys.argv", ["train_cvlm.py", "--output_dir", "/tmp/x"])
    assert train_cvlm.parse_args().fp32_master_scope == "all"

    monkeypatch.setenv("FP32_MASTER_SCOPE", "vision")
    monkeypatch.setattr("sys.argv", ["train_cvlm.py", "--output_dir", "/tmp/x"])
    assert train_cvlm.parse_args().fp32_master_scope == "vision"

    monkeypatch.setattr(
        "sys.argv",
        ["train_cvlm.py", "--output_dir", "/tmp/x", "--fp32_master_scope", "bogus"],
    )
    with pytest.raises(SystemExit):
        train_cvlm.parse_args()
