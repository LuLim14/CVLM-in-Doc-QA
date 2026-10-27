"""Tests for the optional top-K text-encoder unfreeze.

Avoids constructing a full CVLM (would download HF weights). Drives the
unfreeze logic against a lightweight stand-in encoder that mimics the
ModernBERT structure (.layers list + .final_norm) the production code expects.
"""
import pytest
import torch
import torch.nn as nn


def _make_dummy_cvlm(n_layers: int = 6, top_k: int = 0):
    """Build a minimal object with the attributes _freeze_frozen_submodules
    touches: model_args.unfreeze_encoder_top_k and a fake text_encoder with
    .layers (ModuleList) + .final_norm (a Module)."""
    from modeling import CVLM

    class _Args:
        pass

    encoder = nn.Module()
    encoder.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(n_layers)])
    encoder.final_norm = nn.LayerNorm(8)
    # Stub eval/train mode hooks (nn.Module already supplies them).

    dummy = type("D", (), {})()
    dummy.text_encoder = encoder
    dummy.model_args = _Args()
    dummy.model_args.unfreeze_encoder_top_k = int(top_k)
    return dummy, CVLM._freeze_frozen_submodules


def test_unfreeze_zero_keeps_encoder_fully_frozen():
    dummy, freeze_fn = _make_dummy_cvlm(n_layers=6, top_k=0)
    freeze_fn(dummy)
    assert all(not p.requires_grad for p in dummy.text_encoder.parameters())
    # Eval mode (legacy behaviour).
    assert not dummy.text_encoder.training


def test_unfreeze_top_k_marks_correct_layers():
    dummy, freeze_fn = _make_dummy_cvlm(n_layers=6, top_k=2)
    freeze_fn(dummy)
    layers = dummy.text_encoder.layers
    # bottom (6 - 2 = 4) layers frozen
    for i in range(4):
        assert all(not p.requires_grad for p in layers[i].parameters()), \
            f"layer {i} should be frozen"
    # top 2 layers unfrozen
    for i in range(4, 6):
        assert all(p.requires_grad for p in layers[i].parameters()), \
            f"layer {i} should be unfrozen"
    # final_norm unfrozen
    assert all(p.requires_grad for p in dummy.text_encoder.final_norm.parameters())
    # train mode (so dropout in unfrozen blocks fires)
    assert dummy.text_encoder.training


def test_unfreeze_top_k_clamped_to_total_layers():
    dummy, freeze_fn = _make_dummy_cvlm(n_layers=4, top_k=999)
    freeze_fn(dummy)
    # All 4 layers + final_norm unfrozen.
    for layer in dummy.text_encoder.layers:
        assert all(p.requires_grad for p in layer.parameters())
    assert all(p.requires_grad for p in dummy.text_encoder.final_norm.parameters())


def test_unfreeze_negative_or_zero_treated_as_off():
    for k in (0, -1, -10):
        dummy, freeze_fn = _make_dummy_cvlm(n_layers=4, top_k=k)
        freeze_fn(dummy)
        assert all(not p.requires_grad for p in dummy.text_encoder.parameters()), \
            f"top_k={k} must keep encoder frozen"


def test_freeze_is_idempotent():
    dummy, freeze_fn = _make_dummy_cvlm(n_layers=4, top_k=2)
    freeze_fn(dummy)
    freeze_fn(dummy)  # second call must not change the layout
    layers = dummy.text_encoder.layers
    for i in range(2):
        assert all(not p.requires_grad for p in layers[i].parameters())
    for i in range(2, 4):
        assert all(p.requires_grad for p in layers[i].parameters())


def test_missing_layer_container_raises():
    """If the encoder lacks both .layers and .encoder.layer, the helper must
    fail loudly rather than silently no-op."""
    from modeling import CVLM

    class _Args:
        pass

    bare_encoder = nn.Module()
    # No layers / no encoder attribute.
    dummy = type("D", (), {})()
    dummy.text_encoder = bare_encoder
    dummy.model_args = _Args()
    dummy.model_args.unfreeze_encoder_top_k = 2

    with pytest.raises(RuntimeError, match="transformer layers"):
        CVLM._freeze_frozen_submodules(dummy)


def test_legacy_encoder_layer_naming_supported():
    """HF bert-style models nest layers at .encoder.layer — make sure the
    fallback path finds them too."""
    from modeling import CVLM

    class _Args:
        pass

    inner = nn.Module()
    inner.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    enc = nn.Module()
    enc.encoder = inner
    # No final_norm — code must tolerate its absence.
    dummy = type("D", (), {})()
    dummy.text_encoder = enc
    dummy.model_args = _Args()
    dummy.model_args.unfreeze_encoder_top_k = 1

    CVLM._freeze_frozen_submodules(dummy)
    layers = enc.encoder.layer
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert all(not p.requires_grad for p in layers[1].parameters())
    assert all(p.requires_grad for p in layers[2].parameters())
