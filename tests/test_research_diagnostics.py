import json
import random
from types import SimpleNamespace

import pytest
import torch


def test_mechanistic_jsonl_writer_is_append_only(tmp_path):
    from train_logging import MechanisticJsonlWriter

    path = tmp_path / "mechanistic.jsonl"
    first = {"step": 10, "compression_rate": 4, "mixer_gates": [0.1, -0.2]}
    second = {"step": 20, "compression_rate": 8, "mixer_gates": None}

    writer = MechanisticJsonlWriter(str(path))
    writer.append(first)
    writer.close()
    writer = MechanisticJsonlWriter(str(path))
    writer.append(second)
    writer.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [first, second]


def test_paired_block_bootstrap_is_deterministic_and_paired():
    from tools.block_bootstrap_train import paired_block_bootstrap

    a = {step: step * 0.01 + 0.3 for step in range(1, 101)}
    b = {step: step * 0.01 for step in range(1, 101)}
    result_a = paired_block_bootstrap(
        a,
        b,
        lo=1,
        hi=100,
        tail=100,
        n_boot=200,
        block=7,
        rng=random.Random(123),
    )
    result_b = paired_block_bootstrap(
        a,
        b,
        lo=1,
        hi=100,
        tail=100,
        n_boot=200,
        block=7,
        rng=random.Random(123),
    )
    assert result_a == result_b
    mean, lower, upper, n = result_a
    assert n == 100
    assert abs(mean - 0.3) < 1e-12
    assert lower > 0
    assert upper > 0


def test_padding_attention_collector_observes_native_pad_mass():
    from transformers import ViTConfig, ViTModel

    from tools.measure_padding_contamination import PaddingAttentionCollector

    config = ViTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        image_size=16,
        patch_size=8,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    layer = ViTModel(config).encoder.layer[0].eval()

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer_layers = torch.nn.ModuleList([layer])
            self.vision_encoder = None
            self._current_vision_attention_mask = torch.tensor(
                [[1, 1, 1, 0, 0]],
                dtype=torch.long,
            )

    model = Stub()
    collector = PaddingAttentionCollector(model, max_batches=1, query_chunk=2)
    hidden = torch.randn(1, 5, 32)
    native = layer(hidden)
    measured = collector.means(mask_safe=False)

    assert native.shape == hidden.shape
    assert measured[0] is not None
    assert 0.0 < measured[0] < 1.0
    attention = layer.attention.attention
    normalized = layer.layernorm_before(hidden)
    heads = attention.num_attention_heads
    head_dim = attention.attention_head_size
    shape = (1, -1, heads, head_dim)
    query = attention.query(normalized).view(*shape).transpose(1, 2)
    key = attention.key(normalized).view(*shape).transpose(1, 2)
    scaling = float(getattr(attention, "scaling", head_dim**-0.5))
    probabilities = torch.softmax(
        torch.matmul(query, key.transpose(-1, -2)) * scaling,
        dim=-1,
        dtype=torch.float32,
    )
    expected = probabilities[:, :, :3, 3:].sum(dim=-1).mean()
    assert abs(measured[0] - float(expected.detach())) < 1e-7

    # max_batches=1 means a second forward does not silently reweight the
    # diagnostic toward whichever condition happens to have more batches.
    layer(torch.randn_like(hidden))
    assert collector.means(mask_safe=False) == measured
    assert collector.means(mask_safe=True) == [0.0]
    collector.close()


def test_initialization_artifacts_are_exact_and_resume_checked(tmp_path):
    from safetensors.torch import load_file

    from train_cvlm import _write_or_verify_initialization_artifacts

    class TinyCompressionPath(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_projector = torch.nn.Linear(4, 4)
            self.vision_projector = torch.nn.Linear(4, 4)
            self.vision_pos_embed = torch.nn.Embedding(8, 4)
            self.pool_queries = torch.nn.Parameter(torch.zeros(2, 4))
            self.mixer_layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.mixer_gates = torch.nn.Parameter(torch.zeros(1))
            self.vision_encoder = None
            config = SimpleNamespace(_commit_hash="fixed-revision")
            self.decoder = SimpleNamespace(config=config)
            self.text_encoder = SimpleNamespace(config=config)
            self.model_args = SimpleNamespace(
                model_name_or_path="decoder",
                text_encoder_name="encoder",
                vision_encoder_name="vision",
            )

    args = SimpleNamespace(
        output_dir=str(tmp_path),
        save_mechanistic_init=True,
        seed=42,
        vit_init="pretrained",
        freeze_vit=False,
        no_vit=False,
        mixer_depth_map="1:1",
        mask_vit_padding=True,
        vit_apply_final_layernorm=True,
        compact_decoder_positions=False,
        num_pool_latents=2,
        model_name_or_path="decoder",
        text_encoder_name="encoder",
        vision_encoder_name="vision",
    )
    model = TinyCompressionPath()
    _write_or_verify_initialization_artifacts(model, args)
    manifest = json.loads(
        (tmp_path / "initialization_manifest.json").read_text(encoding="utf-8")
    )
    initial = load_file(str(tmp_path / "mechanistic_init.safetensors"))

    assert manifest["branch_kind"] == "zero_gated_mixer"
    assert len(manifest["hashes"]["post_bottleneck_layers"]) == 1
    assert torch.equal(
        initial["mixer_layers.0.weight"],
        model.mixer_layers[0].weight.detach(),
    )

    # An exact reconstruction verifies and preserves the original files.
    _write_or_verify_initialization_artifacts(model, args)
    with torch.no_grad():
        model.text_projector.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="Initialization mismatch"):
        _write_or_verify_initialization_artifacts(model, args)


def test_mechanistic_probe_is_cached_and_data_checked(tmp_path):
    from train_cvlm import _save_mechanistic_probe

    dataset = [
        {"source_ids": torch.tensor([index, index + 1])}
        for index in range(4)
    ]

    def collate(samples):
        source_ids = torch.stack([sample["source_ids"] for sample in samples])
        return {
            "source_ids": source_ids,
            "source_attention_mask": torch.ones_like(source_ids),
        }

    metadata = {"dataset": "fixed", "split": "train"}
    _save_mechanistic_probe(dataset, collate, str(tmp_path), 3, metadata)
    _save_mechanistic_probe(dataset, collate, str(tmp_path), 3, metadata)

    payload = torch.load(
        tmp_path / "mechanistic_probe.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["dataset_indices"] == [0, 1, 2]
    assert payload["batch"]["source_ids"].shape == (3, 2)

    dataset[0]["source_ids"][0] = 999
    with pytest.raises(RuntimeError, match="Mechanistic probe mismatch"):
        _save_mechanistic_probe(dataset, collate, str(tmp_path), 3, metadata)


def test_partial_checkpoint_validation_rejects_architecture_mismatch():
    from train_utils import validate_partial_checkpoint_load

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    with pytest.raises(RuntimeError, match="unexpected keys"):
        validate_partial_checkpoint_load(
            model,
            missing=[],
            unexpected=["vision_encoder.encoder.layer.0.weight"],
            checkpoint_path="wrong.safetensors",
        )
    with pytest.raises(RuntimeError, match="missing trainable keys"):
        validate_partial_checkpoint_load(
            model,
            missing=["0.weight"],
            unexpected=[],
            checkpoint_path="incomplete.safetensors",
        )
    validate_partial_checkpoint_load(
        model,
        missing=["0.weight"],
        unexpected=[],
        checkpoint_path="legacy.safetensors",
        allowed_missing_trainable={"0.weight"},
    )


def test_compact_position_ids_ignore_internal_masked_slots():
    from modeling import _compact_position_ids

    mask = torch.tensor(
        [
            [0, 0, 1, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    positions = _compact_position_ids(mask)
    assert positions.tolist() == [
        [1, 1, 0, 1, 1, 1, 2, 3],
        [0, 1, 2, 1, 3, 1, 1, 1],
    ]


def test_compact_positions_make_llama_invariant_to_masked_holes():
    from transformers import LlamaConfig, LlamaModel

    from modeling import _compact_position_ids

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    torch.manual_seed(1234)
    model = LlamaModel(config).eval()
    prefix = torch.randn(1, 4, 32)
    answer = torch.randn(1, 2, 32)
    holes = torch.randn(1, 3, 32)
    base = torch.cat([prefix, answer], dim=1)
    padded = torch.cat([prefix, holes, answer], dim=1)
    base_mask = torch.ones(1, 6, dtype=torch.long)
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1, 1]])

    with torch.no_grad():
        compact_base = model(
            inputs_embeds=base,
            attention_mask=base_mask,
            position_ids=_compact_position_ids(base_mask),
        ).last_hidden_state
        compact_padded = model(
            inputs_embeds=padded,
            attention_mask=padded_mask,
            position_ids=_compact_position_ids(padded_mask),
        ).last_hidden_state
        legacy_base = model(
            inputs_embeds=base,
            attention_mask=base_mask,
        ).last_hidden_state
        legacy_padded = model(
            inputs_embeds=padded,
            attention_mask=padded_mask,
        ).last_hidden_state

    assert torch.allclose(
        compact_base[:, -2:],
        compact_padded[:, -2:],
        atol=2e-5,
        rtol=2e-5,
    )
    assert not torch.allclose(
        legacy_base[:, -2:],
        legacy_padded[:, -2:],
        atol=1e-5,
        rtol=1e-5,
    )
