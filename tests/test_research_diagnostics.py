import json
import random

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
    collector = PaddingAttentionCollector(model)
    hidden = torch.randn(1, 5, 32)
    native = layer(hidden)
    measured = collector.means(mask_safe=False)

    assert native.shape == hidden.shape
    assert measured[0] is not None
    assert 0.0 < measured[0] < 1.0
    assert collector.means(mask_safe=True) == [0.0]
    collector.close()
