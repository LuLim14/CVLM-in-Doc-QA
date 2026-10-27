"""Smoke test: a Llama-style model with HF gradient_checkpointing_enable
(use_reentrant=False) runs forward+backward and produces non-None .grad on its
parameters. Regression guard for the historical 'checkpointing silently no-ops
when no input requires grad' foot-gun (see spec Risks table).
"""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for checkpointing path")
def test_llama_gradient_checkpointing_runs_and_produces_grads():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )

    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).cuda().train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    B, L = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device="cuda")
    labels = input_ids.clone()
    labels[:, :4] = -100

    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"

    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0, "no trainable parameters"
    assert all(g is not None for g in grads), \
        "at least one trainable param has no .grad — checkpointing may be silently no-opping"
    assert all(torch.isfinite(g).all() for g in grads), \
        "non-finite gradients found"
