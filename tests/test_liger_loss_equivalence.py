"""Liger fused-linear-CE should match nn.CrossEntropyLoss on a tiny Llama in bf16."""
import pytest
import torch

try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False


@pytest.mark.skipif(not LIGER_AVAILABLE, reason="liger-kernel not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger Triton kernels require CUDA")
def test_liger_loss_matches_pytorch_ce():
    from transformers import LlamaConfig, LlamaForCausalLM

    device = torch.device("cuda")

    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )

    B, L = 2, 16
    torch.manual_seed(1234)
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    labels = input_ids.clone()
    labels[:, :4] = -100  # mimic the prompt+vision prefix mask in CVLM

    # 1. Build vanilla model FIRST (before Liger patches the class), capture loss
    torch.manual_seed(0)
    vanilla = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    state = {k: v.clone() for k, v in vanilla.state_dict().items()}
    with torch.no_grad():
        vanilla_loss = vanilla(input_ids=input_ids, labels=labels).loss.float().item()

    # 2. Patch class registry, build a fresh model, load same weights, capture loss
    apply_liger_kernel_to_llama(
        rope=True,
        rms_norm=True,
        swiglu=True,
        fused_linear_cross_entropy=True,
        cross_entropy=False,
    )
    torch.manual_seed(0)
    liger = LlamaForCausalLM(cfg).to(device=device, dtype=torch.bfloat16).eval()
    liger.load_state_dict(state)
    with torch.no_grad():
        liger_loss = liger(input_ids=input_ids, labels=labels).loss.float().item()

    # bf16-realistic tolerance per spec
    assert abs(vanilla_loss - liger_loss) < 5e-3, (
        f"loss mismatch: vanilla={vanilla_loss:.6f} liger={liger_loss:.6f}"
    )

    # Cleanup: revert the global class patch so subsequent tests in the same
    # session see vanilla HF behaviour. revert_liger_kernel_to_llama exists in
    # liger-kernel >= 0.4; if it doesn't, the test still passed — leave a note.
    try:
        from liger_kernel.transformers import revert_liger_kernel_to_llama
        revert_liger_kernel_to_llama()
    except (ImportError, AttributeError):
        pass
