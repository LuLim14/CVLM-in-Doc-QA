import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from safetensors.torch import load_file

# Liger dispatch: pick the apply_liger_kernel_to_<model_type> matching the
# decoder being loaded. Patches the HF class globally (monkey-patches forward),
# so it must run BEFORE AutoModelForCausalLM.from_pretrained(...). The
# fused_linear_cross_entropy fuse is the critical one — it avoids materialising
# the `[B, L, vocab]` fp32 logits tensor, which on Qwen3 (vocab=151936) is
# ~3x larger than on Llama/SmolLM2 (vocab=49152) and OOMs at moderate batch.
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, ViTModel


def _apply_liger_for(model_name_or_path: str) -> None:
    if os.environ.get("DISABLE_LIGER", "0").lower() in ("1", "true", "yes"):
        print("[modeling] DISABLE_LIGER is set; decoder will run without Liger kernels.")
        return
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "")
    fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(mt)
    if fn is None:
        print(
            f"[modeling] WARN: no liger patch for model_type={mt!r}; "
            f"decoder will run unfused — expect ~3-30 GB of fp32 logits "
            f"depending on (vocab, batch, seq_len)."
        )
        return
    fn(rope=True, rms_norm=True, swiglu=True, fused_linear_cross_entropy=True, cross_entropy=False)
    print(f"[modeling] applied {fn.__name__} (model_type={mt})")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="HuggingFaceTB/SmolLM-135M-Instruct")
    vision_encoder_name: str = field(default="google/vit-base-patch16-224")
    text_encoder_name: str = field(
        default="answerdotai/ModernBERT-base",
        metadata={"help": "HF name of the frozen text encoder that turns source tokens into the vision sequence."},
    )
    embed_input_dim: int = field(
        default=768, metadata={"help": "Hidden size of the text encoder output (ModernBERT-base=768)."}
    )
    max_vision_len: int = field(
        default=512, metadata={"help": "Max compressed-vision-sequence length; sizes the learned positional embedding."}
    )
    compression_rate: int = field(
        default=4, metadata={"help": "Average-pool this many source tokens into one vision token."}
    )
    unfreeze_encoder_top_k: int = field(
        default=0,
        metadata={
            "help": "If >0, unfreeze the top-K transformer layers of the text encoder (plus final_norm) and allow gradient flow through them. 0 = legacy fully-frozen behaviour."
        },
    )
    num_pool_latents: int = field(
        default=1,
        metadata={
            "help": "Number of learnable latent queries (K) used by the chunked attention pool. K=1 = single-query (legacy). K>1 = Perceiver-style multi-query pool whose K outputs are mean-pooled to produce 1 vision token per chunk."
        },
    )
    vision_backend: str = field(
        default="vit",
        metadata={
            "help": "Ablation switch for the vision stage. 'vit' = pretrained ViT (default). "
            "'random_vit' = same ViT architecture with randomly-initialised weights (isolates "
            "the value of the pretrained visual prior). 'none' = drop the ViT transformer+layernorm "
            "entirely; pool -> text_projector -> +pos -> vision_projector -> decoder (isolates the "
            "value of the ViT's transformer computation)."
        },
    )
    random_vit_seed: int = field(
        default=0,
        metadata={"help": "Seed for the explicit random re-initialisation when vision_backend='random_vit'."},
    )
    projector_use_input_rmsnorm: bool = field(
        default=True,
        metadata={
            "help": "If False, projectors are plain 2-layer MLP (up_proj/GELU/down_proj) — matches pre-RMSNorm checkpoints with no input_norm.* keys."
        },
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    restore_from: str = field(
        default="",
        metadata={"help": "Safetensors checkpoint to preload trainable sub-modules from."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Enable bfloat16 precision for the model"},
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    pct = 100 * trainable_parameters / all_param if all_param else 0.0
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {pct:.4f}")


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class RMSNorm(nn.Module):
    """RMS normalization over the last dimension (weight-only)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * inv_rms).to(dtype) * self.weight


class Projector(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_input_rmsnorm: bool = True):
        super().__init__()
        self.input_norm = RMSNorm(input_dim) if use_input_rmsnorm else None
        self.up_proj = torch.nn.Linear(in_features=input_dim, out_features=4 * input_dim)
        self.down_proj = torch.nn.Linear(in_features=4 * input_dim, out_features=output_dim)
        self.gelu = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_norm is not None:
            x = self.input_norm(x)
        x = self.up_proj(x)
        x = self.gelu(x)
        x = self.down_proj(x)
        return x


def _chunked_attention_pool(
    hidden: torch.Tensor,
    source_attention_mask: torch.Tensor,
    compression_rate: int,
    max_vision_len: int,
    pool_queries: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool contiguous chunks of `compression_rate` real tokens per sample.

    When ``pool_queries`` is None, falls back to plain unweighted mean-pool
    (legacy behaviour). When ``pool_queries`` is provided, each chunk is
    reduced via softmax-weighted multi-query attention:

        scores  = (chunk_tokens @ Q.T) / sqrt(H)          # [cr, K]
        weights = softmax(scores, dim=cr)                 # [cr, K]
        per_q   = einsum("ck,ch->kh", weights, chunk)     # [K, H]
        v_i     = mean(per_q, dim=K)                       # [H]

    Accepted shapes for ``pool_queries``:
        [H]      -> single-query pool (K=1), bit-equivalent to legacy attention path.
        [K, H]   -> Perceiver-style multi-query pool; the K outputs are mean-pooled
                    to produce one vision token per chunk (output sequence length
                    unchanged).

    The padded tail of the last chunk (when L_real % cr != 0) is masked out
    with -inf scores along the cr axis (broadcast across K) so padded positions
    carry zero softmax weight regardless of K. With ``pool_queries`` initialised
    to zeros, all scores are zero, softmax is uniform over real tokens for every
    K, every per-query output equals the (corrected) mean-pool, and the K-mean
    is bit-identical to mean-pool. So this remains a strict drop-in
    generalisation: K=1 zero-init = legacy; K>1 zero-init = legacy.

    Args:
        hidden: [B, L, H] encoder output (any dtype).
        source_attention_mask: [B, L] long/int mask (1 for real, 0 for pad).
        compression_rate: pool size.
        max_vision_len: hard cap on output V per sample (truncates longer docs).
        pool_queries: optional [H] or [K, H] learnable parameter; None -> mean-pool.

    Returns:
        pooled: [B, V_max, H] where V_max = max V_i across batch.
        vision_attention_mask: [B, V_max] long (1 for real pooled slots, 0 for pad).
    """
    B, L, H = hidden.shape
    cr = max(int(compression_rate), 1)
    lens = source_attention_mask.sum(dim=1).tolist()

    V_list = []
    for l_real in lens:
        v = (l_real + cr - 1) // cr
        if v > max_vision_len:
            v = max_vision_len
        V_list.append(v)
    V_max = max(V_list) if V_list else 1
    V_max = max(V_max, 1)

    pooled = torch.zeros((B, V_max, H), dtype=hidden.dtype, device=hidden.device)
    v_mask = torch.zeros((B, V_max), dtype=torch.long, device=hidden.device)

    Q = None
    scale = 1.0
    if pool_queries is not None:
        scale = float(H) ** -0.5
        Q = pool_queries.to(dtype=hidden.dtype)
        if Q.dim() == 1:
            Q = Q.unsqueeze(0)  # [H] -> [1, H]; K=1 single-query path.
        # Q is now [K, H].

    for b in range(B):
        L_real = int(lens[b])
        V_i = int(V_list[b])
        if L_real == 0 or V_i == 0:
            continue
        # If the sample would need more than max_vision_len chunks, truncate.
        L_use = min(L_real, V_i * cr)
        h = hidden[b, :L_use]  # [L_use, H]
        pad = V_i * cr - L_use
        if pad > 0:
            h = torch.nn.functional.pad(h, (0, 0, 0, pad))  # zero-pad tail
        chunked = h.view(V_i, cr, H)  # [V_i, cr, H]

        if Q is None:
            chunk_pooled = chunked.mean(dim=1)  # [V_i, H]
            if pad > 0:
                # Last chunk's mean was divided by cr, should be divided by (cr - pad).
                true_last = cr - pad
                chunk_pooled[-1] = chunk_pooled[-1] * cr / true_last
        else:
            # scores[v, c, k] = chunked[v, c] · Q[k] / sqrt(H)
            scores = torch.einsum("vch,kh->vck", chunked, Q) * scale  # [V_i, cr, K]
            if pad > 0:
                # Mask padded tail of the LAST chunk along the cr axis (broadcasts over K).
                neg_inf = torch.finfo(scores.dtype).min
                scores[-1, cr - pad :, :] = neg_inf
            weights = torch.softmax(scores, dim=1)  # [V_i, cr, K]
            # Per-query outputs, then mean over K -> single vector per chunk.
            per_q = torch.einsum("vck,vch->vkh", weights, chunked)  # [V_i, K, H]
            chunk_pooled = per_q.mean(dim=1)  # [V_i, H]

        pooled[b, :V_i] = chunk_pooled
        v_mask[b, :V_i] = 1

    return pooled, v_mask


class CVLM(torch.nn.Module):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

        # Patch decoder class with liger kernels BEFORE instantiating.
        _apply_liger_for(self.model_name)

        # Decoder (frozen LLM)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=compute_dtype,
            use_safetensors=True,
        )

        # Text encoder (frozen; runs in forward pass instead of offline preprocess).
        self.text_encoder = AutoModel.from_pretrained(
            model_args.text_encoder_name,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            use_safetensors=True,
        )
        # Stash the encoder's hidden size for the text_projector input dim.
        enc_hidden = self.text_encoder.config.hidden_size

        # Vision backend ablation. In every case we instantiate a ViT so the
        # projector dims (text_projector out / vision_projector in) are sized by
        # its hidden_size; the backend only controls whether/how the ViT weights
        # are initialised and whether its transformer runs in _encode_vision.
        self.vision_backend = str(getattr(model_args, "vision_backend", "vit") or "vit").lower()
        if self.vision_backend not in ("vit", "random_vit", "none"):
            raise ValueError(
                f"vision_backend must be one of 'vit'/'random_vit'/'none', got {self.vision_backend!r}"
            )
        if self.vision_backend == "random_vit":
            # Same architecture, random weights: isolates the pretrained prior.
            # ViTModel(config) already runs HF's post_init -> _init_weights, but
            # we re-initialise *explicitly under a fixed local seed* so the random
            # ViT is fully reproducible and independent of construction order /
            # any prior RNG consumption. We fork the RNG so this does not perturb
            # the global stream used by the rest of the model's init.
            vit_config = AutoConfig.from_pretrained(model_args.vision_encoder_name)
            self.vision_encoder = ViTModel(vit_config)
            rvit_seed = int(getattr(model_args, "random_vit_seed", 0) or 0)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(rvit_seed)
                # Re-apply the model's own weight initialiser to every submodule.
                self.vision_encoder.apply(self.vision_encoder._init_weights)
            n_vit = sum(p.numel() for p in self.vision_encoder.parameters())
            print(
                f"[modeling] vision_backend=random_vit: ViT re-initialised from scratch "
                f"(config={model_args.vision_encoder_name}, seed={rvit_seed}, "
                f"{n_vit/1e6:.1f}M params, NO pretrained weights loaded)"
            )
        else:
            self.vision_encoder = ViTModel.from_pretrained(
                model_args.vision_encoder_name,
                use_safetensors=True,
            )
            if self.vision_backend == "none":
                print("[modeling] vision_backend=none: ViT transformer+layernorm will be BYPASSED in _encode_vision")
        use_prms = bool(getattr(model_args, "projector_use_input_rmsnorm", True))
        self.text_projector = Projector(
            enc_hidden,
            self.vision_encoder.config.hidden_size,
            use_input_rmsnorm=use_prms,
        )
        self.vision_projector = Projector(
            self.vision_encoder.config.hidden_size,
            self.decoder.config.hidden_size,
            use_input_rmsnorm=use_prms,
        )
        # Learned positional embedding for the compressed vision sequence.
        self.max_vision_len = int(model_args.max_vision_len)
        self.compression_rate = int(model_args.compression_rate)
        self.vision_pos_embed = nn.Embedding(self.max_vision_len, self.vision_encoder.config.hidden_size)
        nn.init.normal_(self.vision_pos_embed.weight, std=0.02)

        # Learnable queries for attention pooling over each chunk of
        # `compression_rate` encoder tokens. Shape [K, H]: K=1 reproduces the
        # single-query pool; K>1 is a Perceiver-style multi-query pool whose
        # K outputs are mean-pooled to produce a single vision token per chunk
        # (so downstream geometry is unchanged for any K).
        # Zero-init makes the softmax uniform over real chunk tokens at step 0,
        # which reduces _chunked_attention_pool to plain mean-pool — strict
        # drop-in generalisation. Old checkpoints lacking this parameter load
        # via strict=False and behave identically to legacy mean-pool.
        num_pool_latents = max(int(getattr(model_args, "num_pool_latents", 1) or 1), 1)
        self.pool_queries = nn.Parameter(torch.zeros(num_pool_latents, enc_hidden))

        if training_args.bf16:
            self.text_projector.to(torch.bfloat16)
            self.vision_projector.to(torch.bfloat16)
            self.vision_encoder.to(torch.bfloat16)
            self.vision_pos_embed.to(torch.bfloat16)

        self.dim = self.decoder.config.hidden_size
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            model_args.text_encoder_name, use_fast=True, trust_remote_code=True
        )

        self._init_for_training = model_args.train
        if self._init_for_training:
            self._init_training()
        else:
            # Eval: freeze + eval everything that is frozen during training too, so
            # behaviour matches regardless of how the model is constructed.
            self._freeze_frozen_submodules()

    def _freeze_frozen_submodules(self):
        freeze_model(self.text_encoder)
        self.text_encoder.eval()
        k = int(getattr(self.model_args, "unfreeze_encoder_top_k", 0) or 0)
        if k <= 0:
            return
        # Locate the transformer layer container. ModernBERT uses .layers; HF
        # bert-style models nest at .encoder.layer. Support both.
        layers = getattr(self.text_encoder, "layers", None)
        if layers is None:
            inner_enc = getattr(self.text_encoder, "encoder", None)
            layers = getattr(inner_enc, "layer", None) if inner_enc is not None else None
        if layers is None:
            raise RuntimeError(
                "Cannot locate text_encoder transformer layers for top-K unfreeze; "
                "expected .layers or .encoder.layer attribute."
            )
        L = len(layers)
        k_eff = min(k, L)
        for layer in layers[L - k_eff :]:
            for p in layer.parameters():
                p.requires_grad = True
        # Final layer norm sits on top of the stack — unfreeze with the top block
        # so its grad can update.
        for attr in ("final_norm", "norm", "LayerNorm"):
            mod = getattr(self.text_encoder, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
                break
        # Switch to train mode so dropout in the unfrozen blocks is active.
        # Frozen lower layers' params still have requires_grad=False so they
        # receive no parameter gradients.
        self.text_encoder.train()

    def set_compression_rate(self, cr: int) -> None:
        """Mutate compression_rate for subsequent forward passes.

        Safe to call at any optimizer-step boundary: _chunked_attention_pool
        reads self.compression_rate fresh each call, no graph state to migrate,
        no parameters added or removed.
        """
        cr = int(cr)
        if cr < 1:
            raise ValueError(f"compression_rate must be >= 1, got {cr}")
        self.compression_rate = cr

    def _init_training(self):
        print("Decoder is trainable; freezing the text encoder...")
        self._freeze_frozen_submodules()
        k_enc = int(getattr(self.model_args, "unfreeze_encoder_top_k", 0) or 0)
        if k_enc <= 0:
            assert all(not p.requires_grad for p in self.text_encoder.parameters()), (
                "text_encoder must be frozen when unfreeze_encoder_top_k=0"
            )
        else:
            n_train_enc = sum(p.requires_grad for p in self.text_encoder.parameters())
            n_total_enc = sum(1 for _ in self.text_encoder.parameters())
            print(
                f"text_encoder: top-{k_enc} layers + final_norm UNFROZEN "
                f"({n_train_enc}/{n_total_enc} param tensors trainable)"
            )
        print_trainable_parameters(self)

        if getattr(self.training_args, "gradient_checkpointing", False):
            # Activation checkpointing on the activation-heavy submodules.
            # use_reentrant=False is the modern PyTorch 2.x non-reentrant path:
            # supports gradient flowing through inputs (we need this — vision
            # input enters ViT from the trainable text_projector, and decoder
            # input enters from the trainable vision_projector chain) and
            # avoids the reentrant deprecation warning. No
            # enable_input_require_grads() needed because both decoder and ViT
            # have requires_grad=True params.
            self.decoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.vision_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            msg = "Gradient checkpointing: ENABLED on decoder + vision_encoder"
            if k_enc > 0 and hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                # text_encoder input is integer ids (no upstream grad); non-reentrant
                # gc handles that correctly without enable_input_require_grads.
                self.text_encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                msg += " + text_encoder"
            print(msg)

        if self.training_args.restore_from:
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            # Allow partial loads (trainable-only checkpoints).
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"  restored; missing={len(missing)} unexpected={len(unexpected)}")

    def _encode_vision(self, source_input_ids, source_attention_mask, return_intermediates=False):
        """source text ids -> text encoder -> chunked pool -> ViT encoder (+pos) -> vision_projector.

        When ``return_intermediates`` is set, the method additionally returns a dict of
        the internal representations (pooled/vit_input/vit_output/vision_embeds) for the
        mechanistic representation analysis. This flag has NO effect on the default
        (train/eval) return signature or on any tensor math.
        """
        # When the encoder is fully frozen (unfreeze_encoder_top_k=0) we wrap it
        # in no_grad so the 22-layer activation chain is not kept alive for the
        # downstream backward pass. When top-K layers are unfrozen and we are
        # training, gradient must flow back through the encoder's top block so
        # we drop the no_grad — frozen lower layers still get no parameter grad
        # (requires_grad=False); their activation memory is bounded by the
        # encoder's gradient checkpointing if enabled.
        k_enc = int(getattr(self.model_args, "unfreeze_encoder_top_k", 0) or 0)
        if k_enc > 0 and self.training:
            enc_out = self.text_encoder(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
            ).last_hidden_state  # [B, L, H_enc] in compute_dtype
        else:
            with torch.no_grad():
                enc_out = self.text_encoder(
                    input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                ).last_hidden_state  # [B, L, H_enc] in compute_dtype
            enc_out = enc_out.detach()

        pooled, vision_attention_mask = _chunked_attention_pool(
            enc_out,
            source_attention_mask,
            compression_rate=self.compression_rate,
            max_vision_len=self.max_vision_len,
            pool_queries=self.pool_queries,
        )  # [B, V_max, H_enc], [B, V_max]

        vit_input = self.text_projector(pooled)  # [B, V_max, vit_dim]
        V = vit_input.size(1)
        if V > self.max_vision_len:
            raise ValueError(
                f"Vision sequence length {V} exceeds max_vision_len={self.max_vision_len}; "
                "increase max_vision_len or compression_rate."
            )
        pos = torch.arange(V, device=vit_input.device)
        vit_input = vit_input + self.vision_pos_embed(pos).unsqueeze(0).to(vit_input.dtype)
        if self.vision_backend == "none":
            # Ablation: bypass the ViT transformer + final layernorm. The pooled,
            # projected, position-augmented sequence is fed straight to the
            # vision_projector. This keeps the exact same tensor geometry and
            # both projectors, isolating the contribution of the ViT's own
            # transformer computation.
            vit_output = vit_input
        else:
            vit_output = self.vision_encoder.encoder(vit_input).last_hidden_state  # [B, V, vit_dim]
            vit_output = self.vision_encoder.layernorm(vit_output)
        vision_embeds = self.vision_projector(vit_output)  # [B, V, llm_dim]
        if return_intermediates:
            return vision_embeds, vision_attention_mask, {
                "pooled": pooled,            # [B, V, H_enc]  pre-projection pooled text
                "vit_input": vit_input,      # [B, V, vit_dim] projected + pos (backend-independent)
                "vit_output": vit_output,    # [B, V, vit_dim] after ViT (or bypass for 'none')
                "vision_embeds": vision_embeds,  # [B, V, llm_dim] decoder-facing
            }
        return vision_embeds, vision_attention_mask  # [B, V, llm_dim], [B, V]

    def forward(
        self,
        source_input_ids: torch.LongTensor = None,
        source_attention_mask: torch.LongTensor = None,
        prompt_ids: torch.LongTensor = None,
        answer_ids: torch.LongTensor = None,
        answer_labels: torch.LongTensor = None,
        prompt_mask: torch.LongTensor = None,
        answer_mask: torch.LongTensor = None,
    ):
        batch_size = prompt_ids.size(0)

        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        # [B, V, llm_dim], [B, V]

        embed_layer = self.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)  # [B, P, llm_dim]
        answer_embs = embed_layer(answer_ids)  # [B, A, llm_dim]

        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        # Combined attention mask: prompt_mask | vision_mask | answer_mask.
        # (Collator provides the three sub-masks so we don't need to reconstruct them.)
        if prompt_mask is None or answer_mask is None:
            raise ValueError("prompt_mask and answer_mask are required in forward().")
        attention_mask = torch.cat([prompt_mask, vision_mask, answer_mask], dim=1)

        labels_src = answer_labels if answer_labels is not None else answer_ids
        P = prompt_ids.size(1)
        V = vision_embeds.size(1)
        ignore = torch.full((batch_size, P + V), -100, dtype=labels_src.dtype, device=labels_src.device)
        labels = torch.cat([ignore, labels_src], dim=1)

        if self.training:
            # Train: pass labels so Liger's fused-linear-CE fuses lm_head x
            # labels into one Triton kernel and never materialises the
            # [B, L, vocab] logits tensor. decoder_outputs.logits is None.
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_input,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = decoder_outputs.loss
            logits = decoder_outputs.logits  # None under fused-linear-CE
        else:
            # Eval: callers (eval_cvlm.py) need logits for per-token loss and
            # argmax accuracy. Skip fused-CE here; eval runs in no_grad so
            # there is no activation pressure to save.
            decoder_outputs = self.decoder(
                inputs_embeds=decoder_input,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = decoder_outputs.logits
            effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            target = labels[:, 1:].reshape(-1)
            loss = self.loss_fct(effective_logits, target)
        return {
            "loss": loss,
            "logits": logits,
            "vision_mask": vision_mask,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        source_input_ids,
        source_attention_mask,
        prompt_ids,
        prompt_mask,
        max_new_tokens=512,
        temperature=0.0,
        suffix_ids=None,
        suffix_mask=None,
    ):
        # Layout: [prompt][vision][suffix]. ``suffix`` carries the chat-template
        # tokens that must follow the compressed document at generation time —
        # the document-suffix + question + assistant-open tag — so decoding
        # begins on the answer. When absent (raw path), the sequence is just
        # [prompt][vision] and generation continues right after the vision block.
        embed_layer = self.decoder.get_input_embeddings()
        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        prompt_embs = embed_layer(prompt_ids)
        parts = [prompt_embs, vision_embeds]
        masks = [prompt_mask, vision_mask]
        if suffix_ids is not None:
            parts.append(embed_layer(suffix_ids))
            masks.append(suffix_mask)
        decoder_input = torch.cat(parts, dim=1)
        attention_mask = torch.cat(masks, dim=1)

        gen_kwargs = dict(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature == 0:
            # Greedy. Explicitly null the sampling knobs so the decoder's
            # pretrained generation_config (e.g. temperature/top_p/top_k) cannot
            # leak in and silently make decoding non-greedy. Matches the eval-mode
            # baselines' _greedy_gen_kwargs so CVLM and baselines decode identically.
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = 1
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        return self.decoder.generate(**gen_kwargs)
