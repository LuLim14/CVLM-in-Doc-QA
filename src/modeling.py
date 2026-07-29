import os
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers

try:
    from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
except ImportError:
    # liger-kernel needs torch>=2.5; allow running without it (set DISABLE_LIGER=1
    # and budget for unfused fp32 logits: shrink batch + raise grad_accum).
    MODEL_TYPE_TO_APPLY_LIGER_FN = None
from safetensors.torch import load_file

# Liger dispatch: pick the apply_liger_kernel_to_<model_type> matching the
# decoder being loaded. Patches the HF class globally (monkey-patches forward),
# so it must run BEFORE AutoModelForCausalLM.from_pretrained(...). The
# fused_linear_cross_entropy fuse is the critical one — it avoids materialising
# the `[B, L, vocab]` fp32 logits tensor, which on Qwen3 (vocab=151936) is
# ~3x larger than on Llama/SmolLM2 (vocab=49152) and OOMs at moderate batch.
from torch.utils.checkpoint import checkpoint as _torch_checkpoint
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, ViTConfig, ViTModel


def _apply_liger_for(model_name_or_path: str) -> None:
    if os.environ.get("DISABLE_LIGER", "0").lower() in ("1", "true", "yes"):
        print("[modeling] DISABLE_LIGER is set; decoder will run without Liger kernels.")
        return
    if MODEL_TYPE_TO_APPLY_LIGER_FN is None:
        print(
            "[modeling] WARN: liger-kernel is not installed; decoder will run "
            "unfused — expect ~3-30 GB of fp32 logits depending on (vocab, batch, seq_len)."
        )
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
    mixed_fidelity_p: float = field(
        default=0.0,
        metadata={
            "help": "MIXED-FIDELITY training: per-chunk probability that a document chunk enters the "
            "decoder as raw decoder-token embeddings instead of its single compressed vision token. "
            "Blocks are interleaved in document order. 0.0 (default) = pure compressed, legacy path. "
            "Requires the dataset/collator to supply doc_dec_ids + chunk_dec_spans + chunk_counts. "
            "Fine-tuning feature: combine with --restore_from."
        },
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
    projector_use_input_rmsnorm: bool = field(
        default=True,
        metadata={
            "help": "If False, projectors are plain 2-layer MLP (up_proj/GELU/down_proj) — matches pre-RMSNorm checkpoints with no input_norm.* keys."
        },
    )
    vit_init: str = field(
        default="pretrained",
        metadata={
            "help": "'pretrained' loads vision_encoder_name weights; 'random' builds the identical ViT architecture from its config with random init (prior ablation)."
        },
    )
    vit_freeze: bool = field(
        default=False,
        metadata={
            "help": "Freeze the vision encoder (requires_grad=False + eval mode). Gradient still flows THROUGH it to the projector/pool/encoder. Frozen ViT weights are not saved in checkpoints; eval reloads them from HF, so combining with vit_init='random' is rejected."
        },
    )
    vit_skip: bool = field(
        default=False,
        metadata={
            "help": "Bypass the ViT block entirely (no-ViT ablation): encoder -> pool -> text_projector -> +pos_embed -> vision_projector. Projector/pos_embed geometry is kept identical to the ViT branches. Must be passed to eval too, or eval silently rebuilds a pretrained ViT."
        },
    )
    mixer_depth_map: str = field(
        default="",
        metadata={
            "help": "RATE-DEPTH MIXER: comma-separated 'cr:depth' map (e.g. '1:0,2:1,4:3,8:6,16:12'). "
            "Non-empty ENABLES the mixer and fully replaces the ViT branching: instead of a ViT encoder, "
            "a stack of zero-gated ViTLayer blocks is built (config-init from vision_encoder_name) and the "
            "ACTIVE depth is looked up from the current compression_rate (exact key; else nearest LOWER key; "
            "else the smallest key). Depth 0 / zero gates are byte-identical to the no-ViT path. "
            "Incompatible with vit_skip / vit_freeze / vit_init='random'. Empty (default) = legacy ViT branching."
        },
    )
    shared_init_seed: int = field(
        default=0,
        metadata={
            "help": "If >0, re-seed torch RNG (a) with shared_init_seed+1 right before building the ViT and (b) with shared_init_seed right after — so the branch-specific ViT paths (from_pretrained / config-init / skip) consume different RNG amounts WITHOUT desynchronising the shared modules (projectors, pos_embed, pool_queries): those become bit-identical across ablation branches for a given seed."
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


def _parse_mixer_depth_map(s: Optional[str], max_depth: Optional[int] = None) -> Optional[dict]:
    """Parse a RATE-DEPTH MIXER map string "cr:depth,..." into {cr: depth}.

    Returns None for empty/None input (mixer disabled). Raises ValueError on
    malformed input: missing ':', non-integer fields, cr < 1, depth < 0,
    duplicate cr keys, or depth > max_depth (when max_depth is given —
    depth == max_depth, i.e. "use every layer", is legal).
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: dict = {}
    for tok in s.split(","):
        tok = tok.strip()
        if ":" not in tok:
            raise ValueError(f"mixer_depth_map token {tok!r} missing ':' separator")
        cr_str, depth_str = tok.split(":", 1)
        try:
            cr = int(cr_str.strip())
            depth = int(depth_str.strip())
        except ValueError:
            raise ValueError(f"mixer_depth_map token {tok!r} has non-integer fields")
        if cr < 1:
            raise ValueError(f"mixer_depth_map cr={cr} must be >= 1")
        if depth < 0:
            raise ValueError(f"mixer_depth_map depth={depth} must be >= 0")
        if max_depth is not None and depth > max_depth:
            raise ValueError(f"mixer_depth_map depth={depth} exceeds available mixer layers ({max_depth})")
        if cr in out:
            raise ValueError(f"mixer_depth_map has duplicate cr={cr}")
        out[cr] = depth
    return out


def _mixer_depth_for_cr(depth_map: dict, cr: int) -> int:
    """Depth lookup rule: exact key; else nearest LOWER key; else smallest key.

    The nearest-lower rule makes eval on an unseen cr conservative: e.g. with a
    map up to 16, cr=32 uses the depth trained for cr=16 (never extrapolates to
    a deeper stack than was ever trained).
    """
    cr = int(cr)
    if cr in depth_map:
        return depth_map[cr]
    lower = [k for k in depth_map if k < cr]
    if lower:
        return depth_map[max(lower)]
    return depth_map[min(depth_map)]


def _mixer_forward(
    h: torch.Tensor,
    layers,
    gates: torch.Tensor,
    depth: int,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Zero-gated interpolation stack over the first ``depth`` ViTLayer blocks.

    Each step is ``h <- h + g_i * (Layer_i(h) - h)``. ViTLayer already contains
    its internal residual connections, so the gate interpolates between the
    identity (g=0) and the full layer output (g=1): at zero gates — and at
    depth=0, where ``h`` is returned untouched — the path is byte-identical to
    the no-ViT path (vit_output = vit_input, no final layernorm).

    ``layers[i](h)`` returns a bare Tensor on transformers >= 4.56 and a
    ``(hidden, ...)`` tuple on older versions; both are handled — blindly
    indexing ``[0]`` on a Tensor would silently slice the batch dimension.
    """
    for i in range(depth):
        layer = layers[i]
        if use_checkpoint:
            out = _torch_checkpoint(layer, h, use_reentrant=False)
        else:
            out = layer(h)
        if isinstance(out, tuple):
            out = out[0]
        h = h + gates[i].to(h.dtype) * (out - h)
    return h


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


def _sample_raw_chunk_flags(
    chunk_lens: list[int],
    p: float,
    budget: int,
    rng: Optional[random.Random] = None,
) -> list[bool]:
    """Sample which chunks are fed RAW (decoder tokens) vs COMPRESSED (1 vision token).

    Each chunk is independently raw with probability ``p`` (fraction ~p in
    expectation). A raw chunk costs ``chunk_lens[i]`` decoder tokens; a
    compressed chunk costs exactly 1. If the total cost exceeds ``budget``,
    raw chunks are demoted back to compressed in random order until the cost
    fits (unbiased trim — no position preference). Chunks with an empty
    decoder span (len 0) are never raw: they must keep their vision token so
    the chunk stays represented in the sequence.

    Args:
        chunk_lens: decoder-token length of each chunk's raw span, doc order.
        p: raw probability in [0, 1].
        budget: max total decoder positions for the mixed block.
        rng: optional random.Random for determinism (tests); default global RNG.

    Returns:
        List of booleans, one per chunk, in document order. True = raw.
    """
    r: random.Random = rng if rng is not None else random  # type: ignore[assignment]
    n = len(chunk_lens)
    flags = [(chunk_lens[i] > 0) and (r.random() < p) for i in range(n)]
    cost = sum(chunk_lens[i] if flags[i] else 1 for i in range(n))
    if cost > budget:
        raw_idx = [i for i in range(n) if flags[i]]
        r.shuffle(raw_idx)
        while cost > budget and raw_idx:
            i = raw_idx.pop()
            flags[i] = False
            cost -= chunk_lens[i] - 1
    return flags


def _assemble_mixed_decoder_input(
    prompt_embs: torch.Tensor,
    prompt_mask: torch.Tensor,
    vision_embeds: torch.Tensor,
    vision_mask: torch.Tensor,
    doc_embs: torch.Tensor,
    chunk_spans: torch.Tensor,
    chunk_counts,
    raw_flags: list[list[bool]],
    answer_embs: torch.Tensor,
    answer_mask: torch.Tensor,
    labels_src: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Build the MIXED-FIDELITY decoder input: [prompt | mixed blocks | answer].

    The mixed segment is a per-sample concatenation of blocks in DOCUMENT
    ORDER: chunk j contributes either its raw decoder-token embeddings
    ``doc_embs[b, lo_j:hi_j]`` (raw_flags[b][j] True and span non-empty) or its
    single compressed vision embedding ``vision_embeds[b, j]``. Variable mixed
    lengths are right-padded with zero embeddings and mask 0 (same convention
    as the vision block in the legacy path). Labels are -100 over
    prompt+mixed; the answer part carries ``labels_src`` (loss on answer only).

    Args:
        prompt_embs/prompt_mask: [B, P, D] / [B, P] (left-padded prompt).
        vision_embeds/vision_mask: [B, V, D] / [B, V] from _encode_vision.
        doc_embs: [B, T_doc, D] decoder embeddings of the whole document.
        chunk_spans: [B, C, 2] (start, end) decoder-token spans per chunk.
        chunk_counts: [B] number of valid chunks per sample (int or tensor).
        raw_flags: per-sample list of booleans, document order.
        answer_embs/answer_mask: [B, A, D] / [B, A] (right-padded answer).
        labels_src: [B, A] answer labels (-100 on answer pad).

    Returns:
        (decoder_input [B, P+M+A, D], attention_mask [B, P+M+A],
         labels [B, P+M+A], stats) where stats has per-sample python lists:
         "mixed_lens" (real mixed-block length), "raw_counts", "n_chunks".
    """
    B, P, D = prompt_embs.shape
    blocks_per_sample: list[torch.Tensor] = []
    mixed_lens: list[int] = []
    raw_counts: list[int] = []
    n_chunks: list[int] = []
    for b in range(B):
        n = min(int(vision_mask[b].sum()), int(chunk_counts[b]))
        blocks: list[torch.Tensor] = []
        n_raw = 0
        for j in range(n):
            lo = int(chunk_spans[b, j, 0])
            hi = int(chunk_spans[b, j, 1])
            if j < len(raw_flags[b]) and raw_flags[b][j] and hi > lo:
                blocks.append(doc_embs[b, lo:hi])
                n_raw += 1
            else:
                blocks.append(vision_embeds[b, j : j + 1])
        mixed = torch.cat(blocks, dim=0) if blocks else vision_embeds.new_zeros((0, D))
        blocks_per_sample.append(mixed)
        mixed_lens.append(int(mixed.size(0)))
        raw_counts.append(n_raw)
        n_chunks.append(n)

    M_max = max(max(mixed_lens), 1)
    mixed_embs = vision_embeds.new_zeros((B, M_max, D))
    mixed_mask = torch.zeros((B, M_max), dtype=torch.long, device=vision_embeds.device)
    for b, m in enumerate(blocks_per_sample):
        if m.size(0) > 0:
            mixed_embs[b, : m.size(0)] = m
            mixed_mask[b, : m.size(0)] = 1

    decoder_input = torch.cat([prompt_embs, mixed_embs, answer_embs], dim=1)
    attention_mask = torch.cat([prompt_mask, mixed_mask, answer_mask], dim=1)
    ignore = torch.full((B, P + M_max), -100, dtype=labels_src.dtype, device=labels_src.device)
    labels = torch.cat([ignore, labels_src], dim=1)
    stats = {"mixed_lens": mixed_lens, "raw_counts": raw_counts, "n_chunks": n_chunks}
    return decoder_input, attention_mask, labels, stats


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

        vit_init = str(getattr(model_args, "vit_init", "pretrained") or "pretrained")
        if vit_init not in ("pretrained", "random"):
            raise ValueError(f"vit_init must be 'pretrained' or 'random', got {vit_init!r}")
        # RATE-DEPTH MIXER: non-empty map fully replaces the ViT branching below.
        mixer_map = _parse_mixer_depth_map(getattr(model_args, "mixer_depth_map", "") or "")
        if mixer_map is not None:
            if getattr(model_args, "vit_skip", False):
                raise ValueError(
                    "mixer_depth_map is incompatible with vit_skip: the mixer replaces the ViT "
                    "branching entirely (depth 0 already IS the no-ViT path)."
                )
            if getattr(model_args, "vit_freeze", False):
                raise ValueError(
                    "mixer_depth_map is incompatible with vit_freeze: mixer layers are always "
                    "trainable (frozen params are excluded from checkpoints and would be lost)."
                )
            if vit_init == "random":
                raise ValueError(
                    "mixer_depth_map is incompatible with vit_init='random': the mixer is always "
                    "config-initialised (random) by construction; pass the default vit_init."
                )
        self.mixer_depth_map = None
        self.mixer_layers = None
        self.mixer_gates = None
        self._mixer_grad_ckpt = False
        shared_seed = int(getattr(model_args, "shared_init_seed", 0) or 0)
        if shared_seed:
            # Reproducible branch-specific ViT init (matters for vit_init='random').
            torch.manual_seed(shared_seed + 1)
        if vit_init == "random" and getattr(model_args, "vit_freeze", False):
            raise ValueError(
                "vit_init='random' + vit_freeze is unsupported: frozen params are not "
                "saved in checkpoints, so the random weights would be lost for eval."
            )
        if mixer_map is not None:
            # RATE-DEPTH MIXER: built HERE — after manual_seed(shared_seed + 1) —
            # so mixer weights are reproducible across runs like the other ViT
            # branches. Config-init only (no pretrained weights); embeddings and
            # pooler of the throwaway ViTModel are NOT kept, only the layer stack.
            vit_cfg = ViTConfig.from_pretrained(model_args.vision_encoder_name)
            _vit_tmp = ViTModel(vit_cfg)
            self.mixer_layers = nn.ModuleList(list(_vit_tmp.encoder.layer))
            del _vit_tmp
            bad = {k: v for k, v in mixer_map.items() if v > len(self.mixer_layers)}
            if bad:
                raise ValueError(
                    f"mixer_depth_map depths {bad} exceed available mixer layers ({len(self.mixer_layers)})"
                )
            self.mixer_gates = nn.Parameter(torch.zeros(len(self.mixer_layers)))
            self.mixer_depth_map = dict(sorted(mixer_map.items()))
            self.vision_encoder = None
            vit_hidden = vit_cfg.hidden_size
            n_mixer = sum(p.numel() for p in self.mixer_layers.parameters()) + self.mixer_gates.numel()
            print(
                f"[modeling] RATE-DEPTH MIXER enabled: depth_map={self.mixer_depth_map} "
                f"layers={len(self.mixer_layers)} params={n_mixer} "
                f"(zero-gated; depth 0 == no-ViT path byte-for-byte)"
            )
        elif getattr(model_args, "vit_skip", False):
            # no-ViT ablation: keep projector/pos_embed geometry via the ViT config,
            # but never instantiate the encoder itself.
            self.vision_encoder = None
            vit_hidden = AutoConfig.from_pretrained(model_args.vision_encoder_name).hidden_size
            print("[modeling] vit_skip: ViT block is BYPASSED (encoder->pool->projectors only)")
        elif vit_init == "random":
            vit_cfg = AutoConfig.from_pretrained(model_args.vision_encoder_name)
            self.vision_encoder = ViTModel(vit_cfg)
            vit_hidden = self.vision_encoder.config.hidden_size
            print(f"[modeling] vit_init=random: ViT built from config ({model_args.vision_encoder_name}), no pretrained weights")
        else:
            self.vision_encoder = ViTModel.from_pretrained(
                model_args.vision_encoder_name,
                use_safetensors=True,
            )
            vit_hidden = self.vision_encoder.config.hidden_size
        if shared_seed:
            # CRITICAL for ablation fairness: the three ViT paths above consume
            # different amounts of global RNG. Re-seed so every shared module
            # below starts bit-identical across branches.
            torch.manual_seed(shared_seed)
        use_prms = bool(getattr(model_args, "projector_use_input_rmsnorm", True))
        self.text_projector = Projector(
            enc_hidden,
            vit_hidden,
            use_input_rmsnorm=use_prms,
        )
        self.vision_projector = Projector(
            vit_hidden,
            self.decoder.config.hidden_size,
            use_input_rmsnorm=use_prms,
        )
        # Learned positional embedding for the compressed vision sequence.
        self.max_vision_len = int(model_args.max_vision_len)
        self.compression_rate = int(model_args.compression_rate)
        # MIXED-FIDELITY: per-chunk probability of feeding raw decoder tokens
        # instead of the compressed vision token (0.0 = legacy pure-compressed).
        self.mixed_fidelity_p = float(getattr(model_args, "mixed_fidelity_p", 0.0) or 0.0)
        if not (0.0 <= self.mixed_fidelity_p <= 1.0):
            raise ValueError(f"mixed_fidelity_p must be in [0, 1], got {self.mixed_fidelity_p}")
        if self.mixed_fidelity_p > 0.0:
            print(f"[modeling] MIXED-FIDELITY enabled: p={self.mixed_fidelity_p} (mixed block budget = 2*max_vision_len = {2 * self.max_vision_len})")
        self.vision_pos_embed = nn.Embedding(self.max_vision_len, vit_hidden)
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
            if self.vision_encoder is not None:
                self.vision_encoder.to(torch.bfloat16)
            if self.mixer_layers is not None:
                # Gates stay fp32 (like pool_queries) and are cast at use site.
                self.mixer_layers.to(torch.bfloat16)
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
        if getattr(self.model_args, "vit_freeze", False) and self.vision_encoder is not None:
            # Frozen-prior ablation: params get no updates, but gradient still
            # flows THROUGH the ViT to pool/projector/encoder (requires_grad=False
            # on parameters does not block input gradients). No no_grad wrapper!
            freeze_model(self.vision_encoder)
            self.vision_encoder.eval()
            print("[modeling] vit_freeze: vision encoder FROZEN (weights excluded from checkpoints)")
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

    def _mixer_depth_for(self, cr: int) -> int:
        """Active mixer depth for compression rate `cr` (see _mixer_depth_for_cr)."""
        if self.mixer_depth_map is None:
            raise RuntimeError("_mixer_depth_for called but the mixer is not enabled (empty mixer_depth_map)")
        return _mixer_depth_for_cr(self.mixer_depth_map, cr)

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
            msg = "Gradient checkpointing: ENABLED on decoder"
            if self.vision_encoder is not None:
                self.vision_encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
                msg += " + vision_encoder"
            if self.mixer_layers is not None:
                # Mixer layers are a bare ModuleList (no HF PreTrainedModel
                # wrapper), so activation checkpointing is applied manually in
                # _mixer_forward via torch.utils.checkpoint (non-reentrant).
                self._mixer_grad_ckpt = True
                msg += " + mixer_layers (manual torch.utils.checkpoint)"
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

    def _encode_vision(self, source_input_ids, source_attention_mask):
        """source text ids -> text encoder -> chunked pool -> ViT encoder (+pos) -> vision_projector."""
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
        if self.mixer_layers is not None:
            # RATE-DEPTH MIXER: zero-gated interpolation stack whose depth is a
            # function of the CURRENT compression rate. depth=0 (and zero gates)
            # is byte-identical to the no-ViT branch below: vit_output is the
            # untouched vit_input — no final layernorm.
            depth = self._mixer_depth_for(self.compression_rate)
            vit_output = _mixer_forward(
                vit_input,
                self.mixer_layers,
                self.mixer_gates,
                depth,
                use_checkpoint=self._mixer_grad_ckpt and self.training and torch.is_grad_enabled(),
            )
        elif self.vision_encoder is None:
            # no-ViT ablation: identical pipeline minus the ViT block itself.
            vit_output = vit_input
        else:
            vit_output = self.vision_encoder.encoder(vit_input).last_hidden_state  # [B, V, vit_dim]
            vit_output = self.vision_encoder.layernorm(vit_output)
        return self.vision_projector(vit_output), vision_attention_mask  # [B, V, llm_dim], [B, V]

    def forward(
        self,
        source_input_ids: torch.LongTensor = None,
        source_attention_mask: torch.LongTensor = None,
        prompt_ids: torch.LongTensor = None,
        answer_ids: torch.LongTensor = None,
        answer_labels: torch.LongTensor = None,
        prompt_mask: torch.LongTensor = None,
        answer_mask: torch.LongTensor = None,
        doc_dec_ids: torch.LongTensor = None,
        chunk_dec_spans: torch.LongTensor = None,
        chunk_counts: torch.LongTensor = None,
    ):
        batch_size = prompt_ids.size(0)

        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        # [B, V, llm_dim], [B, V]

        embed_layer = self.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)  # [B, P, llm_dim]
        answer_embs = embed_layer(answer_ids)  # [B, A, llm_dim]

        if prompt_mask is None or answer_mask is None:
            raise ValueError("prompt_mask and answer_mask are required in forward().")
        labels_src = answer_labels if answer_labels is not None else answer_ids

        mixed_stats = None
        use_mixed = self.mixed_fidelity_p > 0.0 and doc_dec_ids is not None
        if use_mixed:
            # MIXED-FIDELITY: interleave raw decoder-token chunks with
            # compressed vision-token chunks IN DOCUMENT ORDER.
            if chunk_dec_spans is None or chunk_counts is None:
                raise ValueError(
                    "mixed-fidelity forward requires chunk_dec_spans and chunk_counts alongside doc_dec_ids "
                    "(enable mixed_fidelity_p in CvlmTrainDataset)."
                )
            doc_embs = embed_layer(doc_dec_ids)  # [B, T_doc, llm_dim]
            # Budget: a raw chunk costs ~cr decoder tokens instead of 1 vision
            # token. Cap the mixed block at 2*max_vision_len; prompt/answer are
            # already length-filtered by the dataset, so the full sequence stays
            # <= max_prompt_len + 2*max_vision_len + max_answer_len.
            budget = 2 * self.max_vision_len
            raw_flags: list[list[bool]] = []
            for b in range(batch_size):
                n = min(int(vision_mask[b].sum()), int(chunk_counts[b]))
                lens = [
                    max(int(chunk_dec_spans[b, j, 1]) - int(chunk_dec_spans[b, j, 0]), 0)
                    for j in range(n)
                ]
                raw_flags.append(_sample_raw_chunk_flags(lens, self.mixed_fidelity_p, budget))
            decoder_input, attention_mask, labels, stats = _assemble_mixed_decoder_input(
                prompt_embs,
                prompt_mask,
                vision_embeds,
                vision_mask,
                doc_embs,
                chunk_dec_spans,
                chunk_counts,
                raw_flags,
                answer_embs,
                answer_mask,
                labels_src,
            )
            total_chunks = max(sum(stats["n_chunks"]), 1)
            mixed_stats = {
                "raw_frac": float(sum(stats["raw_counts"]) / total_chunks),
                "mixed_len": float(sum(stats["mixed_lens"]) / max(len(stats["mixed_lens"]), 1)),
                "input_len": float(attention_mask.sum(dim=1).float().mean()),
            }
        else:
            decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

            # Combined attention mask: prompt_mask | vision_mask | answer_mask.
            # (Collator provides the three sub-masks so we don't need to reconstruct them.)
            attention_mask = torch.cat([prompt_mask, vision_mask, answer_mask], dim=1)

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
            if use_mixed:
                # DDP safety (find_unused_parameters=False in train_cvlm): if
                # every chunk in the batch happened to be sampled raw, the
                # vision path would contribute no grads and the reducer would
                # error. A zero-weighted sum keeps it in the graph at no cost.
                loss = loss + vision_embeds.float().sum() * 0.0
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
        out = {
            "loss": loss,
            "logits": logits,
            "vision_mask": vision_mask,
            "attention_mask": attention_mask,
        }
        if mixed_stats is not None:
            out["mixed_stats"] = mixed_stats
        return out

    @torch.no_grad()
    def generate(
        self,
        source_input_ids,
        source_attention_mask,
        prompt_ids,
        prompt_mask,
        max_new_tokens=512,
        temperature=0.0,
    ):
        vision_embeds, vision_mask = self._encode_vision(source_input_ids, source_attention_mask)
        prompt_embs = self.decoder.get_input_embeddings()(prompt_ids)
        decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, vision_mask], dim=1)

        gen_kwargs = dict(
            inputs_embeds=decoder_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature == 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        return self.decoder.generate(**gen_kwargs)
