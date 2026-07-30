# Full CVLM training entrypoint (reference: src/train.py — DDP, accum, cosine LR, ckpt).

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from cvlm_dataset import CvlmTrainDataset, make_collate_fn
from modeling import CVLM, ModelArguments, TrainingArguments
from train_utils import (
    AverageMeter,
    cleanup_distributed,
    find_latest_checkpoint,
    load_cvlm_checkpoint,
    save_cvlm_checkpoint,
    setup_distributed,
    unwrap_model,
)


def _parse_cr_schedule(s: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse a "cr:end_step,..." schedule string.

    Returns None for empty/None input. Raises ValueError on malformed input.
    Last entry's end_step MUST be 0 (sentinel for +inf). Non-last entries'
    end_step must be strictly increasing positive integers.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: List[Tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if ":" not in tok:
            raise ValueError(f"cr_schedule token {tok!r} missing ':' separator")
        cr_str, end_str = tok.split(":", 1)
        try:
            cr = int(cr_str.strip())
            end_step = int(end_str.strip())
        except ValueError:
            raise ValueError(f"cr_schedule token {tok!r} has non-integer fields")
        if cr < 1:
            raise ValueError(f"cr_schedule cr={cr} must be >= 1")
        if end_step < 0:
            raise ValueError(f"cr_schedule end_step={end_step} must be >= 0")
        out.append((cr, end_step))
    if not out:
        return None
    if out[-1][1] != 0:
        raise ValueError(f"cr_schedule last entry must have end_step=0 (sentinel for +inf); got {out[-1]}")
    prev_end = 0
    for i, (_, end) in enumerate(out[:-1]):
        if end <= prev_end:
            raise ValueError(
                f"cr_schedule end_step values must be strictly increasing; "
                f"entry {i} has end_step={end} <= previous {prev_end}"
            )
        prev_end = end
    return out


def _parse_mixed_cr_set(s: Optional[str]) -> Optional[List[int]]:
    """Parse a MIXED-CR set string like "1 2 4 8 16" (whitespace-separated).

    Returns None for empty/None input (mixed-cr off). Raises ValueError on
    non-integer tokens, cr < 1, or duplicate values.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    out: List[int] = []
    for tok in s.split():
        try:
            cr = int(tok)
        except ValueError:
            raise ValueError(f"mixed_cr_set token {tok!r} is not an integer")
        if cr < 1:
            raise ValueError(f"mixed_cr_set cr={cr} must be >= 1")
        if cr in out:
            raise ValueError(f"mixed_cr_set has duplicate cr={cr}")
        out.append(cr)
    return out


def _build_param_groups(model: torch.nn.Module, lr: float, gate_lr_mult: float) -> List[dict]:
    """AdamW param groups, optionally isolating the mixer gates at their own LR.

    GATE-LR CONTROL. The zero-gated mixer is a diagnostic instrument: "given the
    chance to add depth for free, does the model take it?". A near-zero gate is
    ambiguous — it can mean "the depth is useless" OR "the gate is simply slow
    to learn" (one scalar per layer, tiny gradient, shared LR with ~100M other
    params). Training a control arm with ``gate_lr_mult`` >> 1 separates the two:
    if the gates stay shut at 100x LR, slow optimisation is ruled out.

    Group layout — deliberately a function of ``gate_lr_mult`` ONLY:
      * ``gate_lr_mult == 1.0`` (default): ONE group with every trainable param
        — byte-identical to the legacy optimizer, so old ``trainer_step_*.pt``
        files still resume.
      * ``gate_lr_mult != 1.0``: TWO groups, [0] = everything except
        ``mixer_gates`` at ``lr``, [1] = ``mixer_gates`` alone at
        ``lr * gate_lr_mult``. The order is fixed (base first, gates second) so
        optimizer state maps back onto the same groups on resume.

    Resuming across a change of ``gate_lr_mult`` therefore changes the group
    count; ``load_cvlm_checkpoint`` detects that and fails with an explicit
    message instead of silently mis-assigning optimizer state.
    """
    if gate_lr_mult <= 0.0:
        raise ValueError(f"--gate_lr_mult must be > 0, got {gate_lr_mult}")
    core = unwrap_model(model)
    gate_param = getattr(core, "mixer_gates", None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if gate_lr_mult == 1.0:
        return [{"params": trainable, "lr": lr}]
    if gate_param is None:
        raise ValueError(
            "--gate_lr_mult != 1.0 requires the rate-depth mixer (--mixer_depth_map); "
            "there are no mixer_gates to give a separate learning rate to."
        )
    if not gate_param.requires_grad:
        raise ValueError("mixer_gates is frozen; --gate_lr_mult would have no effect.")
    gate_ids = {id(gate_param)}
    base_params = [p for p in trainable if id(p) not in gate_ids]
    gate_params = [p for p in trainable if id(p) in gate_ids]
    if not gate_params:
        raise ValueError("mixer_gates was not found among the trainable parameters.")
    return [
        {"params": base_params, "lr": lr},
        {"params": gate_params, "lr": lr * gate_lr_mult},
    ]


class MixedCrSampler:
    """Balanced deterministic per-optimizer-step cr sequence for MIXED-CR training.

    NOT naive per-step choice: the sequence is a concatenation of BLOCKS, each
    block a fresh ``random.Random(seed)``-shuffled permutation of the FULL cr
    set (sorted ascending before shuffling, so the sequence does not depend on
    the formatting order of --mixed_cr_set). Guarantees:
      (a) identical cr sequence across arms/runs for a given (seed, cr set);
      (b) an EQUAL number of optimizer steps per cr in any prefix whose length
          is a multiple of len(cr_set).
    Blocks are generated lazily; on resume the sequence is reconstructed from
    (seed, global_step) alone — nothing is stored in checkpoints.
    """

    def __init__(self, cr_set: List[int], seed: int) -> None:
        if not cr_set:
            raise ValueError("MixedCrSampler needs a non-empty cr set")
        self._base = sorted(int(c) for c in cr_set)
        self._rng = random.Random(int(seed))
        self._seq: List[int] = []

    def cr_for_step(self, step: int) -> int:
        """cr for the 1-indexed optimizer step ``step`` (== global_step)."""
        if step < 1:
            raise ValueError(f"optimizer step must be >= 1, got {step}")
        while len(self._seq) < step:
            block = list(self._base)
            self._rng.shuffle(block)
            self._seq.extend(block)
        return self._seq[step - 1]


class EpochRandomSampler(Sampler[int]):
    """Deterministic single-process shuffle with explicit epoch control.

    Unlike DataLoader(shuffle=True), the permutation is a pure function of
    ``seed + epoch``. A resumed run can therefore recreate the current epoch
    and skip exactly the batches already consumed without serialising an opaque
    RandomSampler generator state.
    """

    def __init__(self, data_source, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.data_source)


def _next_resume_position(local_epoch: int, local_step: int, loader_len: int) -> Tuple[int, int]:
    """Epoch/batch coordinates of the next batch after ``local_step``."""
    if local_step >= loader_len:
        return int(local_epoch) + 1, 1
    return int(local_epoch), int(local_step) + 1


def _active_cr(schedule: List[Tuple[int, int]], step: int) -> int:
    """Return the cr for the bucket containing `step`.

    The last entry's end_step=0 is treated as +inf.
    """
    for cr, end_step in schedule:
        if end_step == 0 or step < end_step:
            return cr
    raise RuntimeError(f"no active cr for step={step} in schedule={schedule}")


def _training_cr_for_step(schedule: List[Tuple[int, int]], step: int) -> int:
    """Return the cr that PRODUCED the weights at `step`.

    Differs from `_active_cr` at transition boundary steps: at end_step=N, the
    saved checkpoint contains weights trained at the *previous* stage's cr (the
    forced save fires at the end_step boundary BEFORE any training at the new
    cr). This helper returns the producing cr by using `step <= end_step`.
    Used by eval to map saved checkpoints back to the cr that trained them.
    """
    for cr, end_step in schedule:
        if end_step == 0 or step <= end_step:
            return cr
    raise RuntimeError(f"no training cr for step={step} in schedule={schedule}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CVLM (on-the-fly text-encoder variant).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="HF id for the frozen decoder (e.g. HuggingFaceTB/SmolLM-135M-Instruct).",
    )
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument(
        "--text_encoder_name",
        type=str,
        default=None,
        help="HF id for the frozen text encoder (default: answerdotai/ModernBERT-base).",
    )
    p.add_argument(
        "--compression_rate",
        type=int,
        default=4,
        help="Chunk size for mean-pool over encoder tokens into vision tokens.",
    )
    p.add_argument(
        "--unfreeze_encoder_top_k",
        type=int,
        default=int(os.environ.get("UNFREEZE_ENCODER_TOP_K", "0")),
        help="Unfreeze the top-K transformer layers of the text encoder (plus final_norm). "
        "0 (default) = encoder fully frozen, legacy behaviour.",
    )
    p.add_argument(
        "--mixed_fidelity_p",
        type=float,
        default=float(os.environ.get("MIXED_FIDELITY_P", "0.0")),
        help="MIXED-FIDELITY: per-chunk probability of feeding the decoder raw decoder-token "
        "embeddings for that chunk instead of its 1 compressed vision token, interleaved in "
        "document order. 0.0 (default) = off, legacy pure-compressed path. Fine-tuning feature: "
        "combine with --restore_from. Incompatible with --cr_schedule (chunk->decoder-token "
        "spans are precomputed in the dataset with the static --compression_rate).",
    )
    p.add_argument(
        "--mixer_depth_map",
        type=str,
        default=os.environ.get("MIXER_DEPTH_MAP", ""),
        help="RATE-DEPTH MIXER: 'cr:depth,...' map (e.g. '1:0,2:1,4:3,8:6,16:12'). Non-empty REPLACES "
        "the ViT with a stack of zero-gated ViTLayer blocks whose active depth is looked up from the "
        "current compression rate (exact key; else nearest lower key; else smallest key). Incompatible "
        "with --no_vit / --freeze_vit / --vit_init random. Must also be passed to eval. "
        "Empty (default) = legacy ViT branching. Env: MIXER_DEPTH_MAP.",
    )
    p.add_argument(
        "--mask_vit_padding",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("MASK_VIT_PADDING", "0") == "1",
        help="Apply vision_attention_mask inside every ViT/mixer self-attention layer. "
        "Enable for new controlled experiments; keep disabled only to reproduce legacy "
        "checkpoints that mixed real latents with batch padding. Must match eval. "
        "Env: MASK_VIT_PADDING.",
    )
    p.add_argument(
        "--vit_apply_final_layernorm",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("VIT_APPLY_FINAL_LAYERNORM", "1") == "1",
        help="Apply the plain ViT encoder's final LayerNorm. Set false for the fixed-depth "
        "direct control compared with the bare mixer layer stack. Must match eval. "
        "Env: VIT_APPLY_FINAL_LAYERNORM.",
    )
    p.add_argument(
        "--gate_lr_mult",
        type=float,
        default=float(os.environ.get("GATE_LR_MULT", "1.0")),
        help="RATE-DEPTH MIXER diagnostic: learning-rate multiplier for the mixer_gates "
        "parameter. 1.0 (default) = single param group, legacy behaviour. Any other value "
        "puts mixer_gates in its OWN AdamW group at lr*gate_lr_mult (group order: base, "
        "gates), which is the control that separates 'the added depth is not useful' from "
        "'the gates just learn too slowly' — e.g. --gate_lr_mult 100. Requires "
        "--mixer_depth_map. Resuming a run must reuse the same value (the group count is "
        "part of the optimizer state). Env: GATE_LR_MULT.",
    )
    p.add_argument(
        "--mixed_cr_set",
        type=str,
        default=os.environ.get("MIXED_CR_SET", ""),
        help="MIXED-CR training: whitespace-separated cr set, e.g. '1 2 4 8 16'. Non-empty samples one "
        "cr per optimizer step (all grad-accum micro-batches of a step share it) from a BALANCED "
        "deterministic sequence: concatenated random.Random(seed+30000)-shuffled permutations of the "
        "full set, indexed by global_step — equal per-cr update counts in any prefix divisible by the "
        "set size, identical across arms for a given seed. Incompatible with --cr_schedule. "
        "Empty (default) = off. Env: MIXED_CR_SET.",
    )
    p.add_argument(
        "--num_pool_latents",
        type=int,
        default=int(os.environ.get("NUM_POOL_LATENTS", "1")),
        help="Number of learnable latent queries (K) used by the chunked attention pool. "
        "K=1 (default) = single-query pool. K>1 = Perceiver-style multi-query pool whose "
        "K outputs are mean-pooled to produce 1 vision token per chunk.",
    )
    p.add_argument(
        "--legacy_projector",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("LEGACY_PROJECTOR", "0") == "1",
        help="Use projectors without pre-MLP RMSNorm (checkpoint keys: up_proj/down_proj only). "
        "Set LEGACY_PROJECTOR=1 or pass --legacy_projector true for old safetensors.",
    )
    p.add_argument(
        "--vit_init",
        type=str,
        choices=["pretrained", "random"],
        default=os.environ.get("VIT_INIT", "pretrained"),
        help="ViT weights: 'pretrained' (default) or 'random' (same architecture, config init) — prior ablation.",
    )
    p.add_argument(
        "--freeze_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("FREEZE_VIT", "0") == "1",
        help="Freeze the ViT (no weight updates; gradient still flows through it).",
    )
    p.add_argument(
        "--no_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("NO_VIT", "0") == "1",
        help="Bypass the ViT block entirely (no-ViT ablation). Must also be passed at eval.",
    )
    p.add_argument("--max_samples", type=int, default=0, help="Cap HF dataset to first N rows (0 = all).")
    p.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="Keep only samples with prompt token count <= this (no truncation).",
    )
    p.add_argument(
        "--max_answer_len",
        type=int,
        default=2048,
        help="Keep only samples with answer token count <= this (no truncation).",
    )
    p.add_argument(
        "--max_vision_len",
        type=int,
        default=512,
        help="Max compressed-vision sequence length per sample; sizes the learned positional embedding.",
    )
    p.add_argument(
        "--max_source_len",
        type=int,
        default=0,
        help="Max source-encoder tokens per sample (0 = compression_rate * max_vision_len).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--adamw_beta1", type=float, default=0.9)
    p.add_argument("--adamw_beta2", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--enable_warmup", action="store_true")
    p.add_argument("--warmup_ratio", type=int, default=0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument(
        "--disable_lr_schedule",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("DISABLE_LR_SCHEDULE", "0") == "1",
        help="If set, skip cosine LR decay — keep LR constant at init_lr after warmup. "
        "Useful for compression-rate curricula where late stages need fresh LR.",
    )
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument(
        "--save_interval_steps",
        type=int,
        default=0,
        help=(
            "Save every N optimizer steps (sync points). "
            "Set to 0 to save only at the end of each epoch (no extra mid-epoch saves)."
        ),
    )
    p.add_argument(
        "--mechanistic_probe_samples",
        type=int,
        default=int(os.environ.get("MECHANISTIC_PROBE_SAMPLES", "16")),
        help="Save the first N fully tokenized training examples as "
        "<output_dir>/mechanistic_probe.pt for checkpoint-level CKA/activation "
        "analysis without reloading a mutable dataset. 0 disables. "
        "Env: MECHANISTIC_PROBE_SAMPLES.",
    )
    p.add_argument(
        "--save_mechanistic_init",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("SAVE_MECHANISTIC_INIT", "1") != "0",
        help="Save the exact initial compression-path weights as "
        "<output_dir>/mechanistic_init.safetensors. This makes later weight-drift "
        "analysis independent of reconstructing random initialization. "
        "Env: SAVE_MECHANISTIC_INIT.",
    )
    p.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="Directory with model_step_*.safetensors + trainer_step_*.pt; picks latest.",
    )
    p.add_argument(
        "--restore_from", type=str, default="", help="HF-style; sets TrainingArguments.restore_from for init only."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bf16 (use float32 on CUDA).",
    )
    p.add_argument(
        "--tensorboard_dir",
        type=str,
        default="",
        help=(
            "[DEPRECATED] TensorBoard has been replaced by trackio; this flag is "
            "accepted for backward compatibility with launch scripts but is no "
            "longer wired to anything. Use --trackio_project / --trackio_run_name."
        ),
    )
    p.add_argument(
        "--plot_interval",
        type=int,
        default=100,
        help="Refresh PNG dashboards every N optimizer steps. 0 = disable PNG plots.",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default="",
        help="Override CSV metric file location. Defaults to <output_dir>/metrics.csv.",
    )
    p.add_argument(
        "--cr_schedule",
        type=str,
        default=os.environ.get("CR_SCHEDULE", ""),
        help=(
            "Compression-rate curriculum, comma-separated 'cr:end_step' pairs. "
            "Last entry's end_step must be 0 (sentinel for forever). "
            "Empty (default) = static cr from --compression_rate. "
            "Example: '1:6000,2:12000,4:18000,8:0'."
        ),
    )
    p.add_argument(
        "--gradient_checkpointing",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("GRADIENT_CHECKPOINTING", "1") != "0",
        help="Enable activation checkpointing on decoder + ViT (default: on; "
        "override via env GRADIENT_CHECKPOINTING=0 or --gradient_checkpointing=false).",
    )
    p.add_argument(
        "--trackio_project",
        type=str,
        default=os.environ.get("TRACKIO_PROJECT", "cvlm"),
        help="trackio project name. Default: cvlm or $TRACKIO_PROJECT.",
    )
    p.add_argument(
        "--trackio_run_name",
        type=str,
        default=os.environ.get("TRACKIO_RUN_NAME", ""),
        help="trackio run name. Default: basename(output_dir) or $TRACKIO_RUN_NAME.",
    )
    p.add_argument(
        "--trackio_space_id",
        type=str,
        default=os.environ.get("TRACKIO_SPACE_ID", ""),
        help="Optional HF Space id (user/space) to host the dashboard.",
    )
    p.add_argument(
        "--trackio_disable",
        action="store_true",
        default=os.environ.get("TRACKIO_DISABLE", "0") == "1",
        help="Disable trackio logging (CSV/PNG still run).",
    )
    return p.parse_args()


def set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def move_batch(batch: dict, device: torch.device, dtype: torch.dtype) -> dict:
    # All tensors are now long ids / masks; no fp cast needed. dtype kept for API compat.
    del dtype
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable hash over dtype, shape and exact tensor bytes."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def _compression_path_initial_state(model: CVLM) -> dict:
    """Exact step-0 weights needed by post-hoc compressor analyses.

    This intentionally excludes the decoder and text encoder: the former is
    frozen, while this campaign's mechanistic question concerns capacity after
    pooling. The text/projector/pooling/position modules are included because
    they define the interface into the post-bottleneck stack.
    """
    state = {}

    def add_module(prefix: str, module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return
        for name, value in module.state_dict().items():
            state[f"{prefix}.{name}"] = value.detach().cpu().clone().contiguous()

    add_module("text_projector", model.text_projector)
    add_module("vision_projector", model.vision_projector)
    add_module("vision_pos_embed", model.vision_pos_embed)
    state["pool_queries"] = model.pool_queries.detach().cpu().clone().contiguous()
    if model.mixer_layers is not None:
        add_module("mixer_layers", model.mixer_layers)
        state["mixer_gates"] = model.mixer_gates.detach().cpu().clone().contiguous()
    elif model.vision_encoder is not None:
        add_module("vision_encoder_layers", model.vision_encoder.encoder.layer)
        add_module("vision_encoder_final_layernorm", model.vision_encoder.layernorm)
    return state


def _initialization_manifest(model: CVLM, args: argparse.Namespace) -> dict:
    if model.mixer_layers is not None:
        branch_kind = "zero_gated_mixer"
        branch_layers = model.mixer_layers
    elif model.vision_encoder is not None:
        branch_kind = f"vit_{args.vit_init}"
        branch_layers = model.vision_encoder.encoder.layer
    else:
        branch_kind = "no_vit"
        branch_layers = None

    layer_hashes = (
        [_module_sha256(layer) for layer in branch_layers]
        if branch_layers is not None
        else []
    )
    hashes = {
        "text_projector": _module_sha256(model.text_projector),
        "vision_projector": _module_sha256(model.vision_projector),
        "vision_pos_embed": _module_sha256(model.vision_pos_embed),
        "pool_queries": _tensor_sha256(model.pool_queries),
        "post_bottleneck_stack": (
            _module_sha256(branch_layers) if branch_layers is not None else None
        ),
        "post_bottleneck_layers": layer_hashes,
        "post_bottleneck_final_layernorm": (
            _module_sha256(model.vision_encoder.layernorm)
            if model.vision_encoder is not None
            else None
        ),
        "mixer_gates": (
            _tensor_sha256(model.mixer_gates)
            if model.mixer_gates is not None
            else None
        ),
    }
    return {
        "schema_version": 1,
        "seed": int(args.seed),
        "shared_init_seed": int(args.seed) + 10000,
        "branch_kind": branch_kind,
        "architecture": {
            "vit_init": str(args.vit_init),
            "freeze_vit": bool(args.freeze_vit),
            "no_vit": bool(args.no_vit),
            "mixer_depth_map": str(args.mixer_depth_map or ""),
            "mask_vit_padding": bool(args.mask_vit_padding),
            "vit_apply_final_layernorm": bool(args.vit_apply_final_layernorm),
            "num_pool_latents": int(args.num_pool_latents),
        },
        "models": {
            "decoder": str(args.model_name_or_path or model.model_args.model_name_or_path),
            "text_encoder": str(args.text_encoder_name or model.model_args.text_encoder_name),
            "vision_encoder": str(
                args.vision_encoder_name or model.model_args.vision_encoder_name
            ),
            "decoder_commit": getattr(model.decoder.config, "_commit_hash", None),
            "text_encoder_commit": getattr(model.text_encoder.config, "_commit_hash", None),
            "vision_encoder_commit": (
                getattr(model.vision_encoder.config, "_commit_hash", None)
                if model.vision_encoder is not None
                else None
            ),
        },
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "numpy": str(np.__version__),
        },
        "hashes": hashes,
    }


def _write_or_verify_initialization_artifacts(
    model: CVLM,
    args: argparse.Namespace,
) -> None:
    """Persist exact step-0 provenance, failing on an incompatible resume."""
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = _initialization_manifest(model, args)
    manifest_path = os.path.join(args.output_dir, "initialization_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
        for field in ("architecture", "hashes"):
            if recorded.get(field) != manifest.get(field):
                raise RuntimeError(
                    f"Initialization mismatch in {manifest_path}: field {field!r} "
                    "does not match the model reconstructed for resume. Refusing "
                    "to mix trajectories from different initial states."
                )
        print(f"[init-manifest] verified existing {manifest_path}")
    else:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[init-manifest] saved {manifest_path}")

    if args.save_mechanistic_init:
        init_path = os.path.join(args.output_dir, "mechanistic_init.safetensors")
        if os.path.exists(init_path):
            print(f"[mech-init] preserving existing {init_path}")
        else:
            save_file(_compression_path_initial_state(model), init_path)
            print(f"[mech-init] saved exact compression-path initialization -> {init_path}")


def _save_mechanistic_probe(
    dataset,
    collate,
    output_dir: str,
    sample_count: int,
    metadata: dict,
) -> None:
    """Cache a fixed tokenized probe so CKA needs no dataset reload."""
    count = min(max(int(sample_count), 0), len(dataset))
    if count == 0:
        return
    path = os.path.join(output_dir, "mechanistic_probe.pt")
    if os.path.exists(path):
        print(f"[mech-probe] preserving existing {path}")
        return
    batch = collate([dataset[index] for index in range(count)])
    keep = {
        key: value.detach().cpu().contiguous()
        for key, value in batch.items()
        if torch.is_tensor(value) and not key.startswith("full_prompt")
    }
    payload = {
        "schema_version": 1,
        "dataset_indices": list(range(count)),
        "metadata": metadata,
        "batch": keep,
    }
    torch.save(payload, path)
    source_hash = _tensor_sha256(keep["source_ids"])
    print(
        f"[mech-probe] saved {count} tokenized examples -> {path} "
        f"(source_ids_sha256={source_hash})"
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device, use_ddp = setup_distributed()
    is_master = rank == 0
    set_seed(args.seed, rank)

    use_bf16 = device.type == "cuda" and not args.no_bf16

    model_args = ModelArguments()
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name
    if args.text_encoder_name:
        model_args.text_encoder_name = args.text_encoder_name
    if not (0.0 <= args.mixed_fidelity_p <= 1.0):
        raise ValueError(f"--mixed_fidelity_p must be in [0, 1], got {args.mixed_fidelity_p}")
    model_args.max_vision_len = args.max_vision_len
    model_args.compression_rate = args.compression_rate
    model_args.mixed_fidelity_p = float(args.mixed_fidelity_p)
    model_args.unfreeze_encoder_top_k = int(args.unfreeze_encoder_top_k)
    model_args.num_pool_latents = max(int(args.num_pool_latents), 1)
    model_args.projector_use_input_rmsnorm = not bool(args.legacy_projector)
    model_args.vit_init = str(args.vit_init)
    model_args.vit_freeze = bool(args.freeze_vit)
    model_args.vit_skip = bool(args.no_vit)
    model_args.mixer_depth_map = str(args.mixer_depth_map or "")
    model_args.mask_vit_padding = bool(args.mask_vit_padding)
    model_args.vit_apply_final_layernorm = bool(args.vit_apply_final_layernorm)
    # Bit-identical shared-module init across ablation branches (see ModelArguments).
    model_args.shared_init_seed = int(args.seed) + 10000

    training_args = TrainingArguments(output_dir=args.output_dir)
    training_args.bf16 = bool(use_bf16)
    training_args.restore_from = args.restore_from or ""
    training_args.gradient_checkpointing = args.gradient_checkpointing

    model = CVLM(model_args, training_args)
    if is_master:
        _write_or_verify_initialization_artifacts(model, args)
        print(
            f"[init-hash] text_projector={_module_sha256(model.text_projector)[:12]} "
            f"vision_projector={_module_sha256(model.vision_projector)[:12]} "
            f"vision_pos_embed={_module_sha256(model.vision_pos_embed)[:12]} "
            f"(shared_init_seed={model_args.shared_init_seed}; must match across branches)"
        )
    model.to(device)

    # ---- compression-rate curriculum ------------------------------------
    cr_schedule = _parse_cr_schedule(args.cr_schedule)
    if cr_schedule is not None and args.mixed_fidelity_p > 0.0:
        raise ValueError(
            "--mixed_fidelity_p > 0 is incompatible with --cr_schedule: chunk->decoder-token "
            "spans are precomputed in the dataset with the static --compression_rate."
        )
    if cr_schedule is not None:
        # Validate vision-len budget for the smallest cr in the schedule.
        min_cr = min(cr for cr, _ in cr_schedule)
        effective_src = args.max_source_len if args.max_source_len > 0 else args.compression_rate * args.max_vision_len
        required_vl = (effective_src + min_cr - 1) // min_cr
        if args.max_vision_len < required_vl:
            raise ValueError(
                f"max_vision_len={args.max_vision_len} too small for cr_schedule "
                f"min_cr={min_cr} with effective_source_len={effective_src} "
                f"(needs >= {required_vl}). Raise cr_min in the schedule, "
                f"raise --max_vision_len, or lower --max_source_len."
            )
        first_cr = cr_schedule[0][0]
        if first_cr != args.compression_rate and is_master:
            print(
                f"[warn] cr_schedule overrides --compression_rate; "
                f"using first-stage cr={first_cr} "
                f"(--compression_rate was {args.compression_rate})"
            )
        unwrap_model(model).set_compression_rate(first_cr)
        current_cr = first_cr
    else:
        current_cr = args.compression_rate
    # ---- mixed-cr per-step sampling (balanced, deterministic) -----------
    mixed_cr_set = _parse_mixed_cr_set(args.mixed_cr_set)
    mixed_cr_sampler: Optional[MixedCrSampler] = None
    if mixed_cr_set is not None:
        if cr_schedule is not None:
            raise ValueError(
                "--mixed_cr_set is incompatible with --cr_schedule: use per-step cr sampling OR a "
                "stage curriculum, not both (set CR_SCHEDULE='' for mixed-cr runs)."
            )
        if args.mixed_fidelity_p > 0.0:
            raise ValueError(
                "--mixed_cr_set is incompatible with --mixed_fidelity_p > 0: chunk->decoder-token "
                "spans are precomputed in the dataset with the static --compression_rate."
            )
        # Same vision-len budget validation as cr_schedule: the smallest cr in
        # the set must fit the source into max_vision_len chunks.
        min_cr = min(mixed_cr_set)
        effective_src = args.max_source_len if args.max_source_len > 0 else args.compression_rate * args.max_vision_len
        required_vl = (effective_src + min_cr - 1) // min_cr
        if args.max_vision_len < required_vl:
            raise ValueError(
                f"max_vision_len={args.max_vision_len} too small for mixed_cr_set "
                f"min_cr={min_cr} with effective_source_len={effective_src} "
                f"(needs >= {required_vl}). Raise the min cr in the set, "
                f"raise --max_vision_len, or lower --max_source_len."
            )
        mixed_cr_sampler = MixedCrSampler(mixed_cr_set, int(args.seed) + 30000)
        if is_master:
            print(
                f"[mixed-cr] BALANCED per-step cr sampling from {sorted(mixed_cr_set)} "
                f"(shuffled-block sequence, rng seed = seed+30000 = {int(args.seed) + 30000}; "
                f"equal per-cr update counts in any prefix divisible by {len(mixed_cr_set)})"
            )
    # ---------------------------------------------------------------------

    dec_pad = model.tokenizer.pad_token_id
    enc_pad = model.encoder_tokenizer.pad_token_id
    if dec_pad is None or enc_pad is None:
        raise ValueError("Both decoder and encoder tokenizers must define pad_token for batching")

    collate = make_collate_fn(dec_pad_id=dec_pad, enc_pad_id=enc_pad)
    max_source_len = args.max_source_len if args.max_source_len > 0 else args.compression_rate * args.max_vision_len
    dataset = CvlmTrainDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        decoder_tokenizer_name=model_args.model_name_or_path,
        encoder_tokenizer_name=model_args.text_encoder_name,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_source_len=max_source_len,
        max_samples=args.max_samples,
        mixed_fidelity_p=float(args.mixed_fidelity_p),
        mixed_chunk_cr=args.compression_rate,
        mixed_max_chunks=args.max_vision_len,
    )
    if is_master:
        _save_mechanistic_probe(
            dataset,
            collate,
            args.output_dir,
            args.mechanistic_probe_samples,
            {
                "dataset_name": args.dataset_name,
                "dataset_split": args.dataset_split,
                "max_source_len": max_source_len,
                "max_prompt_len": args.max_prompt_len,
                "max_answer_len": args.max_answer_len,
                "max_samples": args.max_samples,
                "seed": args.seed,
            },
        )

    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=False,
            seed=int(args.seed) + 20000,
        )
    else:
        sampler = EpochRandomSampler(dataset, seed=int(args.seed) + 20000)

    # Dedicated generator for DataLoader internals. Sample order itself comes
    # from the explicit epoch-addressable sampler above.
    loader_gen = torch.Generator()
    loader_gen.manual_seed(int(args.seed) + 21000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
        generator=loader_gen,
    )

    if use_ddp:
        # All trainable sub-modules (text_projector, ViT, vision_projector,
        # vision_pos_embed) are hit every forward, and the frozen sub-modules
        # have requires_grad=False so DDP already skips them. find_unused=True
        # forces a reducer walk over all parameters every step (slower) and is
        # unnecessary here; keep it False so a real graph break shows up loudly.
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    param_groups = _build_param_groups(model, args.lr, float(args.gate_lr_mult))
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=0.0,
    )
    if is_master and len(param_groups) > 1:
        print(
            f"[gate-lr] mixer_gates in a separate AdamW group: "
            f"lr={args.lr} * gate_lr_mult={args.gate_lr_mult} = {args.lr * float(args.gate_lr_mult)} "
            f"({len(param_groups[0]['params'])} base tensors + {len(param_groups[1]['params'])} gate tensor(s))"
        )
    grad_clip = args.grad_clip

    grad_accum_steps = args.gradient_accumulation_steps
    batch_size_per_rank = args.batch_size
    eff_batch = args.batch_size * world_size * grad_accum_steps

    steps_per_epoch = max(len(loader) // grad_accum_steps, 1)
    total_steps = max(steps_per_epoch * args.epochs, 1)

    if args.enable_warmup:
        warmup_steps = int(max(args.warmup_ratio * total_steps // 100, args.warmup_steps))
    else:
        warmup_steps = 0

    init_lrs: List[float] = [pg["lr"] for pg in optimizer.param_groups]
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_lr)

    start_epoch = 1
    start_batch = 1
    global_step = 1
    resume_from_dir = args.resume_dir.strip()
    if resume_from_dir:
        if not os.path.isdir(resume_from_dir):
            raise FileNotFoundError(resume_from_dir)
        found = find_latest_checkpoint(resume_from_dir)
        if found is None:
            raise FileNotFoundError(f"No model_step_*.safetensors + trainer_step_*.pt pair in {resume_from_dir}")
        mp, tp, _ = found
        start_epoch, global_step, start_batch = load_cvlm_checkpoint(
            mp,
            tp,
            model,
            optimizer,
            lr_scheduler,
            device,
        )
        if use_ddp:
            dist.barrier()
        if is_master:
            print(
                f"Resumed from {resume_from_dir} next_start_epoch={start_epoch} "
                f"next_start_batch={start_batch} global_step={global_step}"
            )

    if mixed_cr_sampler is not None:
        # cr for the upcoming optimizer step, set BEFORE its first micro-batch.
        # On resume this reconstructs the same sequence position an
        # uninterrupted run would be at (pure function of seed + global_step).
        current_cr = mixed_cr_sampler.cr_for_step(global_step)
        unwrap_model(model).set_compression_rate(current_cr)
        if is_master:
            print(f"[mixed-cr] step {global_step}: cr={current_cr}")

    if is_master:
        print(optimizer)
        print("init_lrs:", init_lrs)
        print(
            f"total training steps (approx): {total_steps}\n"
            f"  epochs: {args.epochs}\n"
            f"  steps_per_epoch (accum): {steps_per_epoch}\n"
            f"  dataloader len: {len(loader)}\n"
            f"  world_size: {world_size}\n"
            f"  grad_accum: {grad_accum_steps}\n"
            f"  batch_size / rank: {batch_size_per_rank}\n"
            f"  effective batch (approx): {eff_batch}\n"
            f"  warmup_steps: {warmup_steps}\n"
            f"  grad_clip: {grad_clip}\n"
        )

    trackio_run = None
    csv_writer = None
    plotter = None
    mechanistic_writer = None
    if is_master:
        from train_logging import MechanisticJsonlWriter, TrackioRun, make_logger

        run_name = args.trackio_run_name.strip() or os.path.basename(args.output_dir.rstrip("/"))
        config_payload = {
            "model_name_or_path": model_args.model_name_or_path,
            "text_encoder_name": model_args.text_encoder_name,
            "vision_encoder_name": getattr(model_args, "vision_encoder_name", ""),
            "compression_rate": args.compression_rate,
            "max_vision_len": args.max_vision_len,
            "mixed_fidelity_p": float(args.mixed_fidelity_p),
            "unfreeze_encoder_top_k": int(args.unfreeze_encoder_top_k),
            "num_pool_latents": int(model_args.num_pool_latents),
            "projector_use_input_rmsnorm": bool(model_args.projector_use_input_rmsnorm),
            "vit_init": model_args.vit_init,
            "vit_freeze": bool(model_args.vit_freeze),
            "vit_skip": bool(model_args.vit_skip),
            "mixer_depth_map": model_args.mixer_depth_map,
            "mask_vit_padding": bool(model_args.mask_vit_padding),
            "vit_apply_final_layernorm": bool(model_args.vit_apply_final_layernorm),
            "gate_lr_mult": float(args.gate_lr_mult),
            "mixed_cr_set": str(args.mixed_cr_set or ""),
            "max_source_len": args.max_source_len,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "lr": args.lr,
            "warmup_steps": warmup_steps,
            "world_size": world_size,
        }
        trackio_run = TrackioRun(
            project=args.trackio_project,
            name=run_name,
            config=config_payload,
            space_id=args.trackio_space_id or None,
            disable=args.trackio_disable,
        )
        csv_writer, plotter = make_logger(args.output_dir, args.csv_path, args.plot_interval)
        mechanistic_writer = MechanisticJsonlWriter(
            os.path.join(args.output_dir, "mechanistic.jsonl")
        )

    model.train()
    # Frozen sub-modules must stay in eval mode (no dropout noise on frozen weights).
    # When the encoder is partially unfrozen (top-K), keep it in train() mode so
    # dropout in the unfrozen blocks fires; lower frozen layers receive no
    # parameter gradients regardless of the train/eval flag.
    if int(args.unfreeze_encoder_top_k) <= 0:
        unwrap_model(model).text_encoder.eval()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    pgs = -1
    # Cross-arm batch-comparability audit: log the sha256 of the first batch's
    # source ids once per process start (i.e. at step 1 and after each resume).
    batch_hash_logged = False

    if start_epoch > args.epochs:
        if is_master:
            print(f"No epochs to run (start_epoch={start_epoch} > epochs={args.epochs}). Exiting.")
        cleanup_distributed(use_ddp)
        return

    for local_epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(local_epoch)

        optimizer.zero_grad(set_to_none=True)
        running_avg_batch_time = AverageMeter("time", ":6.3f")
        running_avg_loss_value = AverageMeter("loss", ":6.3f")
        # MIXED-FIDELITY telemetry (stays at 0 when mixed_fidelity_p == 0).
        running_avg_raw_frac = AverageMeter("raw_frac", ":6.3f")
        running_avg_input_len = AverageMeter("input_len", ":8.1f")
        grad_norm = 0.0
        t0 = time.perf_counter()
        curr_lrs: List[float] = []
        # Latest mixer contribution ratios (device tensor; None until the first
        # forward, and always None when the mixer is disabled) + the cr they
        # were measured at (the vector length == the mixer depth for that cr).
        last_mixer_contrib: Optional[torch.Tensor] = None
        last_mixer_contrib_cr: int = current_cr
        gate_grads_for_log = None
        mixer_layer_grad_norms_for_log = None
        post_bottleneck_layer_grad_norms_for_log = None

        for local_step, batch_data in enumerate(loader, start=1):
            if local_epoch == start_epoch and local_step < start_batch:
                # Recreate the deterministic epoch permutation and consume
                # already-finished batches without touching model/global RNG.
                continue
            data_time = time.perf_counter() - t0
            batch_cpu = batch_data
            if is_master and not batch_hash_logged:
                # Same seed => this line must print an identical hash across
                # ablation arms (loader order is decoupled from model init RNG).
                n_h = min(8, batch_cpu["source_ids"].size(0))
                hb = batch_cpu["source_ids"][:n_h].detach().to(torch.long).contiguous()
                print(f"[batch-hash] step={global_step} hash={hashlib.sha256(hb.numpy().tobytes()).hexdigest()}")
                batch_hash_logged = True
            batch_data = move_batch(batch_cpu, device, dtype)
            # Pin the rate that THIS forward/update uses. `current_cr` is
            # advanced to the next optimizer step inside the sync branch.
            cr_used = int(current_cr)

            # LR schedule: warmup (if any) uses linear on global_step, else read
            # current group LR. The scheduler itself is stepped *after*
            # optimizer.step() so the first update sees the initial LR.
            curr_lrs = []
            if global_step <= warmup_steps and warmup_steps > 0:
                for group_id, param_group in enumerate(optimizer.param_groups):
                    curr_lr = init_lrs[group_id] * global_step / warmup_steps
                    param_group["lr"] = curr_lr
                    curr_lrs.append(curr_lr)
            else:
                for param_group in optimizer.param_groups:
                    curr_lrs.append(param_group["lr"])

            out = model(
                source_input_ids=batch_data["source_ids"],
                source_attention_mask=batch_data["source_attention_mask"],
                prompt_ids=batch_data["prompt_ids"],
                answer_ids=batch_data["answer_ids"],
                answer_labels=batch_data["answer_labels"],
                prompt_mask=batch_data["prompt_mask"],
                answer_mask=batch_data["answer_mask"],
                # MIXED-FIDELITY extras (None when mixed_fidelity_p == 0).
                doc_dec_ids=batch_data.get("doc_dec_ids"),
                chunk_dec_spans=batch_data.get("chunk_dec_spans"),
                chunk_counts=batch_data.get("chunk_counts"),
            )
            loss = out["loss"]
            mixed_stats = out.get("mixed_stats")
            if mixed_stats is not None:
                running_avg_raw_frac.update(mixed_stats["raw_frac"])
                running_avg_input_len.update(mixed_stats["input_len"])
            # RATE-DEPTH MIXER diagnostics: keep the latest per-layer
            # contribution ratios as a DEVICE tensor (detached, no sync); it is
            # copied to CPU only inside the log_interval branch below.
            mixer_contrib_t = out.get("mixer_contrib")
            if mixer_contrib_t is not None:
                last_mixer_contrib = mixer_contrib_t
                # Pin the cr THIS forward ran at: `current_cr` is advanced to the
                # next step's value inside the sync branch below, so the cr shown
                # on a log line can otherwise be one step ahead of the ratios.
                last_mixer_contrib_cr = cr_used
            loss = loss / grad_accum_steps

            sync_step = local_step % grad_accum_steps == 0 or local_step == len(loader)

            if use_ddp and not sync_step:
                with model.no_sync():
                    loss.backward()
            elif not use_ddp and not sync_step:
                loss.backward()
            else:
                loss.backward()
                if grad_clip > 0.0:
                    params = unwrap_model(model).parameters()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
                # Capture gradients BEFORE optimizer.step()/zero_grad so the
                # mechanistic JSONL can show when gates/layers begin learning.
                diag_due = is_master and (global_step + 1) % args.log_interval == 0
                if diag_due:
                    _core_for_grad = unwrap_model(model)
                    _gates_for_grad = getattr(_core_for_grad, "mixer_gates", None)
                    if _gates_for_grad is not None:
                        if _gates_for_grad.grad is None:
                            gate_grads_for_log = [0.0] * int(_gates_for_grad.numel())
                        else:
                            gate_grads_for_log = (
                                _gates_for_grad.grad.detach().float().cpu().tolist()
                            )
                    _layers_for_grad = getattr(_core_for_grad, "mixer_layers", None)
                    if (
                        _layers_for_grad is None
                        and getattr(_core_for_grad, "vision_encoder", None) is not None
                    ):
                        _layers_for_grad = _core_for_grad.vision_encoder.encoder.layer
                    if _layers_for_grad is not None:
                        post_bottleneck_layer_grad_norms_for_log = []
                        for _layer in _layers_for_grad:
                            _norms = [
                                torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
                                for p in _layer.parameters()
                                if p.grad is not None
                            ]
                            _layer_norm = (
                                torch.linalg.vector_norm(torch.stack(_norms))
                                if _norms
                                else torch.zeros((), device=device, dtype=torch.float32)
                            )
                            post_bottleneck_layer_grad_norms_for_log.append(
                                float(_layer_norm)
                            )
                        if getattr(_core_for_grad, "mixer_layers", None) is not None:
                            mixer_layer_grad_norms_for_log = (
                                post_bottleneck_layer_grad_norms_for_log
                            )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if global_step > warmup_steps and not args.disable_lr_schedule:
                    lr_scheduler.step()
                global_step += 1
                if cr_schedule is not None:
                    cr_now = _active_cr(cr_schedule, global_step)
                    if cr_now != current_cr:
                        unwrap_model(model).set_compression_rate(cr_now)
                        if is_master:
                            print(f"[step {global_step}] cr {current_cr} -> {cr_now}")
                            next_epoch, next_batch = _next_resume_position(
                                local_epoch,
                                local_step,
                                len(loader),
                            )
                            save_cvlm_checkpoint(
                                args.output_dir,
                                model,
                                optimizer,
                                lr_scheduler,
                                next_start_epoch=next_epoch,
                                global_step=global_step,
                                is_master=True,
                                next_start_batch=next_batch,
                            )
                        current_cr = cr_now
                elif mixed_cr_sampler is not None:
                    # cr for the NEW global_step, set here — i.e. BEFORE the
                    # first micro-batch of that optimizer step; all grad-accum
                    # micro-batches of a step therefore share one cr.
                    cr_next = mixed_cr_sampler.cr_for_step(global_step)
                    if cr_next != current_cr:
                        unwrap_model(model).set_compression_rate(cr_next)
                    current_cr = cr_next
                if trackio_run is not None and is_master:
                    trackio_run.log({"train/compression_rate": float(cr_used)}, step=global_step)

            t1 = time.perf_counter()
            batch_time = t1 - t0
            running_avg_batch_time.update(batch_time)
            curr_loss = loss.item() * grad_accum_steps
            running_avg_loss_value.update(curr_loss)

            if is_master and sync_step and global_step % args.log_interval == 0:
                rp = global_step / max(total_steps, 1) * 100
                curr_lrs_str = " ' ".join(f"{lr:.10f}" for lr in curr_lrs)
                gn = f"grad norm: {grad_norm:>6.3f} " if grad_norm > 0 else ""
                mx = (
                    f"raw_frac: {running_avg_raw_frac.avg:>5.3f} input_len: {running_avg_input_len.avg:>7.1f} "
                    if args.mixed_fidelity_p > 0
                    else ""
                )
                crs = f"cr: {cr_used} " if mixed_cr_sampler is not None else ""
                print(
                    f"epoch: {local_epoch} step: {global_step} ({total_steps}) | {rp:5.2f}% ",
                    f"lr: {curr_lrs_str} loss: {curr_loss:>6.3f} ({running_avg_loss_value.avg:>6.3f}) ",
                    crs,
                    gn,
                    mx,
                    f"data: {data_time:>7.5f}s batch: {batch_time:>6.3f}s",
                )
                # RATE-DEPTH MIXER ablation telemetry: gate values show how much
                # of each ViTLayer the model actually uses.
                mixer_gates_now = None
                mixer_gates_full = None
                _core = unwrap_model(model)
                if getattr(_core, "mixer_gates", None) is not None:
                    mixer_gates_full = _core.mixer_gates.detach().float().cpu().tolist()
                    mixer_gates_now = [round(float(g), 4) for g in mixer_gates_full]
                    print(f"    mixer_gates={mixer_gates_now}")
                # CONTRIBUTION RATIO: ||g_i*(Layer_i(h)-h)||_F / ||h||_F per
                # applied layer (see modeling._mixer_forward). Unlike the raw
                # gate value this is scale-aware — it says how much the gated
                # update actually moves the representation. Length == the ACTIVE
                # depth for the current cr, so it shrinks/grows under mixed-cr.
                mixer_contrib_now = None
                mixer_contrib_total = None
                if last_mixer_contrib is not None:
                    _contrib_raw = last_mixer_contrib.detach().float().cpu().tolist()
                    mixer_contrib_now = [round(float(c), 4) for c in _contrib_raw]
                    mixer_contrib_total = float(sum(_contrib_raw))
                    print(
                        f"    mixer_contrib={mixer_contrib_now} "
                        f"mixer_contrib_total={mixer_contrib_total:.4f} "
                        f"(cr={last_mixer_contrib_cr}, depth={len(mixer_contrib_now)})"
                    )
                if trackio_run is not None:
                    metrics = {
                        "train/loss": curr_loss,
                        "train/loss_avg": running_avg_loss_value.avg,
                        "train/lr": curr_lrs[0] if curr_lrs else args.lr,
                        "train/batch_time": batch_time,
                    }
                    if grad_norm > 0:
                        metrics["train/grad_norm"] = grad_norm
                    if args.mixed_fidelity_p > 0 and running_avg_raw_frac.count > 0:
                        metrics["train/mixed_raw_frac"] = running_avg_raw_frac.avg
                        metrics["train/mixed_input_len"] = running_avg_input_len.avg
                    if mixer_gates_now is not None:
                        for gi, gv in enumerate(mixer_gates_now):
                            metrics[f"train/mixer_gate_{gi:02d}"] = gv
                    if mixer_contrib_now is not None:
                        metrics["train/mixer_contrib_total"] = mixer_contrib_total
                        # cr the ratios were measured at — under mixed-cr the
                        # depth (== len of the vector) is a function of it, so
                        # any per-cr analysis must group on this, not on
                        # train/compression_rate (which is the NEXT step's cr).
                        metrics["train/mixer_contrib_cr"] = float(last_mixer_contrib_cr)
                        for ci, cv in enumerate(mixer_contrib_now):
                            metrics[f"train/mixer_contrib_{ci:02d}"] = cv
                    trackio_run.log(metrics, step=global_step)
                if mechanistic_writer is not None:
                    vision_mask_diag = out["vision_mask"].detach()
                    vision_lens = vision_mask_diag.sum(dim=1)
                    real_slots = int(vision_mask_diag.sum().item())
                    total_slots = int(vision_mask_diag.numel())
                    source_lens = batch_cpu["source_attention_mask"].sum(dim=1)
                    n_h = min(8, batch_cpu["source_ids"].size(0))
                    hb = batch_cpu["source_ids"][:n_h].detach().to(torch.long).contiguous()
                    if getattr(_core, "mixer_depth_map", None) is not None:
                        active_depth = int(_core._mixer_depth_for(cr_used))
                    elif getattr(_core, "vision_encoder", None) is not None:
                        active_depth = len(_core.vision_encoder.encoder.layer)
                    else:
                        active_depth = 0
                    mechanistic_writer.append(
                        {
                            "step": int(global_step),
                            "optimizer_update": int(global_step - 1),
                            "epoch": int(local_epoch),
                            "local_batch": int(local_step),
                            "batch_size": int(vision_mask_diag.size(0)),
                            "compression_rate": int(cr_used),
                            "active_depth": int(active_depth),
                            "loss": float(curr_loss),
                            "loss_avg": float(running_avg_loss_value.avg),
                            "grad_norm_after_clip": float(grad_norm if grad_norm > 0 else 0.0),
                            "batch_hash": hashlib.sha256(hb.numpy().tobytes()).hexdigest(),
                            "source_len_mean": float(source_lens.float().mean()),
                            "source_len_min": int(source_lens.min()),
                            "source_len_max": int(source_lens.max()),
                            "vision_len_mean": float(real_slots / max(vision_mask_diag.size(0), 1)),
                            "vision_len_min": int(vision_lens.min()),
                            "vision_len_max": int(vision_lens.max()),
                            "latent_slots_total": total_slots,
                            "latent_slots_real": real_slots,
                            "latent_padding_fraction": float(
                                (total_slots - real_slots) / max(total_slots, 1)
                            ),
                            "mask_vit_padding": bool(model_args.mask_vit_padding),
                            "vit_apply_final_layernorm": bool(
                                model_args.vit_apply_final_layernorm
                            ),
                            "mixer_gates": mixer_gates_full,
                            "mixer_gate_grads": gate_grads_for_log,
                            "mixer_contrib": (
                                last_mixer_contrib.detach().float().cpu().tolist()
                                if last_mixer_contrib is not None
                                else None
                            ),
                            "mixer_layer_grad_norms": mixer_layer_grad_norms_for_log,
                            "post_bottleneck_layer_grad_norms": (
                                post_bottleneck_layer_grad_norms_for_log
                            ),
                        }
                    )
                    gate_grads_for_log = None
                    mixer_layer_grad_norms_for_log = None
                    post_bottleneck_layer_grad_norms_for_log = None
                if csv_writer is not None:
                    csv_writer.append(
                        step=global_step,
                        loss=curr_loss,
                        loss_avg=running_avg_loss_value.avg,
                        lr=curr_lrs[0] if curr_lrs else args.lr,
                        grad_norm=grad_norm if grad_norm > 0 else 0.0,
                        batch_time=batch_time,
                    )

            if (
                is_master
                and plotter is not None
                and args.plot_interval > 0
                and global_step % args.plot_interval == 0
                and sync_step
            ):
                plotter.refresh()

            if is_master and args.save_interval_steps > 0 and global_step % args.save_interval_steps == 0 and sync_step:
                if global_step != pgs:
                    next_epoch, next_batch = _next_resume_position(
                        local_epoch,
                        local_step,
                        len(loader),
                    )
                    save_cvlm_checkpoint(
                        args.output_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        next_start_epoch=next_epoch,
                        global_step=global_step,
                        is_master=True,
                        next_start_batch=next_batch,
                    )
                    pgs = global_step

            t0 = time.perf_counter()

        # Without step interval, checkpoint once per epoch. With N>0, rely on
        # save_interval_steps only (otherwise every epoch end also saves, which is
        # often much more frequent than N when epochs are short).
        if is_master and args.save_interval_steps <= 0:
            if global_step != pgs:
                save_cvlm_checkpoint(
                    args.output_dir,
                    model,
                    optimizer,
                    lr_scheduler,
                    next_start_epoch=local_epoch + 1,
                    global_step=global_step,
                    is_master=True,
                    next_start_batch=1,
                )
                pgs = global_step

    if is_master and args.save_interval_steps > 0 and global_step != pgs:
        save_cvlm_checkpoint(
            args.output_dir,
            model,
            optimizer,
            lr_scheduler,
            next_start_epoch=args.epochs + 1,
            global_step=global_step,
            is_master=True,
            next_start_batch=1,
        )

    if is_master:
        print("Training finished.")
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[train] peak GPU memory: {peak_gb:.2f} GB")
            if trackio_run is not None and trackio_run.enabled:
                trackio_run.log({"system/peak_gpu_gb": peak_gb}, step=global_step)

    if trackio_run is not None:
        trackio_run.finish()
    if csv_writer is not None:
        csv_writer.close()
    if mechanistic_writer is not None:
        mechanistic_writer.close()
    if plotter is not None:
        plotter.refresh()

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
