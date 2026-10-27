# Utilities for CVLM training (patterns from src/train.py).

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer


class AverageMeter:
    """Computes and stores the average and current value (llama-cookbook style)."""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __str__(self) -> str:
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


def setup_distributed() -> Tuple[int, int, int, torch.device, bool]:
    """Returns rank, world_size, local_rank, device, use_ddp."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return rank, world_size, local_rank, device, True

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return 0, 1, 0, device, False


def cleanup_distributed(use_ddp: bool) -> None:
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def state_dict_for_safetensors(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Build a state dict safe for ``save_file``.

    - Keeps only parameters whose ``requires_grad`` is True. Frozen sub-modules
      (decoder, text_encoder) are reloaded from their HF checkpoints at
      `from_pretrained` time and don't need to live in our per-step ckpts.
      This shrinks a 500+ MB checkpoint down to ~100 MB of trainable weights.
    - Clones any later tensor that shares storage with an earlier one (tied
      LM heads, etc.), which ``safetensors.save_file`` cannot represent.
    """
    # Build the set of parameter names we want to save (trainable only).
    trainable_names: Set[str] = {
        name for name, p in module.named_parameters() if p.requires_grad
    }
    sd = module.state_dict()
    seen: Set[tuple] = set()
    out: Dict[str, torch.Tensor] = {}
    for name, t in sd.items():
        if name not in trainable_names:
            continue
        stor = t.untyped_storage()
        ident = (stor.data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        if ident in seen:
            out[name] = t.detach().clone().contiguous()
        else:
            seen.add(ident)
            out[name] = t.detach()
    return out


def save_cvlm_checkpoint(
    output_dir: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    next_start_epoch: int,
    global_step: int,
    is_master: bool,
    prune_old_trainer_state: bool = True,
) -> None:
    if not is_master:
        return
    os.makedirs(output_dir, exist_ok=True)
    core = unwrap_model(model)
    model_path = os.path.join(output_dir, f"model_step_{global_step}.safetensors")
    save_file(state_dict_for_safetensors(core), model_path)
    trainer_path = os.path.join(output_dir, f"trainer_step_{global_step}.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_start_epoch": next_start_epoch,
            "global_step": global_step,
        },
        trainer_path,
    )
    # Optimizer state is ~2x the model size and is only needed to *resume* from
    # the latest step. Keep just the trainer file we wrote here and drop older
    # ones. Model safetensors are never pruned (the eval sweep needs them all).
    if prune_old_trainer_state:
        prune_old_trainer_files(output_dir, keep_step=global_step)


def prune_old_trainer_files(output_dir: str, keep_step: int) -> None:
    """Delete trainer_step_*.pt except the one for ``keep_step``."""
    for name in os.listdir(output_dir):
        m = re.match(r"trainer_step_(\d+)\.pt$", name)
        if m and int(m.group(1)) != keep_step:
            try:
                os.remove(os.path.join(output_dir, name))
            except OSError as exc:  # pragma: no cover - best-effort cleanup
                print(f"[train_utils] could not prune {name}: {exc}")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[Tuple[str, str, int]]:
    """Return (model_path, trainer_path, step) for highest model_step_*.safetensors."""
    if not os.path.isdir(checkpoint_dir):
        return None
    best: Optional[Tuple[int, str, str]] = None
    for name in os.listdir(checkpoint_dir):
        m = re.match(r"model_step_(\d+)\.safetensors$", name)
        if m:
            step = int(m.group(1))
            mp = os.path.join(checkpoint_dir, name)
            tp = os.path.join(checkpoint_dir, f"trainer_step_{step}.pt")
            if os.path.isfile(tp):
                if best is None or step > best[0]:
                    best = (step, mp, tp)
    if best is None:
        return None
    _, mp, tp = best
    return (mp, tp, best[0])


def load_cvlm_checkpoint(
    model_path: str,
    trainer_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, int]:
    """Load weights and trainer state. Returns (local_epoch, global_step) from checkpoint."""
    state = load_file(model_path)
    # strict=False because the checkpoint only contains trainable params;
    # frozen sub-modules (decoder, text_encoder) were re-loaded from HF at init.
    missing, unexpected = unwrap_model(model).load_state_dict(state, strict=False)
    if unexpected:
        print(f"load_cvlm_checkpoint: {len(unexpected)} unexpected keys in ckpt (ignored)")
    try:
        payload = torch.load(trainer_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(trainer_path, map_location=device)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    gstep = int(payload["global_step"])
    if "next_start_epoch" in payload:
        nxt = int(payload["next_start_epoch"])
    else:
        nxt = int(payload.get("local_epoch", 0)) + 1
    return nxt, gstep
