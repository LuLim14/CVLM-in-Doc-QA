# Utilities for CVLM training (patterns from src/train.py).

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
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
    next_start_batch: int = 1,
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
            "next_start_batch": int(next_start_batch),
            "global_step": global_step,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        trainer_path,
    )


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


def check_optimizer_groups_match(
    optimizer: Optimizer,
    saved_state: Dict[str, Any],
    trainer_path: str = "",
) -> None:
    """Fail loudly when a saved optimizer state does not fit the live optimizer.

    ``Optimizer.load_state_dict`` matches param groups POSITIONALLY and only
    checks that the total parameter count lines up; a run whose group layout
    changed between save and resume (e.g. ``--gate_lr_mult`` added or dropped,
    which splits ``mixer_gates`` into its own group) would either raise an
    opaque "loaded state dict has a different number of parameter groups" or —
    worse, when the counts coincidentally match — silently graft Adam moments
    onto the wrong tensors and the wrong LR onto the wrong group. This check
    turns both into one explicit message.
    """
    saved_groups = saved_state.get("param_groups", [])
    cur_groups = optimizer.param_groups
    where = f" (from {trainer_path})" if trainer_path else ""
    if len(saved_groups) != len(cur_groups):
        raise RuntimeError(
            f"Optimizer resume mismatch{where}: checkpoint has {len(saved_groups)} param group(s), "
            f"the current optimizer has {len(cur_groups)}. The param-group layout is decided by "
            f"--gate_lr_mult (1.0 -> 1 group; anything else -> 2 groups: base, mixer_gates). "
            f"Resume with the SAME --gate_lr_mult (env GATE_LR_MULT) the run was started with, "
            f"or start a fresh run."
        )
    for i, (sg, cg) in enumerate(zip(saved_groups, cur_groups)):
        n_saved = len(sg.get("params", []))
        n_cur = len(cg["params"])
        if n_saved != n_cur:
            raise RuntimeError(
                f"Optimizer resume mismatch{where}: param group {i} has {n_saved} tensor(s) in the "
                f"checkpoint but {n_cur} in the current optimizer. The trainable-parameter set or "
                f"the group split (--gate_lr_mult) changed since the checkpoint was written; "
                f"resume with the same model/optimizer flags or start a fresh run."
            )


def load_cvlm_checkpoint(
    model_path: str,
    trainer_path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, int, int]:
    """Load weights/trainer/RNG state.

    Returns ``(local_epoch, global_step, next_start_batch)``. New checkpoints
    resume at the exact next DataLoader batch; legacy checkpoints without the
    batch field fall back to batch 1 of their recorded epoch.
    """
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
    check_optimizer_groups_match(optimizer, payload["optimizer"], trainer_path)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng_state = payload.get("rng_state")
    if rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        cuda_state = rng_state.get("torch_cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
    gstep = int(payload["global_step"])
    if "next_start_epoch" in payload:
        nxt = int(payload["next_start_epoch"])
    else:
        nxt = int(payload.get("local_epoch", 0)) + 1
    next_batch = int(payload.get("next_start_batch", 1))
    if next_batch < 1:
        raise RuntimeError(
            f"Invalid next_start_batch={next_batch} in checkpoint {trainer_path}; expected >= 1"
        )
    return nxt, gstep, next_batch
