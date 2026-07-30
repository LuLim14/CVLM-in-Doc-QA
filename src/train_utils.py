# Utilities for CVLM training (patterns from src/train.py).

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer

# Parameter dtypes whose ULP is too coarse for a raw AdamW update at our LRs.
LOW_PRECISION_DTYPES = (torch.bfloat16, torch.float16)
# Extra key stored inside the AdamW state_dict by MasterWeightAdamW.
MASTER_WEIGHTS_KEY = "fp32_master_weights"


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


class MasterWeightAdamW(torch.optim.AdamW):
    """AdamW that keeps an fp32 MASTER copy of every low-precision parameter.

    WHY THIS EXISTS. The whole model is built in bf16 and (until this class)
    AdamW updated those bf16 tensors in place, with bf16 moments. bf16 keeps 8
    mantissa bits, so a weight of magnitude ~1.0 has ULP 2**-7 = 7.8e-3 and a
    weight of magnitude ~0.02 has ULP ~1.2e-4. An AdamW update is ~lr in size
    (the moment ratio is O(1)), i.e. 1e-4 in this campaign. For the small
    random-init weights that is above half an ULP and lands; for the ~4x larger
    PRETRAINED ViT weights it is below half an ULP and is silently rounded
    away — every step, forever. Measured on the rd5/rd6 run: 63.6% of the
    pretrained ViT weights were BIT-for-BIT unchanged between step 1000 and
    step 11326 (drift 2.77%) versus 19.5% drift for the random-init arm. Any
    "the pretrained visual prior hurts" claim is therefore confounded with a
    pure floating-point artifact that is COLLINEAR with the treatment.

    WHAT IT DOES. Parameters stay bf16 for forward/backward (activation memory
    and speed are untouched, and the model/checkpoint dtype does not change),
    but the optimizer owns an fp32 clone of each of them:

        master.grad <- p.grad        (upcast copy, after clipping)
        AdamW.step() on the fp32 masters, with fp32 moments
        p <- master                  (round-to-nearest back into bf16)

    Sub-ULP updates now ACCUMULATE in the master instead of being destroyed, so
    a pretrained weight moves as soon as the accumulated update crosses half an
    ULP. fp32 parameters already in the group (pool_queries, mixer_gates) are
    used directly with no master copy.

    STATE / RESUME. ``self.param_groups`` holds the MASTER tensors, so
    LR schedulers, ``check_optimizer_groups_match`` and the AdamW state machine
    all work unchanged, and the moments are created fp32 (``zeros_like(master)``)
    rather than bf16. ``state_dict()`` additionally stores the master tensors
    themselves under ``MASTER_WEIGHTS_KEY``; ``load_state_dict()`` restores them
    in place, so a resumed run continues from the FULL-precision weights instead
    of re-seeding from the rounded bf16 checkpoint.

    MEMORY. +4 bytes/param for the master, +4 for its fp32 grad buffer, and
    fp32 (8B) instead of bf16 (4B) moments == +12 bytes per low-precision
    trainable parameter (~2.5 GB for this campaign's ~211M trainable params).
    """

    def __init__(self, params: Any, **kwargs: Any) -> None:
        groups = list(params)
        if not groups:
            raise ValueError("MasterWeightAdamW got an empty parameter list")
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        master_groups: List[dict] = []
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for group in groups:
            group = dict(group)
            master_params: List[torch.Tensor] = []
            for param in list(group["params"]):
                if param.dtype in LOW_PRECISION_DTYPES:
                    master = param.detach().clone().to(torch.float32)
                    master.requires_grad_(param.requires_grad)
                    pairs.append((param, master))
                    master_params.append(master)
                else:
                    # Already full precision: optimize the live tensor itself.
                    master_params.append(param)
            group["params"] = master_params
            master_groups.append(group)
        super().__init__(master_groups, **kwargs)
        self._master_pairs = pairs

    # -- introspection -----------------------------------------------------
    @property
    def master_pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """(model_param, fp32_master) for every mastered (bf16/fp16) tensor."""
        return list(self._master_pairs)

    @property
    def mastered_numel(self) -> int:
        return sum(master.numel() for _, master in self._master_pairs)

    def _flat_master_params(self) -> List[torch.Tensor]:
        """Optimizer params in the exact order ``state_dict`` indexes them."""
        return [p for group in self.param_groups for p in group["params"]]

    # -- optimization ------------------------------------------------------
    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        with torch.no_grad():
            for model_param, master in self._master_pairs:
                grad = model_param.grad
                if grad is None:
                    # Mirror plain-AdamW semantics: no grad => not updated.
                    master.grad = None
                    continue
                if master.grad is None:
                    master.grad = torch.zeros_like(master)
                master.grad.copy_(grad)
        loss = super().step(closure)
        with torch.no_grad():
            for model_param, master in self._master_pairs:
                model_param.copy_(master)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        # The base call clears the MASTER grads; the live model params have
        # their own grads (produced by backward) and must be cleared too.
        super().zero_grad(set_to_none=set_to_none)
        for model_param, _ in self._master_pairs:
            if model_param.grad is None:
                continue
            if set_to_none:
                model_param.grad = None
            else:
                model_param.grad.detach_()
                model_param.grad.zero_()

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        mastered_ids = {id(master) for _, master in self._master_pairs}
        payload: Dict[int, torch.Tensor] = {}
        for index, param in enumerate(self._flat_master_params()):
            if id(param) in mastered_ids:
                payload[index] = param.detach().cpu().clone()
        state[MASTER_WEIGHTS_KEY] = {"version": 1, "params": payload}
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        payload = state_dict.get(MASTER_WEIGHTS_KEY)
        base = {k: v for k, v in state_dict.items() if k != MASTER_WEIGHTS_KEY}
        super().load_state_dict(base)
        if payload is None:
            # Legacy (pre-master-weights) checkpoint: the masters keep the
            # values they were seeded with, i.e. the rounded bf16 weights that
            # were just loaded into the model. Correct, but the sub-ULP history
            # of that run is gone — say so instead of pretending otherwise.
            print(
                "[fp32-master] WARNING: checkpoint carries no fp32 master weights; "
                "seeding masters from the rounded bf16 parameters."
            )
            return
        saved = payload.get("params", {})
        if len(saved) != len(self._master_pairs):
            raise RuntimeError(
                f"fp32 master-weight resume mismatch: checkpoint has {len(saved)} master "
                f"tensor(s), the current optimizer needs {len(self._master_pairs)}. The "
                "trainable-parameter set or the parameter dtypes changed since the "
                "checkpoint was written."
            )
        flat = self._flat_master_params()
        with torch.no_grad():
            for index, value in saved.items():
                index = int(index)
                if index >= len(flat):
                    raise RuntimeError(
                        f"fp32 master-weight resume mismatch: saved index {index} is out of "
                        f"range for {len(flat)} optimizer tensor(s)."
                    )
                master = flat[index]
                if tuple(value.shape) != tuple(master.shape):
                    raise RuntimeError(
                        f"fp32 master-weight resume mismatch at index {index}: checkpoint "
                        f"shape {tuple(value.shape)} != model shape {tuple(master.shape)}."
                    )
                master.copy_(value.to(device=master.device, dtype=master.dtype))
            # The bf16 weights loaded from the safetensors are round(master), so
            # this is a no-op bit-wise; do it anyway so the live model can never
            # silently disagree with the state the optimizer will step from.
            for model_param, master in self._master_pairs:
                model_param.copy_(master)


def build_optimizer(
    param_groups: Iterable[dict],
    lr: float,
    betas: Tuple[float, float],
    weight_decay: float = 0.0,
    fp32_master_weights: bool = False,
) -> Optimizer:
    """AdamW, optionally with fp32 master weights (see ``MasterWeightAdamW``).

    ``fp32_master_weights=False`` returns a plain ``torch.optim.AdamW`` built
    exactly as before, so the legacy trajectory is reproduced byte for byte.
    """
    if not fp32_master_weights:
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    return MasterWeightAdamW(
        param_groups,
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


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


def validate_partial_checkpoint_load(
    module: nn.Module,
    missing: list,
    unexpected: list,
    checkpoint_path: str,
    allowed_missing_trainable: Optional[Set[str]] = None,
) -> None:
    """Reject architecture/flag mismatches hidden by ``strict=False``.

    CVLM checkpoints intentionally omit frozen decoder/text-encoder weights, so
    strict loading is impossible. Silently accepting unexpected keys or a
    missing trainable compression-path parameter can nevertheless turn an
    evaluation into a different architecture (for example, loading a ViT
    checkpoint with ``--no_vit``).
    """
    allowed = allowed_missing_trainable or set()
    trainable = {
        name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    critical_missing = sorted(
        name
        for name in missing
        if name in trainable and name not in allowed
    )
    if unexpected or critical_missing:
        details = []
        if unexpected:
            details.append(
                f"unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}"
            )
        if critical_missing:
            details.append(
                f"missing trainable keys ({len(critical_missing)}): "
                f"{critical_missing[:8]}"
            )
        raise RuntimeError(
            f"Checkpoint/model architecture mismatch for {checkpoint_path}: "
            + "; ".join(details)
            + ". Reuse the exact training flags (vit_init/no_vit/mixer_depth_map/"
            "final-layernorm/legacy-projector)."
        )


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


def check_master_weights_match(
    optimizer: Optimizer,
    saved_state: Dict[str, Any],
    trainer_path: str = "",
) -> None:
    """Refuse to silently drop fp32 master weights on resume.

    ``Optimizer.load_state_dict`` ignores unknown top-level keys, so resuming a
    ``--fp32_master_weights`` run with the flag OFF would quietly throw the
    full-precision weights away AND downcast the moments back to bf16 — the run
    would keep going and look fine while the thing under test disappeared.
    """
    where = f" (from {trainer_path})" if trainer_path else ""
    saved_has = MASTER_WEIGHTS_KEY in saved_state
    live_has = isinstance(optimizer, MasterWeightAdamW)
    if saved_has and not live_has:
        raise RuntimeError(
            f"Optimizer resume mismatch{where}: the checkpoint stores fp32 master weights but "
            "the current optimizer is a plain AdamW. Resume with --fp32_master_weights true "
            "(env FP32_MASTER_WEIGHTS=1), or start a fresh run — continuing without them would "
            "discard the full-precision weights and silently change the numerics mid-run."
        )
    if live_has and not saved_has:
        print(
            f"[fp32-master] NOTE: checkpoint{where} was written WITHOUT fp32 master weights; "
            "masters are seeded from the rounded bf16 parameters (mid-run precision change)."
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
    check_master_weights_match(optimizer, payload["optimizer"], trainer_path)
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
