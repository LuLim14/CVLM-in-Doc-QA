# Full CVLM training entrypoint (reference: src/train.py — DDP, accum, cosine LR, ckpt).

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
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
        "--vision_backend",
        type=str,
        default=os.environ.get("VISION_BACKEND", "vit"),
        choices=["vit", "random_vit", "none"],
        help="Vision-stage ablation. 'vit' (default) = pretrained ViT. 'random_vit' = same "
        "ViT architecture, random weights. 'none' = bypass the ViT transformer+layernorm "
        "(pool -> projectors -> decoder).",
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
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        default=os.environ.get("USE_CHAT_TEMPLATE", "0") == "1",
        help="Wrap the document+question+answer in the decoder chat template "
        "(system+user turns, assistant-open tag). Must match at eval time.",
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
    model_args.max_vision_len = args.max_vision_len
    model_args.compression_rate = args.compression_rate
    model_args.unfreeze_encoder_top_k = int(args.unfreeze_encoder_top_k)
    model_args.num_pool_latents = max(int(args.num_pool_latents), 1)
    model_args.projector_use_input_rmsnorm = not bool(args.legacy_projector)
    model_args.vision_backend = str(args.vision_backend)

    training_args = TrainingArguments(output_dir=args.output_dir)
    training_args.bf16 = bool(use_bf16)
    training_args.restore_from = args.restore_from or ""
    training_args.gradient_checkpointing = args.gradient_checkpointing

    model = CVLM(model_args, training_args)
    model.to(device)

    # ---- compression-rate curriculum ------------------------------------
    cr_schedule = _parse_cr_schedule(args.cr_schedule)
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
        use_chat_template=args.use_chat_template,
    )

    sampler: Optional[DistributedSampler] = None
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(args.adamw_beta1, args.adamw_beta2),
        weight_decay=0.0,
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
    global_step = 1
    resume_from_dir = args.resume_dir.strip()
    if resume_from_dir:
        if not os.path.isdir(resume_from_dir):
            raise FileNotFoundError(resume_from_dir)
        found = find_latest_checkpoint(resume_from_dir)
        if found is None:
            raise FileNotFoundError(f"No model_step_*.safetensors + trainer_step_*.pt pair in {resume_from_dir}")
        mp, tp, _ = found
        start_epoch, global_step = load_cvlm_checkpoint(mp, tp, model, optimizer, lr_scheduler, device)
        if use_ddp:
            dist.barrier()
        if is_master:
            print(f"Resumed from {resume_from_dir} next_start_epoch={start_epoch} global_step={global_step}")

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
    if is_master:
        from train_logging import TrackioRun, make_logger

        run_name = args.trackio_run_name.strip() or os.path.basename(args.output_dir.rstrip("/"))
        config_payload = {
            "model_name_or_path": model_args.model_name_or_path,
            "text_encoder_name": model_args.text_encoder_name,
            "vision_encoder_name": getattr(model_args, "vision_encoder_name", ""),
            "compression_rate": args.compression_rate,
            "max_vision_len": args.max_vision_len,
            "unfreeze_encoder_top_k": int(args.unfreeze_encoder_top_k),
            "num_pool_latents": int(model_args.num_pool_latents),
            "projector_use_input_rmsnorm": bool(model_args.projector_use_input_rmsnorm),
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

    model.train()
    # Frozen sub-modules must stay in eval mode (no dropout noise on frozen weights).
    # When the encoder is partially unfrozen (top-K), keep it in train() mode so
    # dropout in the unfrozen blocks fires; lower frozen layers receive no
    # parameter gradients regardless of the train/eval flag.
    if int(args.unfreeze_encoder_top_k) <= 0:
        unwrap_model(model).text_encoder.eval()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    pgs = -1

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
        grad_norm = 0.0
        t0 = time.perf_counter()
        curr_lrs: List[float] = []

        for local_step, batch_data in enumerate(loader, start=1):
            data_time = time.perf_counter() - t0
            batch_cpu = batch_data
            batch_data = move_batch(batch_cpu, device, dtype)

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
            )
            loss = out["loss"]
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
                            save_cvlm_checkpoint(
                                args.output_dir,
                                model,
                                optimizer,
                                lr_scheduler,
                                next_start_epoch=local_epoch,
                                global_step=global_step,
                                is_master=True,
                            )
                        current_cr = cr_now
                if trackio_run is not None and is_master:
                    trackio_run.log({"train/compression_rate": float(current_cr)}, step=global_step)

            t1 = time.perf_counter()
            batch_time = t1 - t0
            running_avg_batch_time.update(batch_time)
            curr_loss = loss.item() * grad_accum_steps
            running_avg_loss_value.update(curr_loss)

            if is_master and global_step % args.log_interval == 0:
                rp = global_step / max(total_steps, 1) * 100
                curr_lrs_str = " ' ".join(f"{lr:.10f}" for lr in curr_lrs)
                gn = f"grad norm: {grad_norm:>6.3f} " if grad_norm > 0 else ""
                print(
                    f"epoch: {local_epoch} step: {global_step} ({total_steps}) | {rp:5.2f}% ",
                    f"lr: {curr_lrs_str} loss: {curr_loss:>6.3f} ({running_avg_loss_value.avg:>6.3f}) ",
                    gn,
                    f"data: {data_time:>7.5f}s batch: {batch_time:>6.3f}s",
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
                    trackio_run.log(metrics, step=global_step)
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
                    save_cvlm_checkpoint(
                        args.output_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        next_start_epoch=local_epoch,
                        global_step=global_step,
                        is_master=True,
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
    if plotter is not None:
        plotter.refresh()

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
