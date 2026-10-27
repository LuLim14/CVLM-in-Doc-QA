# SFT baseline trainer.
#
# Trains the same decoder used by CVLM (e.g. SmolLM-135M-Instruct) on
# `document + "\n\n" + question -> answer` with standard next-token prediction.
# Loss is masked with -100 on prompt and pad positions, so cross-entropy is
# computed only on the answer tokens — same masking semantics as CVLM.
#
# Input format mirrors `eval_cvlm.py --mode baseline_llm_full` exactly so the
# trained checkpoint is a true apples-to-apples ceiling baseline for CVLM.
#
# Implementation note: spec mentions TRL `SFTTrainer`. trl is not installed in
# the project env, and the masking we need is small enough to spell out
# directly with `transformers.Trainer` + a custom collator. The result is
# semantically identical (TRL's `DataCollatorForCompletionOnlyLM` does the same
# prompt-region -> -100 masking).

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from train_logging import make_logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# Patch decoder family with fused Triton kernels (RoPE, RMSNorm, SwiGLU,
# fused-linear-CE) BEFORE any `AutoModelForCausalLM.from_pretrained(...)`.
# fused_linear_cross_entropy is the critical one: it fuses lm_head matmul +
# log-softmax + NLL into a single kernel and never materialises the full
# `[B, L, vocab]` fp32 logits tensor. For Qwen3 (vocab=151936) this is the
# difference between fitting and OOM — unfused fp32 logits at B=4, L=1536 are
# already ~3.7 GB and several copies live simultaneously during backward.
from transformers import AutoConfig
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN


def _apply_liger_for(model_name_or_path: str) -> None:
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "")
    fn = MODEL_TYPE_TO_APPLY_LIGER_FN.get(mt)
    if fn is None:
        print(f"[train_sft] WARN: no liger patch for model_type={mt!r}; "
              f"expect large unfused fp32 logits tensors at high vocab.")
        return
    fn(rope=True, rms_norm=True, swiglu=True,
       fused_linear_cross_entropy=True, cross_entropy=False)
    print(f"[train_sft] applied {fn.__name__} (model_type={mt})")


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class SftFullDocDataset(Dataset):
    """Yields {"input_ids", "labels"} where prompt tokens are masked to -100.

    Filter logic mirrors `CvlmTrainDataset` so the accepted-sample set matches
    sample-for-sample (question-only prompt cap, answer cap; document length is
    unbounded — same as CVLM).
    """

    def __init__(
        self,
        hf_dataset_name: str,
        hf_split: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_len: int,
        max_answer_len: int,
        max_samples: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_answer_len = max_answer_len

        ds = load_dataset(hf_dataset_name, split=hf_split)
        if max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        inputs = list(ds["input"])
        prompts = list(ds["prompt"])
        answers = list(ds["answer"])

        print(f"[train_sft] Tokenising {len(inputs)} samples for length filter...")
        prompt_lens = self._batched_lengths(tokenizer, prompts)
        answer_lens = self._batched_lengths(tokenizer, answers)

        keep: List[int] = []
        for i in range(len(inputs)):
            if prompt_lens[i] <= 0 or prompt_lens[i] > max_prompt_len:
                continue
            if answer_lens[i] <= 0 or answer_lens[i] > max_answer_len:
                continue
            keep.append(i)
        self._inputs = [inputs[i] for i in keep]
        self._prompts = [prompts[i] for i in keep]
        self._answers = [answers[i] for i in keep]
        print(
            f"[train_sft] kept {len(keep)}/{len(inputs)} samples "
            f"(prompt<={max_prompt_len}, answer<={max_answer_len})"
        )
        if not keep:
            raise RuntimeError("No samples passed length filters; loosen caps.")

        self.eos_id: Optional[int] = tokenizer.eos_token_id

    @staticmethod
    def _batched_lengths(tokenizer: PreTrainedTokenizer, texts: List[str], batch: int = 1024) -> List[int]:
        out: List[int] = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                [str(t) for t in texts[start:start + batch]],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            out.extend(len(ids) for ids in enc["input_ids"])
        return out

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        full_prompt_text = str(self._inputs[idx]) + "\n\n" + str(self._prompts[idx])
        prompt_ids = self.tokenizer(
            full_prompt_text, add_special_tokens=False, truncation=False
        )["input_ids"]
        answer_ids = self.tokenizer(
            str(self._answers[idx]), add_special_tokens=False, truncation=False
        )["input_ids"]
        if self.eos_id is not None:
            answer_ids = list(answer_ids) + [self.eos_id]
        input_ids = list(prompt_ids) + list(answer_ids)
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {"input_ids": input_ids, "labels": labels}


# -----------------------------------------------------------------------------
# Collator: right-pad input_ids/labels/attention_mask in a batch.
# -----------------------------------------------------------------------------


@dataclass
class SftCollator:
    pad_token_id: int

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(s["input_ids"]) for s in samples)
        b = len(samples)
        input_ids = torch.full((b, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((b, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long)
        for i, s in enumerate(samples):
            n = len(s["input_ids"])
            input_ids[i, :n] = torch.as_tensor(s["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.as_tensor(s["labels"], dtype=torch.long)
            attention_mask[i, :n] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# -----------------------------------------------------------------------------
# CSV/PNG callback — mirrors what train_cvlm.py does for CVLM.
# -----------------------------------------------------------------------------


class PngLoggingCallback(TrainerCallback):
    """Writes per-log-step CSV row + refreshes PNG dashboards on master rank."""

    def __init__(self, output_dir: str, csv_path: str, plot_interval: int) -> None:
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.plot_interval = plot_interval
        self._csv = None
        self._plotter = None
        self._loss_sum = 0.0
        self._loss_count = 0
        self._t_prev: Optional[float] = None

    def _ensure_logger(self) -> None:
        if self._csv is None:
            self._csv, self._plotter = make_logger(
                self.output_dir, self.csv_path, self.plot_interval
            )

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return
        self._ensure_logger()
        self._t_prev = time.perf_counter()

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero or logs is None:
            return
        self._ensure_logger()
        # HF Trainer emits eval logs too; skip them.
        if "loss" not in logs:
            return
        loss = float(logs.get("loss", 0.0))
        lr = float(logs.get("learning_rate", 0.0))
        grad_norm = float(logs.get("grad_norm", 0.0))
        self._loss_sum += loss
        self._loss_count += 1
        loss_avg = self._loss_sum / max(self._loss_count, 1)
        now = time.perf_counter()
        batch_time = (now - self._t_prev) if self._t_prev is not None else 0.0
        # HF logs once per `logging_steps` optimiser steps; divide to estimate
        # per-step wall-time.
        steps_since = max(args.logging_steps, 1)
        per_step = batch_time / steps_since
        self._t_prev = now
        self._csv.append(
            step=int(state.global_step),
            loss=loss,
            loss_avg=loss_avg,
            lr=lr,
            grad_norm=grad_norm,
            batch_time=per_step,
        )

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero or self._plotter is None:
            return
        if self.plot_interval > 0 and state.global_step > 0 and state.global_step % self.plot_interval == 0:
            self._plotter.refresh()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return
        if self._plotter is not None:
            self._plotter.refresh()
        if self._csv is not None:
            self._csv.close()


class TrackioCallback(TrainerCallback):
    """Forwards `Trainer.log` metrics to a `TrackioRun` on master rank only."""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        space_id: Optional[str] = None,
        disable: bool = False,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config
        self.space_id = space_id or None
        self.disable = disable
        self._run = None  # constructed on_train_begin (master only)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if not state.is_world_process_zero:
            return
        from train_logging import TrackioRun
        self._run = TrackioRun(
            project=self.project,
            name=self.name,
            config=self.config,
            space_id=self.space_id,
            disable=self.disable,
        )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):  # type: ignore[override]
        if self._run is None or logs is None:
            return
        # Skip eval-only logs (no "loss" key) — same filter as PngLoggingCallback.
        if "loss" not in logs:
            return
        metrics: Dict[str, float] = {}
        for src_key, dst_key in (
            ("loss", "train/loss"),
            ("learning_rate", "train/lr"),
            ("grad_norm", "train/grad_norm"),
            ("epoch", "train/epoch"),
        ):
            if src_key in logs and isinstance(logs[src_key], (int, float)):
                metrics[dst_key] = float(logs[src_key])
        if metrics:
            self._run.log(metrics, step=int(state.global_step))

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore[override]
        if self._run is not None:
            self._run.finish()
            self._run = None


# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT baseline trainer for CVLM comparison.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--model_name_or_path", type=str, default="HuggingFaceTB/SmolLM-135M-Instruct")
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--enable_warmup", action="store_true")
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval_steps", type=int, default=500,
                   help="Save every N optimizer steps. 0 = end of training only.")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Enable activation checkpointing on the decoder. "
                        "~30%% slower step, but cuts activation memory ~5-8x — "
                        "lets you fit larger per-device batch with the same VRAM.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_interval", type=int, default=100)
    p.add_argument("--csv_path", type=str, default="")
    p.add_argument("--tensorboard_dir", type=str, default="",
                   help="Defaults to <output_dir>/tb if not set.")
    p.add_argument("--trackio_project", type=str,
                   default=os.environ.get("TRACKIO_PROJECT", "cvlm"))
    p.add_argument("--trackio_run_name", type=str,
                   default=os.environ.get("TRACKIO_RUN_NAME", ""))
    p.add_argument("--trackio_space_id", type=str,
                   default=os.environ.get("TRACKIO_SPACE_ID", ""))
    p.add_argument("--trackio_disable", action="store_true",
                   default=os.environ.get("TRACKIO_DISABLE", "0") == "1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and not args.no_bf16

    print(f"[train_sft] Loading tokenizer + model: {args.model_name_or_path}")
    _apply_liger_for(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        # use_cache must be off — kv-cache and checkpointing are incompatible
        # (cache stores activations the checkpoint discards).
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    train_ds = SftFullDocDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_samples=args.max_samples,
    )
    collator = SftCollator(pad_token_id=tokenizer.pad_token_id)

    tb_dir = args.tensorboard_dir.strip() or os.path.join(args.output_dir, "tb")

    save_steps_kw: Dict[str, Any] = {}
    if args.save_interval_steps > 0:
        save_steps_kw["save_strategy"] = "steps"
        save_steps_kw["save_steps"] = args.save_interval_steps
    else:
        save_steps_kw["save_strategy"] = "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_grad_norm=args.grad_clip,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps if args.enable_warmup else 0,
        warmup_ratio=args.warmup_ratio if args.enable_warmup else 0.0,
        bf16=use_bf16,
        logging_strategy="steps",
        logging_steps=args.log_interval,
        logging_dir=tb_dir,
        report_to=[],
        seed=args.seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=3,
        **save_steps_kw,
    )

    csv_path = args.csv_path.strip() or os.path.join(args.output_dir, "metrics.csv")

    sft_run_name = args.trackio_run_name.strip() or os.path.basename(args.output_dir.rstrip("/"))
    sft_config = {
        "model_name_or_path": args.model_name_or_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "lr": args.lr,
        "max_prompt_len": args.max_prompt_len,
        "max_answer_len": args.max_answer_len,
        "seed": args.seed,
        "bf16": use_bf16,
        "warmup_steps": args.warmup_steps if args.enable_warmup else 0,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "task": "sft_baseline",
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[
            PngLoggingCallback(args.output_dir, csv_path, args.plot_interval),
            TrackioCallback(
                project=args.trackio_project,
                name=sft_run_name,
                config=sft_config,
                space_id=args.trackio_space_id,
                disable=args.trackio_disable,
            ),
        ],
    )

    print(f"[train_sft] Starting training: epochs={args.epochs} bs={args.batch_size} "
          f"grad_accum={args.gradient_accumulation_steps} lr={args.lr}")
    trainer.train()
    print("[train_sft] Saving final model to", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[train_sft] Done.")


if __name__ == "__main__":
    main()
