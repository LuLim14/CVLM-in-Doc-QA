# Evaluation script for CVLM (on-the-fly text-encoder variant).
#
# Modes:
#   cvlm          – full pipeline (text_encoder → pool → ViT → projectors → decoder)
#   baseline_llm  – bare decoder on prompt tokens only (no vision span)
#   baseline_proj – text_encoder → chunked pool → random linear projection → decoder (skip ViT)

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from cvlm_dataset import CvlmTrainDataset, make_collate_fn
from modeling import CVLM, ModelArguments, TrainingArguments, _chunked_attention_pool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CVLM (on-the-fly encoder variant)")
    p.add_argument("--checkpoint_path", type=str, default="", help="Path to model_step_*.safetensors")
    p.add_argument(
        "--sft_model_path",
        type=str,
        default="",
        help="HF directory produced by train_sft.py. Required when --mode sft.",
    )
    p.add_argument("--dataset_name", type=str, default="sggetao/PwC")
    p.add_argument("--dataset_split", type=str, default="test")
    p.add_argument("--model_name_or_path", type=str, default=None)
    p.add_argument("--vision_encoder_name", type=str, default=None)
    p.add_argument("--text_encoder_name", type=str, default=None)
    p.add_argument("--compression_rate", type=int, default=4)
    p.add_argument(
        "--num_pool_latents",
        type=int,
        default=int(os.environ.get("NUM_POOL_LATENTS", "1")),
        help="K learnable latent queries for the chunked attention pool. Must match "
        "the value used at training time so checkpoint pool_queries [K,H] loads.",
    )
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_answer_len", type=int, default=2048)
    p.add_argument("--max_vision_len", type=int, default=512)
    p.add_argument("--max_source_len", type=int, default=0, help="0 = compression_rate * max_vision_len")
    p.add_argument("--max_samples", type=int, default=0, help="Limit eval to first N samples (0 = all).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument(
        "--mode",
        type=str,
        default="cvlm",
        choices=[
            "cvlm",
            "cvlm_shuffle",  # sanity: shuffle source tokens before encoder
            "baseline_llm",  # lower bound: question only, no document
            "baseline_llm_full",  # upper bound: question + full document text
            "baseline_proj",  # random linear projection, skips ViT
            "sft",  # SFT-trained decoder loaded from --sft_model_path
        ],
    )
    p.add_argument(
        "--doc_truncation",
        type=str,
        default="none",
        choices=["none", "first", "last", "random", "oracle", "file"],
        help="baseline_llm_full/sft only: replace the document inside full_prompt "
        "with a budgeted decoder-token window before appending '\\n\\n' + question. "
        "first/last/random = contiguous window (random is deterministic: seed + "
        "sample idx); oracle = window centred on the answer's occurrence in the "
        "document (first-N fallback when not found); file = skim-then-read: the "
        "document is replaced by `selected_text` from --doc_selection_file "
        "(tools/score_chunks.py output, matched by idx; budget already applied "
        "offline, so --doc_budget_cr must NOT be passed). 'none' keeps the full "
        "document (current behaviour).",
    )
    p.add_argument(
        "--doc_selection_file",
        type=str,
        default="",
        help="JSONL from tools/score_chunks.py (one line per filtered sample: "
        '{"idx", "selected_text", ...}). Required — and only valid — with '
        "--doc_truncation=file. idx = position in the filtered dataset, so this "
        "eval must use the same dataset/split and filter flags as the scoring run "
        "(a line-count mismatch fails loudly).",
    )
    p.add_argument(
        "--doc_budget_cr",
        type=int,
        default=0,
        help="Document window budget in decoder tokens: min(ceil(L_enc/N), "
        "max_vision_len) with L_enc the source length in encoder tokens — i.e. "
        "exactly the CVLM vision-token budget at compression_rate=N. Required "
        "(>0) when --doc_truncation != none.",
    )
    p.add_argument("--compute_generation_metrics", action="store_true")
    p.add_argument(
        "--num_cached_samples",
        type=int,
        default=int(os.environ.get("NUM_CACHED_SAMPLES", "8")),
        help="When --compute_generation_metrics is set, cache the first N "
        "{document, question, reference, prediction} tuples to a "
        "*_samples.json file next to the per-ckpt eval JSON. 0 disables.",
    )
    p.add_argument("--output_json", type=str, default="")
    p.add_argument(
        "--dump_per_sample",
        type=str,
        default="",
        help="Path to a JSONL file receiving one line per evaluated sample "
        "(answer NLL / token counts; plus prediction/reference/rougeL when "
        "--compute_generation_metrics is set) for bootstrap CIs. '' disables.",
    )
    p.add_argument("--tensorboard_dir", type=str, default="")
    p.add_argument("--tb_run_name", type=str, default="")
    p.add_argument("--trackio_project", type=str, default=os.environ.get("TRACKIO_PROJECT", "cvlm"))
    p.add_argument("--trackio_run_name", type=str, default=os.environ.get("TRACKIO_RUN_NAME", ""))
    p.add_argument("--trackio_space_id", type=str, default=os.environ.get("TRACKIO_SPACE_ID", ""))
    p.add_argument("--trackio_disable", action="store_true", default=os.environ.get("TRACKIO_DISABLE", "0") == "1")
    p.add_argument(
        "--all_checkpoints",
        action="store_true",
        default=os.environ.get("EVAL_ALL_CHECKPOINTS", "0") == "1",
        help="Iterate over all model_step_*.safetensors in the checkpoint directory "
        "and log each at step=<ckpt_step> in a single trackio run.",
    )
    p.add_argument(
        "--cr_schedule",
        type=str,
        default=os.environ.get("EVAL_CR_SCHEDULE", ""),
        help="When combined with --all_checkpoints: evaluate each checkpoint at the "
        "compression_rate that was active when it was trained. Schedule format "
        "matches train_cvlm.py: 'cr:end_step,...' with last end_step=0 sentinel.",
    )
    p.add_argument(
        "--legacy_projector",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("LEGACY_PROJECTOR", "0") == "1",
        help="Must match training: omit pre-MLP RMSNorm so old checkpoints load without extra keys.",
    )
    p.add_argument(
        "--vit_init",
        type=str,
        choices=["pretrained", "random"],
        default=os.environ.get("VIT_INIT", "pretrained"),
        help="Must match training. For 'random' the checkpoint overwrites ViT weights anyway (strict=False), but passing it avoids relying on that.",
    )
    p.add_argument(
        "--freeze_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("FREEZE_VIT", "0") == "1",
        help="Must match training (frozen ViT weights are absent from the checkpoint and reload from HF).",
    )
    p.add_argument(
        "--no_vit",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("NO_VIT", "0") == "1",
        help="MUST match training for the no-ViT branch — otherwise eval silently builds a pretrained ViT and runs the latents through it.",
    )
    p.add_argument(
        "--mixer_depth_map",
        type=str,
        default=os.environ.get("MIXER_DEPTH_MAP", ""),
        help="RATE-DEPTH MIXER: MUST match training. Non-empty builds the model with the zero-gated "
        "mixer stack instead of the ViT; the active depth is looked up from --compression_rate "
        "(exact key; else nearest LOWER key; else smallest key — e.g. cr=32 with a map up to 16 "
        "uses the cr=16 depth). Env: MIXER_DEPTH_MAP.",
    )
    p.add_argument(
        "--mask_vit_padding",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("MASK_VIT_PADDING", "0") == "1",
        help="Must match training. If true, real latent queries cannot attend to padded "
        "latent slots inside ViT/mixer self-attention. Env: MASK_VIT_PADDING.",
    )
    p.add_argument(
        "--vit_apply_final_layernorm",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=os.environ.get("VIT_APPLY_FINAL_LAYERNORM", "1") == "1",
        help="Must match training. Apply the plain ViT encoder's final LayerNorm. "
        "Env: VIT_APPLY_FINAL_LAYERNORM.",
    )
    p.add_argument(
        "--force_zero_gates",
        action="store_true",
        default=os.environ.get("FORCE_ZERO_GATES", "0") == "1",
        help="RATE-DEPTH MIXER diagnostic: after loading each checkpoint, zero mixer_gates "
        "in place so the mixer collapses to the exact depth-0 (no-ViT) path. Compare the "
        "held-out metrics of the SAME checkpoint with and without this flag: if they do not "
        "get worse, the trained model was not actually using the extra depth. Requires "
        "--mixer_depth_map (errors out when the model has no mixer). The output JSON records "
        "force_zero_gates; the file name is the caller's job. Env: FORCE_ZERO_GATES=1.",
    )
    p.add_argument("--global_step", type=int, default=0)
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _encode_source_for_baseline(model: CVLM, source_ids, source_mask):
    """Run the frozen text_encoder + chunked attention pool. Used by baseline_proj.

    Uses the model's trained pool_queries so the ablation isolates "ViT vs random
    linear" rather than mixing in a different pooling rule.
    """
    with torch.no_grad():
        h = model.text_encoder(input_ids=source_ids, attention_mask=source_mask).last_hidden_state
    pooled, vmask = _chunked_attention_pool(
        h.detach(),
        source_mask,
        compression_rate=model.compression_rate,
        max_vision_len=model.max_vision_len,
        pool_queries=model.pool_queries,
    )
    return pooled, vmask


def force_zero_mixer_gates(model: Optional[CVLM]) -> int:
    """Zero the rate-depth mixer's gates IN PLACE; returns how many were zeroed.

    Collapses ``h <- h + g_i*(Layer_i(h)-h)`` to the identity at every depth, so
    the vision path becomes byte-for-byte the depth-0 / no-ViT path while every
    other trained weight (projectors, pool queries, pos-embed, mixer layers)
    stays exactly as the checkpoint left it. Evaluating one checkpoint both ways
    answers "does this model actually USE the depth it was given?" without
    having to interpret the gate magnitudes themselves.

    Raises ValueError when the model has no mixer — silently doing nothing here
    would produce an eval JSON claiming an ablation that never happened.
    """
    gates = getattr(model, "mixer_gates", None) if model is not None else None
    if gates is None:
        raise ValueError(
            "--force_zero_gates requires a model built with the rate-depth mixer "
            "(pass --mixer_depth_map exactly as used at training time); this model has no "
            "mixer_gates, so there is nothing to zero."
        )
    with torch.no_grad():
        gates.zero_()
    return int(gates.numel())


class PerSampleDumper:
    """Streams per-sample eval stats to a JSONL file (--dump_per_sample).

    Teacher-forcing lines ({idx, answer_nll_nats, answer_tokens, correct_tokens})
    are appended batch-by-batch and flushed, so an interrupted run keeps
    everything already evaluated. When generation metrics run afterwards,
    ``merge_generation`` adds prediction/reference/rougeL to the same records
    and rewrites the file with the merged lines.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.records: list[dict] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w")

    def add_tf_batch(self, nll: torch.Tensor, n_tokens: torch.Tensor, n_correct: torch.Tensor) -> None:
        for j in range(nll.size(0)):
            rec = {
                "idx": len(self.records),
                "answer_nll_nats": float(nll[j].item()),
                "answer_tokens": int(n_tokens[j].item()),
                "correct_tokens": int(n_correct[j].item()),
            }
            self.records.append(rec)
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def merge_generation(
        self, predictions: list[str], references: list[str], per_sample_rougeL: list[float]
    ) -> None:
        n = min(len(self.records), len(predictions), len(references), len(per_sample_rougeL))
        for j in range(n):
            self.records[j]["prediction"] = predictions[j]
            self.records[j]["reference"] = references[j]
            self.records[j]["rougeL"] = float(per_sample_rougeL[j])
        self._fh.close()
        self._fh = open(self.path, "w")
        for rec in self.records:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _dump_tf_batch(
    dumper: Optional[PerSampleDumper],
    per_token_loss: torch.Tensor,
    flat_mask: torch.Tensor,
    preds: torch.Tensor,
    answer_labels: torch.Tensor,
    batch_size: int,
) -> None:
    """Reduce one batch's flat per-token losses to per-sample dump stats.

    ``per_token_loss`` / ``flat_mask`` are the flattened shifted tensors the TF
    loops already compute (loss is 0 at ignore_index positions, and we mask
    anyway); ``preds`` / ``answer_labels`` are the [B, A] answer-region tensors.
    """
    if dumper is None:
        return
    loss_2d = per_token_loss.view(batch_size, -1).float()
    mask_2d = flat_mask.view(batch_size, -1)
    nll = (loss_2d * mask_2d).sum(dim=1)
    n_tokens = mask_2d.sum(dim=1)
    ans_mask = answer_labels != -100
    n_correct = ((preds == answer_labels) & ans_mask).sum(dim=1)
    dumper.add_tf_batch(nll, n_tokens, n_correct)


# ---------------------------------------------------------------------------
# Teacher-forcing metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_teacher_forcing_cvlm(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    shuffle_source: bool = False,
    dumper: Optional[PerSampleDumper] = None,
) -> Dict[str, float]:
    """Teacher-forcing eval for CVLM.

    If ``shuffle_source`` is True, permute each sample's real source tokens
    before they hit the text encoder. A well-conditioned CVLM should get
    materially worse PPL on shuffled input; a model that ignores the vision
    span will barely move.
    """
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0
    total_v_real = 0
    total_p_real = 0

    desc = "eval (cvlm_shuffle)" if shuffle_source else "eval (cvlm)"
    for batch in tqdm(loader, desc=desc):
        batch = _move(batch, device)
        source_ids = batch["source_ids"]
        source_mask = batch["source_attention_mask"]
        if shuffle_source:
            # Permute only the real positions of each sample so padding stays
            # at the tail. This preserves the token distribution but destroys
            # all order information the encoder could exploit.
            shuffled = source_ids.clone()
            for i in range(shuffled.size(0)):
                n_real = int(source_mask[i].sum().item())
                if n_real > 1:
                    perm = torch.randperm(n_real, device=shuffled.device)
                    shuffled[i, :n_real] = source_ids[i, :n_real][perm]
            source_ids = shuffled
        out = model(
            source_input_ids=source_ids,
            source_attention_mask=source_mask,
            prompt_ids=batch["prompt_ids"],
            answer_ids=batch["answer_ids"],
            answer_labels=batch["answer_labels"],
            prompt_mask=batch["prompt_mask"],
            answer_mask=batch["answer_mask"],
        )
        logits = out["logits"]
        vision_mask = out["vision_mask"]
        full_attn = out["attention_mask"]
        answer_labels = batch["answer_labels"]
        B = batch["prompt_ids"].size(0)
        P = batch["prompt_ids"].size(1)
        V = vision_mask.size(1)

        real_lens = full_attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))
        total_v_real += int(vision_mask.sum().item())
        total_p_real += int(batch["prompt_mask"].sum().item())

        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels_full = torch.cat([ignore, answer_labels], dim=1)
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels_full[:, 1:].reshape(-1)
        per_tok = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_tok[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P + V - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        _dump_tf_batch(dumper, per_tok, mask, preds, answer_labels, B)
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "vision_len_mean_seen": total_v_real / max(n_samples, 1),
        "prompt_len_mean_seen": total_p_real / max(n_samples, 1),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_baseline_llm(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    dumper: Optional[PerSampleDumper] = None,
) -> Dict[str, float]:
    """Baseline: prompt tokens + answer tokens into the frozen decoder, no vision span."""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0

    for batch in tqdm(loader, desc="eval (baseline_llm)"):
        batch = _move(batch, device)
        prompt_ids = batch["prompt_ids"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        prompt_mask = batch["prompt_mask"]
        answer_mask = batch["answer_mask"]
        B = prompt_ids.size(0)
        P = prompt_ids.size(1)

        attn = torch.cat([prompt_mask, answer_mask], dim=1)
        base_real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(base_real_lens)
        max_dec_len = max(max_dec_len, max(base_real_lens))

        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(prompt_ids)
        answer_embs = embed_layer(answer_ids)
        decoder_input = torch.cat([prompt_embs, answer_embs], dim=1)

        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        _dump_tf_batch(dumper, per_token_loss, mask, preds, answer_labels, B)
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_baseline_llm_full(
    model: CVLM,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    dumper: Optional[PerSampleDumper] = None,
) -> Dict[str, float]:
    """Oracle baseline: full (document + question) text as the decoder prompt.

    Uses ``full_prompt_ids`` from the collator (document + blank line + question).
    No vision span. This is the uncompressed upper bound the CVLM must approach
    while using far fewer tokens.
    """
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0
    total_fp_real = 0

    for batch in tqdm(loader, desc="eval (baseline_llm_full)"):
        batch = _move(batch, device)
        if "full_prompt_ids" not in batch:
            raise RuntimeError(
                "baseline_llm_full needs `full_prompt_ids` from the collator; "
                "re-generate the dataset with the updated cvlm_dataset.py."
            )
        prompt_ids = batch["full_prompt_ids"]
        prompt_mask = batch["full_prompt_mask"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        answer_mask = batch["answer_mask"]
        B = prompt_ids.size(0)
        P = prompt_ids.size(1)

        attn = torch.cat([prompt_mask, answer_mask], dim=1)
        real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))
        total_fp_real += int(prompt_mask.sum().item())

        embed_layer = model.decoder.get_input_embeddings()
        decoder_input = torch.cat([embed_layer(prompt_ids), embed_layer(answer_ids)], dim=1)

        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_tok = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_tok[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        _dump_tf_batch(dumper, per_tok, mask, preds, answer_labels, B)
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "full_prompt_len_mean_seen": total_fp_real / max(n_samples, 1),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_sft(
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    dumper: Optional[PerSampleDumper] = None,
) -> Dict[str, float]:
    """SFT-trained decoder eval: same input format as baseline_llm_full
    (`full_prompt_ids` = document + blank line + question, then answer), but
    runs against a bare HF causal LM rather than the CVLM wrapper's frozen
    decoder. Output schema matches baseline_llm_full for direct comparison.
    """
    decoder.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0
    total_fp_real = 0

    for batch in tqdm(loader, desc="eval (sft)"):
        batch = _move(batch, device)
        if "full_prompt_ids" not in batch:
            raise RuntimeError(
                "sft eval needs `full_prompt_ids` from the collator; "
                "the dataset must be CvlmTrainDataset (already produces this)."
            )
        prompt_ids = batch["full_prompt_ids"]
        prompt_mask = batch["full_prompt_mask"]
        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        answer_mask = batch["answer_mask"]
        B = prompt_ids.size(0)
        P = prompt_ids.size(1)

        attn = torch.cat([prompt_mask, answer_mask], dim=1)
        real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))
        total_fp_real += int(prompt_mask.sum().item())

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        ignore = torch.full((B, P), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        # SFT was trained with right-pad (positions [0..N-1] for real tokens),
        # but the collator left-pads `full_prompt_ids`. Without explicit
        # position_ids, HF defaults to arange(seq_len) regardless of
        # attention_mask, which puts real tokens at shifted RoPE phases vs
        # training. Derive position_ids from the attention mask so real tokens
        # see the same positions they did during SFT training.
        position_ids = attn.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn == 0, 1)

        out = decoder(
            input_ids=input_ids,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_tok = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_tok[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        _dump_tf_batch(dumper, per_tok, mask, preds, answer_labels, B)
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "full_prompt_len_mean_seen": total_fp_real / max(n_samples, 1),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


@torch.no_grad()
def eval_teacher_forcing_baseline_proj(
    model: CVLM,
    proj: nn.Linear,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    dumper: Optional[PerSampleDumper] = None,
) -> Dict[str, float]:
    """Baseline: text encoder → chunked pool → random linear projection → decoder (skip ViT)."""
    model.eval()
    proj.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_samples = 0
    total_dec_len = 0
    max_dec_len = 0

    for batch in tqdm(loader, desc="eval (baseline_proj)"):
        batch = _move(batch, device)
        B = batch["prompt_ids"].size(0)
        pooled, vision_mask = _encode_source_for_baseline(model, batch["source_ids"], batch["source_attention_mask"])
        vision_embeds = proj(pooled)  # [B, V, llm_dim]

        embed_layer = model.decoder.get_input_embeddings()
        prompt_embs = embed_layer(batch["prompt_ids"])
        answer_embs = embed_layer(batch["answer_ids"])
        decoder_input = torch.cat([prompt_embs, vision_embeds, answer_embs], dim=1)

        attn = torch.cat([batch["prompt_mask"], vision_mask, batch["answer_mask"]], dim=1)
        P = batch["prompt_ids"].size(1)
        V = vision_embeds.size(1)
        answer_labels = batch["answer_labels"]
        ignore = torch.full((B, P + V), -100, dtype=answer_labels.dtype, device=device)
        labels = torch.cat([ignore, answer_labels], dim=1)

        real_lens = attn.sum(dim=1).tolist()
        total_dec_len += sum(real_lens)
        max_dec_len = max(max_dec_len, max(real_lens))

        out = model.decoder(inputs_embeds=decoder_input, attention_mask=attn, use_cache=False)
        logits = out.logits

        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        per_token_loss = loss_fct(shift_logits, shift_labels)
        mask = shift_labels != -100
        total_loss += per_token_loss[mask].sum().item()
        total_tokens += int(mask.sum().item())

        answer_logits = logits[:, P + V - 1 : -1, :]
        preds = answer_logits.argmax(dim=-1)
        ans_mask = answer_labels != -100
        total_correct += (preds[ans_mask] == answer_labels[ans_mask]).sum().item()
        _dump_tf_batch(dumper, per_token_loss, mask, preds, answer_labels, B)
        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "perplexity": math.exp(min(avg_loss, 30)),
        "token_accuracy": total_correct / max(total_tokens, 1),
        "avg_loss": avg_loss,
        "n_samples": n_samples,
        "decoder_input_len_mean": total_dec_len / max(n_samples, 1),
        "decoder_input_len_max": int(max_dec_len),
        "total_answer_nll_nats": total_loss,
        "total_answer_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_answers(
    model: Optional[CVLM],
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int,
    max_samples: int,
    mode: str,
    proj: Optional[nn.Linear] = None,
    sft_decoder: Optional[nn.Module] = None,
    sft_tokenizer=None,
    num_cached_samples: int = 0,
    encoder_tokenizer=None,
) -> tuple[list[str], list[str], list[dict]]:
    if mode == "sft":
        assert sft_decoder is not None and sft_tokenizer is not None
        sft_decoder.eval()
        tokenizer = sft_tokenizer
    else:
        assert model is not None
        model.eval()
        tokenizer = model.tokenizer
    predictions: list[str] = []
    references: list[str] = []
    cached_samples: list[dict] = []
    n_samples = 0

    for batch in tqdm(loader, desc=f"generate ({mode})"):
        batch = _move(batch, device)
        B = batch["prompt_ids"].size(0)
        prompt_mask = batch["prompt_mask"]

        if mode == "cvlm" or mode == "cvlm_shuffle":
            src_ids = batch["source_ids"]
            if mode == "cvlm_shuffle":
                src_mask = batch["source_attention_mask"]
                shuffled = src_ids.clone()
                for i in range(shuffled.size(0)):
                    n_real = int(src_mask[i].sum().item())
                    if n_real > 1:
                        perm = torch.randperm(n_real, device=shuffled.device)
                        shuffled[i, :n_real] = src_ids[i, :n_real][perm]
                src_ids = shuffled
            gen_ids = model.generate(
                source_input_ids=src_ids,
                source_attention_mask=batch["source_attention_mask"],
                prompt_ids=batch["prompt_ids"],
                prompt_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
            )
        elif mode == "baseline_llm":
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            gen_ids = model.decoder.generate(
                inputs_embeds=prompt_embs,
                attention_mask=prompt_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        elif mode == "baseline_llm_full":
            fp_ids = batch["full_prompt_ids"]
            fp_mask = batch["full_prompt_mask"]
            prompt_embs = model.decoder.get_input_embeddings()(fp_ids)
            gen_ids = model.decoder.generate(
                inputs_embeds=prompt_embs,
                attention_mask=fp_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        elif mode == "sft":
            fp_ids = batch["full_prompt_ids"]
            fp_mask = batch["full_prompt_mask"]
            gen_ids = sft_decoder.generate(
                input_ids=fp_ids,
                attention_mask=fp_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            # generate() returns the prompt + new tokens; trim the prompt.
            gen_ids = gen_ids[:, fp_ids.size(1) :]
        elif mode == "baseline_proj":
            assert proj is not None
            pooled, vision_mask = _encode_source_for_baseline(
                model, batch["source_ids"], batch["source_attention_mask"]
            )
            vision_embeds = proj(pooled)
            prompt_embs = model.decoder.get_input_embeddings()(batch["prompt_ids"])
            decoder_input = torch.cat([prompt_embs, vision_embeds], dim=1)
            attn = torch.cat([prompt_mask, vision_mask], dim=1)
            gen_ids = model.decoder.generate(
                inputs_embeds=decoder_input,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(pred_texts)

        answer_ids = batch["answer_ids"]
        answer_labels = batch["answer_labels"]
        ref_texts: list[str] = []
        for i in range(B):
            m = answer_labels[i] != -100
            ref_ids = answer_ids[i][m]
            ref_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))
        references.extend(ref_texts)

        # Cache the first N (document, question, reference, prediction) tuples
        # so eval results are inspectable without re-running generation.
        if num_cached_samples > 0 and len(cached_samples) < num_cached_samples:
            # Decode the question using the decoder tokenizer; decode the document
            # using the encoder tokenizer when available (source_ids live in
            # encoder vocab), else fall back to decoder tokenizer.
            prompt_ids = batch["prompt_ids"]
            prompt_mask = batch["prompt_mask"]
            doc_ids = batch.get("source_ids", None)
            doc_mask = batch.get("source_attention_mask", None)
            doc_tok = encoder_tokenizer if encoder_tokenizer is not None else tokenizer
            for i in range(B):
                if len(cached_samples) >= num_cached_samples:
                    break
                q_real = prompt_ids[i][prompt_mask[i].bool()]
                question_text = tokenizer.decode(q_real, skip_special_tokens=True)
                if doc_ids is not None and doc_mask is not None:
                    d_real = doc_ids[i][doc_mask[i].bool()]
                    document_text = doc_tok.decode(d_real, skip_special_tokens=True)
                else:
                    document_text = ""
                cached_samples.append(
                    {
                        "index": n_samples + i,
                        "document": document_text,
                        "question": question_text,
                        "reference": ref_texts[i],
                        "prediction": pred_texts[i],
                    }
                )

        n_samples += B
        if 0 < max_samples <= n_samples:
            break

    if max_samples > 0:
        return predictions[:max_samples], references[:max_samples], cached_samples
    return predictions, references, cached_samples


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
    per_sample_rougeL: Optional[list[float]] = None,
) -> Dict[str, float]:
    import sacrebleu
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    if per_sample_rougeL is not None:
        per_sample_rougeL.extend(rl)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    n_exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return {
        "rouge1": sum(r1) / len(r1) if r1 else 0.0,
        "rouge2": sum(r2) / len(r2) if r2 else 0.0,
        "rougeL": sum(rl) / len(rl) if rl else 0.0,
        "bleu4": bleu.score,
        "exact_match": n_exact / len(predictions) if predictions else 0.0,
    }


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------


def compute_compression_stats(
    dataset: CvlmTrainDataset,
    decoder_tokenizer,
    compression_rate: int,
    max_vision_len: int,
    max_samples: int,
) -> Dict[str, float]:
    """Per-sample source-token and vision-length stats for the on-the-fly pipeline.

    Definitions
    -----------
    S_i : int
        Number of tokens obtained by tokenizing ``record["input"]`` with the
        *decoder's* tokenizer (not ModernBERT's). This makes the number
        directly comparable with the token budget a plain-LLM baseline would
        consume.
    V_i : int
        Number of compressed vision slots the decoder actually sees for this
        sample. Derived as ``min(ceil(L_enc_i / compression_rate), max_vision_len)``
        where ``L_enc_i`` is the source length in encoder tokens (after the
        dataset filter, before truncation).
    P_i, A_i : int
        Prompt / answer decoder-token counts (tracked by the per-mode eval
        loops via ``prompt_len_mean_seen`` / ``answer`` side of the mask).

    Metrics returned
    ----------------
    source_tokens_mean, source_tokens_sum
        Mean and total of S_i across the evaluated samples. ``_sum`` is the
        denominator used by ``bits_per_source_token``.
    vision_len_mean, vision_len_median, vision_len_min, vision_len_max
        Distribution of V_i. Sanity checks that the compression stage is
        neither collapsing to 1 nor saturating at ``max_vision_len``.
    compression_ratio_mean, _median, _p10, _p90, _min, _max
        Distribution of ``S_i / V_i``. With ``--compression_rate K`` the mean
        should sit near K (it can differ because S_i uses the LLM tokenizer
        while V_i is derived from the encoder tokenizer's length).
    n_compression_samples : int
        Number of samples the stats were computed over (≤ ``max_samples``).

    Related joint metrics computed in ``main`` (not here):
    - ``decoder_input_len_mean`` / ``_max`` : mean / max of real P_i+V_i+A_i.
    - ``effective_context_reduction`` : S_i / (V_i + P_i).
    - ``bits_per_source_token`` : (answer NLL in nats / ln 2) / sum(S_i).
      Lower = fewer bits of answer-side cross-entropy per unit of source.
    """
    enc_tok = dataset._enc_tok
    S: list[int] = []
    V: list[int] = []
    n = len(dataset)
    limit = n if max_samples <= 0 else min(n, max_samples)
    cr = max(int(compression_rate), 1)
    for idx in tqdm(range(limit), desc="compression stats"):
        row_id = dataset._row_indices[idx]
        text = dataset._hf[row_id]["input"]
        s_len = len(decoder_tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        l_enc = len(enc_tok(text, add_special_tokens=False, truncation=False)["input_ids"])
        l_enc = min(l_enc, dataset.max_source_len)
        v_len = min((l_enc + cr - 1) // cr, max_vision_len)
        if v_len == 0:
            v_len = 1
        S.append(s_len)
        V.append(v_len)
    S_arr = np.asarray(S, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64).clip(min=1.0)
    ratio = S_arr / V_arr
    return {
        "source_tokens_mean": float(S_arr.mean()),
        "source_tokens_sum": float(S_arr.sum()),
        "vision_len_mean": float(V_arr.mean()),
        "vision_len_median": float(np.median(V_arr)),
        "vision_len_min": float(V_arr.min()),
        "vision_len_max": float(V_arr.max()),
        "compression_ratio_mean": float(ratio.mean()),
        "compression_ratio_median": float(np.median(ratio)),
        "compression_ratio_p10": float(np.percentile(ratio, 10)),
        "compression_ratio_p90": float(np.percentile(ratio, 90)),
        "compression_ratio_min": float(ratio.min()),
        "compression_ratio_max": float(ratio.max()),
        "n_compression_samples": int(len(S_arr)),
    }


def bits_per_source_token(total_answer_nll_nats: float, total_source_tokens: int) -> float:
    """bits_per_source_token = (sum answer NLL in nats / ln 2) / sum(S_i).

    Joint quality/compression score. Expresses the model's answer-side
    cross-entropy in bits, normalised by the number of source-text tokens the
    compressed context stood in for. Lower is better, and it is directly
    comparable across runs with different compression rates (whereas raw
    perplexity is only comparable at a fixed sequence layout).
    """
    if total_source_tokens <= 0:
        return 0.0
    return (total_answer_nll_nats / math.log(2.0)) / float(total_source_tokens)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.mode == "sft" and args.all_checkpoints:
        raise ValueError(
            "--all_checkpoints is incompatible with --mode sft: SFT eval uses "
            "--sft_model_path (an HF directory) rather than model_step_*.safetensors."
        )
    if args.doc_truncation != "none":
        if args.mode not in ("baseline_llm_full", "sft"):
            raise ValueError(
                f"--doc_truncation only applies to modes that consume full_prompt_ids "
                f"(baseline_llm_full / sft); got --mode {args.mode}"
            )
        if args.doc_truncation == "file":
            if not args.doc_selection_file:
                raise ValueError("--doc_truncation=file requires --doc_selection_file <score_chunks JSONL>")
            if args.doc_budget_cr > 0:
                raise ValueError(
                    "--doc_truncation=file: the token budget was applied offline by "
                    "tools/score_chunks.py (--budget_cr); do not pass --doc_budget_cr here."
                )
        elif args.doc_budget_cr <= 0:
            raise ValueError("--doc_truncation != none requires --doc_budget_cr > 0")
    elif args.doc_budget_cr > 0:
        raise ValueError("--doc_budget_cr > 0 requires --doc_truncation != none")
    if args.doc_selection_file and args.doc_truncation != "file":
        raise ValueError("--doc_selection_file is only valid with --doc_truncation=file")
    if args.force_zero_gates:
        # Fail before loading anything: the flag is only meaningful for the modes
        # that actually run the mixer, and only for mixer-built models.
        if args.mode not in ("cvlm", "cvlm_shuffle"):
            raise ValueError(
                f"--force_zero_gates only applies to modes that run the vision path through the "
                f"rate-depth mixer (cvlm / cvlm_shuffle); got --mode {args.mode}"
            )
        if not str(args.mixer_depth_map or "").strip():
            raise ValueError(
                "--force_zero_gates requires --mixer_depth_map (the same map used at training "
                "time); without it the model is built with a plain ViT and has no gates."
            )

    # Optional per-checkpoint cr lookup: when --cr_schedule is set, each ckpt
    # is evaluated at the cr that produced its weights. Requires --all_checkpoints
    # and a cr-aware mode. Schedule helpers come from train_cvlm.py.
    cr_schedule = None
    if args.cr_schedule:
        if not args.all_checkpoints:
            raise ValueError("--cr_schedule requires --all_checkpoints")
        if args.mode not in ("cvlm", "cvlm_shuffle", "baseline_proj"):
            raise ValueError(
                f"--cr_schedule only meaningful for cr-dependent modes "
                f"(cvlm/cvlm_shuffle/baseline_proj); got --mode {args.mode}"
            )
        from train_cvlm import _parse_cr_schedule, _training_cr_for_step

        cr_schedule = _parse_cr_schedule(args.cr_schedule)
        print(f"\n=== --cr_schedule active: {cr_schedule} ===")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and not args.no_bf16

    model_args = ModelArguments(train=False)
    if args.model_name_or_path:
        model_args.model_name_or_path = args.model_name_or_path
    if args.vision_encoder_name:
        model_args.vision_encoder_name = args.vision_encoder_name
    if args.text_encoder_name:
        model_args.text_encoder_name = args.text_encoder_name
    model_args.max_vision_len = args.max_vision_len
    model_args.compression_rate = args.compression_rate
    model_args.num_pool_latents = max(int(args.num_pool_latents), 1)
    model_args.projector_use_input_rmsnorm = not bool(args.legacy_projector)
    model_args.vit_init = str(args.vit_init)
    model_args.vit_freeze = bool(args.freeze_vit)
    model_args.vit_skip = bool(args.no_vit)
    model_args.mixer_depth_map = str(args.mixer_depth_map or "")
    model_args.mask_vit_padding = bool(args.mask_vit_padding)
    model_args.vit_apply_final_layernorm = bool(args.vit_apply_final_layernorm)

    training_args = TrainingArguments(output_dir="/tmp/eval_cvlm_dummy")
    training_args.bf16 = bool(use_bf16)

    sft_decoder: Optional[nn.Module] = None
    sft_tokenizer = None
    model: Optional[CVLM] = None

    if args.mode == "sft":
        if not args.sft_model_path:
            raise ValueError("--mode sft requires --sft_model_path <hf_dir>")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading SFT model from {args.sft_model_path}")
        sft_dtype = torch.bfloat16 if use_bf16 else torch.float32
        sft_decoder = AutoModelForCausalLM.from_pretrained(
            args.sft_model_path, torch_dtype=sft_dtype, use_safetensors=True
        ).to(device)
        sft_decoder.eval()
        sft_tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, use_fast=False)
        if sft_tokenizer.pad_token is None:
            sft_tokenizer.pad_token = sft_tokenizer.eos_token
        # Ensure model_args.model_name_or_path is set to the SFT dir so the
        # CvlmTrainDataset uses the matching decoder tokenizer for filtering
        # (encoder tokenizer still comes from text_encoder_name).
        model_args.model_name_or_path = args.sft_model_path
    else:
        print(f"Loading model ({model_args.model_name_or_path})...")
        model = CVLM(model_args, training_args)
        model.to(device)

        if args.checkpoint_path and not args.all_checkpoints:
            print(f"Loading checkpoint: {args.checkpoint_path}")
            state_dict = load_file(args.checkpoint_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)}")

        model.eval()

    proj: Optional[nn.Linear] = None
    if args.mode == "baseline_proj":
        assert model is not None
        embed_dim = model.text_encoder.config.hidden_size
        llm_dim = model.decoder.config.hidden_size
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        proj = nn.Linear(embed_dim, llm_dim).to(device=device, dtype=dtype)
        proj.eval()
        print(f"baseline_proj: Linear({embed_dim} → {llm_dim}), random init")

    # Pad ids: in SFT mode there's no CVLM wrapper, so look them up directly.
    if args.mode == "sft":
        from transformers import AutoTokenizer as _AutoTok

        enc_tok_for_pad = _AutoTok.from_pretrained(model_args.text_encoder_name, use_fast=True, trust_remote_code=True)
        if enc_tok_for_pad.pad_token is None:
            enc_tok_for_pad.pad_token = enc_tok_for_pad.eos_token or enc_tok_for_pad.cls_token
        dec_pad = sft_tokenizer.pad_token_id
        enc_pad = enc_tok_for_pad.pad_token_id
    else:
        assert model is not None
        dec_pad = model.tokenizer.pad_token_id
        enc_pad = model.encoder_tokenizer.pad_token_id
    if dec_pad is None or enc_pad is None:
        raise ValueError("Both decoder and encoder tokenizers must define pad_token")
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
        doc_truncation=args.doc_truncation,
        doc_budget_cr=args.doc_budget_cr,
        doc_budget_cap=args.max_vision_len,
        doc_truncation_seed=args.seed,
        doc_selection_file=args.doc_selection_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    # Compression stats are weight-independent at fixed cr. With --cr_schedule
    # the cr varies per checkpoint, so we cache stats keyed by cr and look them
    # up inside the per-checkpoint loop. Without a schedule, the cache only
    # ever holds one entry (the static --compression_rate) and we precompute it.
    decoder_tok_for_stats = sft_tokenizer if args.mode == "sft" else model.tokenizer  # type: ignore[union-attr]
    comp_stats_cache: Dict[int, Dict[str, float]] = {}

    def _get_comp_stats(cr_val: int) -> Dict[str, float]:
        if cr_val not in comp_stats_cache:
            print(f"\n=== Compression stats (cr={cr_val}) ===")
            stats = compute_compression_stats(
                dataset,
                decoder_tok_for_stats,
                compression_rate=cr_val,
                max_vision_len=args.max_vision_len,
                max_samples=args.max_samples,
            )
            for k in [
                "source_tokens_mean",
                "vision_len_mean",
                "vision_len_median",
                "vision_len_min",
                "vision_len_max",
                "compression_ratio_mean",
                "compression_ratio_median",
                "compression_ratio_p10",
                "compression_ratio_p90",
                "compression_ratio_min",
                "compression_ratio_max",
                "n_compression_samples",
            ]:
                print(f"  {k:28s}: {stats[k]}")
            comp_stats_cache[cr_val] = stats
        return comp_stats_cache[cr_val]

    if cr_schedule is None:
        comp_stats = _get_comp_stats(int(args.compression_rate))
    else:
        comp_stats = None  # populated per-ckpt inside the loop

    # Open the per-mode trackio run ONCE — all checkpoints log into it.
    run = None
    run_name = ""
    if not args.trackio_disable:
        from train_logging import TrackioRun

        # Always suffix with _{mode} so each eval mode gets its own trackio
        # run when sharing a base TRACKIO_RUN_NAME across cvlm/baseline_*.
        base_run = args.trackio_run_name.strip() or "eval"
        suffix = f"_{args.mode}"
        run_name = base_run if base_run.endswith(suffix) else base_run + suffix
        eval_config = {
            "mode": args.mode,
            "model_name_or_path": args.model_name_or_path,
            "text_encoder_name": args.text_encoder_name,
            "compression_rate": args.compression_rate,
            "max_prompt_len": args.max_prompt_len,
            "max_answer_len": args.max_answer_len,
            "max_vision_len": args.max_vision_len,
            "max_source_len": args.max_source_len,
            "doc_truncation": args.doc_truncation,
            "doc_budget_cr": int(args.doc_budget_cr),
            "doc_selection_file": args.doc_selection_file,
            "checkpoint_path": getattr(args, "checkpoint_path", "") or getattr(args, "sft_model_path", ""),
            "global_step": int(args.global_step),
            "all_checkpoints": bool(args.all_checkpoints),
            "mixer_depth_map": str(args.mixer_depth_map or ""),
            "mask_vit_padding": bool(args.mask_vit_padding),
            "vit_apply_final_layernorm": bool(args.vit_apply_final_layernorm),
            "force_zero_gates": bool(args.force_zero_gates),
        }
        run = TrackioRun(
            project=args.trackio_project,
            name=run_name,
            config=eval_config,
            space_id=args.trackio_space_id or None,
        )

    # Resolve the (step, path) list to iterate over.
    if args.all_checkpoints:
        ckpt_dir = os.path.dirname(args.checkpoint_path) if args.checkpoint_path else None
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            raise ValueError(
                f"--all_checkpoints requires --checkpoint_path pointing inside a directory; "
                f"got {args.checkpoint_path!r}"
            )
        paths = sorted(
            glob.glob(os.path.join(ckpt_dir, "model_step_*.safetensors")),
            key=lambda p: int(re.search(r"model_step_(\d+)\.safetensors$", p).group(1)),
        )
        if not paths:
            raise ValueError(f"No model_step_*.safetensors found in {ckpt_dir}")
        ckpt_pairs = [(int(re.search(r"model_step_(\d+)\.safetensors$", p).group(1)), p) for p in paths]
        print(f"\n=== --all_checkpoints: {len(ckpt_pairs)} checkpoints under {ckpt_dir} ===")
    else:
        ckpt_pairs = [(int(args.global_step), args.checkpoint_path)]

    last_results: Dict = {}
    for ckpt_step, ckpt_path in ckpt_pairs:
        if args.all_checkpoints and ckpt_path:
            print(f"\n[ckpt step={ckpt_step}] loading {ckpt_path}")
            assert model is not None
            state_dict = load_file(ckpt_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"  loaded; missing={len(missing)} unexpected={len(unexpected)}")
            model.eval()

        if args.force_zero_gates:
            # AFTER the checkpoint load (the checkpoint carries the trained
            # gates), and re-applied for every checkpoint in --all_checkpoints.
            n_gates = force_zero_mixer_gates(model)
            print(f"  [force_zero_gates] zeroed {n_gates} mixer gate(s) -> exact depth-0 (no-ViT) path")

        # Per-ckpt cr: look up training cr from schedule, set model accordingly,
        # and pull (or compute) compression stats for this cr.
        if cr_schedule is not None:
            assert model is not None
            eval_cr = _training_cr_for_step(cr_schedule, int(ckpt_step))  # noqa: F821
            print(f"  [cr_schedule] step={ckpt_step} → eval at cr={eval_cr}")
            model.set_compression_rate(eval_cr)
            comp_stats = _get_comp_stats(eval_cr)
        else:
            eval_cr = int(args.compression_rate)
        assert comp_stats is not None

        # Per-sample JSONL dump for bootstrap CIs. With --all_checkpoints each
        # checkpoint gets its own file (same suffix scheme as --output_json).
        dumper: Optional[PerSampleDumper] = None
        if args.dump_per_sample:
            if args.all_checkpoints:
                dbase, dext = os.path.splitext(args.dump_per_sample)
                dext = dext or ".jsonl"
                if cr_schedule is not None:
                    dump_path = f"{dbase}_native_step{ckpt_step}{dext}"
                else:
                    dump_path = f"{dbase}_step{ckpt_step}{dext}"
            else:
                dump_path = args.dump_per_sample
            dumper = PerSampleDumper(dump_path)

        print(f"\n=== Teacher-forcing evaluation (mode={args.mode}, step={ckpt_step}) ===")
        if args.mode == "cvlm":
            assert model is not None
            tf_metrics = eval_teacher_forcing_cvlm(model, loader, device, args.max_samples, dumper=dumper)
        elif args.mode == "cvlm_shuffle":
            assert model is not None
            tf_metrics = eval_teacher_forcing_cvlm(
                model, loader, device, args.max_samples, shuffle_source=True, dumper=dumper
            )
        elif args.mode == "baseline_llm":
            assert model is not None
            tf_metrics = eval_teacher_forcing_baseline_llm(model, loader, device, args.max_samples, dumper=dumper)
        elif args.mode == "baseline_llm_full":
            assert model is not None
            tf_metrics = eval_teacher_forcing_baseline_llm_full(model, loader, device, args.max_samples, dumper=dumper)
        elif args.mode == "baseline_proj":
            assert model is not None and proj is not None
            tf_metrics = eval_teacher_forcing_baseline_proj(model, proj, loader, device, args.max_samples, dumper=dumper)
        elif args.mode == "sft":
            assert sft_decoder is not None
            tf_metrics = eval_teacher_forcing_sft(sft_decoder, loader, device, args.max_samples, dumper=dumper)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        tf_metrics["mode"] = args.mode
        print(f"  Perplexity:      {tf_metrics['perplexity']:.4f}")
        print(f"  Token Accuracy:  {tf_metrics['token_accuracy']:.4f}")
        print(f"  Avg Loss:        {tf_metrics['avg_loss']:.4f}")
        print(f"  Samples:         {tf_metrics['n_samples']}")
        print(
            f"  Decoder in-len mean/max: "
            f"{tf_metrics.get('decoder_input_len_mean', 0):.1f} / "
            f"{tf_metrics.get('decoder_input_len_max', 0)}"
        )

        total_nll = float(tf_metrics.get("total_answer_nll_nats", 0.0))
        total_src = int(comp_stats.get("source_tokens_sum", 0))
        bps = bits_per_source_token(total_nll, total_src)
        # effective_context_reduction = source_tokens_mean / (decoder tokens actually
        # consumed besides the answer). >1 means CVLM uses fewer decoder tokens than
        # the full-text baseline would; ==1 is breakeven; <1 is augmentation.
        if args.mode in ("cvlm", "cvlm_shuffle"):
            mean_p = float(tf_metrics.get("prompt_len_mean_seen", 0.0))
            mean_v = float(tf_metrics.get("vision_len_mean_seen", 0.0))
            denom = mean_v + mean_p
        elif args.mode in ("baseline_llm_full", "sft"):
            denom = float(tf_metrics.get("full_prompt_len_mean_seen", 0.0))
        elif args.mode == "baseline_llm":
            # question-only; "context" for document tokens is effectively zero, so
            # the ratio is undefined — leave as 0 for clarity in the JSON.
            denom = 0.0
        else:
            denom = 0.0
        eff_reduction = comp_stats["source_tokens_mean"] / denom if denom > 0 else 0.0
        print(f"\n  bits_per_source_token       : {bps:.6f}")
        print(f"  effective_context_reduction : {eff_reduction:.4f}  (source_tokens_mean / context_tokens_used)")

        gen_metrics: Dict[str, float] = {}
        cached_samples: list[dict] = []
        if args.compute_generation_metrics:
            print(f"\n=== Generation evaluation (mode={args.mode}, step={ckpt_step}) ===")
            enc_tok_for_cache = None
            if args.mode == "sft":
                enc_tok_for_cache = enc_tok_for_pad  # already loaded above for SFT
            elif model is not None:
                enc_tok_for_cache = getattr(model, "encoder_tokenizer", None)
            preds, refs, cached_samples = generate_answers(
                model,
                loader,
                device,
                args.max_new_tokens,
                args.max_samples,
                args.mode,
                proj=proj,
                sft_decoder=sft_decoder,
                sft_tokenizer=sft_tokenizer,
                num_cached_samples=max(int(args.num_cached_samples), 0),
                encoder_tokenizer=enc_tok_for_cache,
            )
            per_sample_rl: list[float] = []
            gen_metrics = compute_generation_metrics(
                preds, refs, per_sample_rougeL=per_sample_rl if dumper is not None else None
            )
            if dumper is not None:
                dumper.merge_generation(preds, refs, per_sample_rl)
            print(f"  ROUGE-1:       {gen_metrics['rouge1']:.4f}")
            print(f"  ROUGE-2:       {gen_metrics['rouge2']:.4f}")
            print(f"  ROUGE-L:       {gen_metrics['rougeL']:.4f}")
            print(f"  BLEU-4:        {gen_metrics['bleu4']:.2f}")
            print(f"  Exact Match:   {gen_metrics['exact_match']:.4f}")

        if dumper is not None:
            dumper.close()
            print(f"  wrote per-sample dump: {dumper.path} ({len(dumper.records)} lines)")

        # oracle_hit_rate: fraction of evaluated samples whose answer (or an
        # 8-gram of it) was found verbatim in the document. Only meaningful for
        # --doc_truncation oracle; null in the JSON otherwise.
        oracle_hit_rate: Optional[float] = None
        if args.doc_truncation == "oracle":
            hits = dataset._oracle_hits
            oracle_hit_rate = sum(1 for h in hits.values() if h) / max(len(hits), 1)
            print(f"  oracle_hit_rate: {oracle_hit_rate:.4f} ({len(hits)} samples)")

        results = {
            **tf_metrics,
            **gen_metrics,
            **comp_stats,
            "bits_per_source_token": bps,
            "effective_context_reduction": eff_reduction,
            "compression_rate": int(eval_cr),
            "doc_truncation": args.doc_truncation,
            "doc_budget_cr": int(args.doc_budget_cr),
            "doc_selection_file": args.doc_selection_file,
            "oracle_hit_rate": oracle_hit_rate,
            "mixer_depth_map": str(args.mixer_depth_map or ""),
            "mask_vit_padding": bool(args.mask_vit_padding),
            "vit_apply_final_layernorm": bool(args.vit_apply_final_layernorm),
            "force_zero_gates": bool(args.force_zero_gates),
        }
        last_results = results

        # Per-checkpoint JSON. With --cr_schedule we suffix the basename with
        # `_native` so we don't clobber prior fixed-cr eval JSONs alongside.
        if args.output_json:
            if args.all_checkpoints:
                base, ext = os.path.splitext(args.output_json)
                ext = ext or ".json"
                if cr_schedule is not None:
                    step_json = f"{base}_native_step{ckpt_step}{ext}"
                else:
                    step_json = f"{base}_step{ckpt_step}{ext}"
            else:
                step_json = args.output_json
            os.makedirs(os.path.dirname(step_json) or ".", exist_ok=True)
            with open(step_json, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  wrote {step_json}")

            if cached_samples:
                base2, ext2 = os.path.splitext(step_json)
                samples_path = f"{base2}_samples{ext2}"
                with open(samples_path, "w") as f:
                    json.dump(
                        {
                            "mode": args.mode,
                            "step": int(ckpt_step),
                            "compression_rate": int(eval_cr),
                            "n_cached": len(cached_samples),
                            "samples": cached_samples,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"  wrote {samples_path} ({len(cached_samples)} samples)")

        # Log this checkpoint's metrics into the per-mode trackio run.
        if run is not None:
            flat_metrics = {
                f"eval/{k}": float(v)
                for k, v in results.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if flat_metrics:
                run.log(flat_metrics, step=int(ckpt_step))

    # Histogram is weight-independent — log once after the loop, at the LAST
    # checkpoint's step so it lands on the final point of the scatter chart.
    # Skip in --cr_schedule mode: cr varies per ckpt, a single global histogram
    # would mix distributions and mislead.
    if run is not None:
        if cr_schedule is None:
            per_sample_ratios = []
            limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
            cr = max(int(args.compression_rate), 1)
            enc_tok = dataset._enc_tok
            dec_tok_hist = sft_tokenizer if args.mode == "sft" else model.tokenizer  # type: ignore[union-attr]
            for idx in range(limit):
                row_id = dataset._row_indices[idx]
                text = dataset._hf[row_id]["input"]
                s_len = len(dec_tok_hist(text, add_special_tokens=False, truncation=False)["input_ids"])
                l_enc = min(
                    len(enc_tok(text, add_special_tokens=False, truncation=False)["input_ids"]), dataset.max_source_len
                )
                v_len = max(min((l_enc + cr - 1) // cr, args.max_vision_len), 1)
                per_sample_ratios.append(s_len / v_len)
            last_step = int(ckpt_pairs[-1][0])
            if per_sample_ratios:
                run.log_histogram("eval/compression_ratio_dist", per_sample_ratios, step=last_step)
        run.finish()
        print(
            f"\nEval metrics logged to trackio (project={args.trackio_project} "
            f"run={run_name}, {len(ckpt_pairs)} checkpoint(s))"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
