#!/usr/bin/env python3
"""Measure latent-padding contamination on an existing CVLM checkpoint.

No retraining is required. The tool runs the same held-out samples under:

1. batch size 1 versus batch size 8;
2. ordinary versus length-bucketed batches;
3. an explicit synthetic-latent-padding intervention.

For a padding-safe ViT/mixer, teacher-forcing PPL should be invariant up to
floating-point noise. For a legacy unmasked ViT/mixer, appended masked latent
slots can change real-token states because Hugging Face ViTLayer has no key
padding mask. The synthetic intervention is implemented inside CVLM *after*
pooling, so it changes neither source tokens nor real pooled latents.
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("DISABLE_LIGER", "1")


def _as_bool(value):
    return str(value).lower() in ("1", "true", "yes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--model_name_or_path", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--text_encoder_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--vision_encoder_name", default="google/vit-base-patch16-224")
    parser.add_argument("--dataset_name", default="sggetao/PwC")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument("--num_pool_latents", type=int, default=4)
    parser.add_argument("--max_vision_len", type=int, default=1024)
    parser.add_argument("--max_source_len", type=int, default=1024)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_answer_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--vit_init", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--freeze_vit", type=_as_bool, default=False)
    parser.add_argument("--no_vit", type=_as_bool, default=False)
    parser.add_argument("--mixer_depth_map", default="")
    parser.add_argument("--legacy_projector", type=_as_bool, default=False)
    parser.add_argument("--mask_vit_padding", type=_as_bool, default=False)
    parser.add_argument("--vit_apply_final_layernorm", type=_as_bool, default=True)
    parser.add_argument("--pad_fractions", default="0,0.25,0.5,0.75")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_model(args):
    import torch
    from modeling import CVLM, ModelArguments, TrainingArguments
    from safetensors.torch import load_file

    model_args = ModelArguments(train=False)
    model_args.model_name_or_path = args.model_name_or_path
    model_args.text_encoder_name = args.text_encoder_name
    model_args.vision_encoder_name = args.vision_encoder_name
    model_args.compression_rate = args.compression_rate
    model_args.num_pool_latents = max(args.num_pool_latents, 1)
    model_args.max_vision_len = args.max_vision_len
    model_args.projector_use_input_rmsnorm = not args.legacy_projector
    model_args.vit_init = args.vit_init
    model_args.vit_freeze = args.freeze_vit
    model_args.vit_skip = args.no_vit
    model_args.mixer_depth_map = args.mixer_depth_map
    model_args.mask_vit_padding = args.mask_vit_padding
    model_args.vit_apply_final_layernorm = args.vit_apply_final_layernorm

    training_args = TrainingArguments(output_dir="/tmp/_pad_diag")
    training_args.bf16 = torch.cuda.is_available()
    model = CVLM(model_args, training_args)
    state = load_file(args.checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), device


def forward_batch(model, batch, device):
    import torch

    keys = (
        "source_ids",
        "source_attention_mask",
        "prompt_ids",
        "answer_ids",
        "answer_labels",
        "prompt_mask",
        "answer_mask",
    )
    moved = {key: batch[key].to(device) for key in keys}
    with torch.no_grad():
        out = model(
            source_input_ids=moved["source_ids"],
            source_attention_mask=moved["source_attention_mask"],
            prompt_ids=moved["prompt_ids"],
            answer_ids=moved["answer_ids"],
            answer_labels=moved["answer_labels"],
            prompt_mask=moved["prompt_mask"],
            answer_mask=moved["answer_mask"],
        )

    labels_src = moved["answer_labels"]
    prompt_width = moved["prompt_ids"].size(1)
    vision_width = out["vision_mask"].size(1)
    ignore = torch.full(
        (labels_src.size(0), prompt_width + vision_width),
        -100,
        dtype=labels_src.dtype,
        device=device,
    )
    labels = torch.cat([ignore, labels_src], dim=1)
    logits = out["logits"]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    nll = (per_token * mask).sum()
    return float(nll), int(mask.sum()), int(labels_src.size(0))


def ppl_over(model, loader, device):
    total_nll = 0.0
    total_tokens = 0
    total_samples = 0
    for batch in loader:
        nll, tokens, samples = forward_batch(model, batch, device)
        total_nll += nll
        total_tokens += tokens
        total_samples += samples
    ppl = math.exp(min(total_nll / max(total_tokens, 1), 30.0))
    return {
        "ppl": ppl,
        "answer_tokens": int(total_tokens),
        "samples": int(total_samples),
    }


class PaddingAttentionCollector:
    """Forward-hook collector for legacy unmasked ViTSelfAttention modules."""

    def __init__(self, model):
        self.model = model
        if getattr(model, "mixer_layers", None) is not None:
            layers = list(model.mixer_layers)
        elif getattr(model, "vision_encoder", None) is not None:
            layers = list(model.vision_encoder.encoder.layer)
        else:
            layers = []
        self.sums = [0.0] * len(layers)
        self.counts = [0] * len(layers)
        self.handles = []
        for index, layer in enumerate(layers):
            self.handles.append(
                layer.attention.attention.register_forward_hook(self._make_hook(index))
            )

    def _make_hook(self, index):
        def hook(_module, _inputs, output):
            import torch

            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return
            probabilities = output[1].detach()
            valid = getattr(self.model, "_current_vision_attention_mask", None)
            if valid is None or probabilities.size(-1) != valid.size(1):
                return
            valid = valid.to(device=probabilities.device, dtype=torch.bool)
            padded_keys = (~valid)[:, None, None, :]
            pad_mass = (probabilities * padded_keys).sum(dim=-1)
            valid_queries = valid[:, None, :].expand_as(pad_mass)
            values = pad_mass.masked_select(valid_queries)
            if values.numel() > 0:
                self.sums[index] += float(values.sum())
                self.counts[index] += int(values.numel())

        return hook

    def reset(self):
        self.sums = [0.0] * len(self.sums)
        self.counts = [0] * len(self.counts)

    def means(self, mask_safe):
        if mask_safe and self.sums:
            # The custom mask-safe path bypasses native ViTSelfAttention.forward;
            # its key mask makes this quantity exactly zero by construction.
            return [0.0] * len(self.sums)
        return [
            (total / count if count else None)
            for total, count in zip(self.sums, self.counts)
        ]

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    import torch
    from cvlm_dataset import CvlmTrainDataset, make_collate_fn
    from torch.utils.data import DataLoader, Subset

    args = parse_args()
    torch.manual_seed(args.seed)
    model, device = build_model(args)
    dataset = CvlmTrainDataset(
        hf_dataset_name=args.dataset_name,
        hf_split=args.dataset_split,
        decoder_tokenizer_name=args.model_name_or_path,
        encoder_tokenizer_name=args.text_encoder_name,
        max_prompt_len=args.max_prompt_len,
        max_answer_len=args.max_answer_len,
        max_source_len=args.max_source_len,
        max_samples=args.max_samples,
    )
    collate = make_collate_fn(
        dec_pad_id=model.tokenizer.pad_token_id,
        enc_pad_id=model.encoder_tokenizer.pad_token_id,
    )
    attention_collector = PaddingAttentionCollector(model)

    results = {
        "checkpoint": args.checkpoint_path,
        "compression_rate": args.compression_rate,
        "mixer_depth_map": args.mixer_depth_map,
        "no_vit": args.no_vit,
        "vit_init": args.vit_init,
        "mask_vit_padding": args.mask_vit_padding,
        "vit_apply_final_layernorm": args.vit_apply_final_layernorm,
        "max_samples": len(dataset),
    }

    # 1. Batch-size invariance on exactly the same dataset rows.
    for batch_size in (1, 8):
        attention_collector.reset()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        metrics = ppl_over(model, loader, device)
        metrics["padding_attention_mass_by_layer"] = attention_collector.means(
            args.mask_vit_padding
        )
        results[f"batch_size_{batch_size}"] = metrics
        print(f"[batch-size={batch_size}] PPL={metrics['ppl']:.6f}")
    results["batch_size_ppl_delta"] = (
        results["batch_size_8"]["ppl"] - results["batch_size_1"]["ppl"]
    )

    # 2. Length bucketing: same examples, only batch composition changes.
    lengths = [(index, int(dataset[index]["source_ids"].numel())) for index in range(len(dataset))]
    bucket_order = [index for index, _ in sorted(lengths, key=lambda item: item[1])]
    bucket_loader = DataLoader(
        Subset(dataset, bucket_order),
        batch_size=8,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    attention_collector.reset()
    bucket_metrics = ppl_over(model, bucket_loader, device)
    bucket_metrics["padding_attention_mass_by_layer"] = attention_collector.means(
        args.mask_vit_padding
    )
    results["batch_size_8_bucketed"] = bucket_metrics
    results["bucketing_ppl_delta"] = (
        results["batch_size_8"]["ppl"] - bucket_metrics["ppl"]
    )
    print(f"[length-bucketed] PPL={bucket_metrics['ppl']:.6f}")

    # 3. Causal intervention: append masked slots after pooling.
    fractions = [float(value) for value in args.pad_fractions.split(",") if value.strip()]
    if 0.0 not in fractions:
        fractions.insert(0, 0.0)
    results["synthetic_latent_padding"] = {}
    base_ppl = None
    for fraction in fractions:
        model.synthetic_latent_pad_fraction = float(fraction)
        attention_collector.reset()
        loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        metrics = ppl_over(model, loader, device)
        metrics["padding_attention_mass_by_layer"] = attention_collector.means(
            args.mask_vit_padding
        )
        if fraction == 0.0:
            base_ppl = metrics["ppl"]
        metrics["ppl_delta_vs_zero"] = (
            metrics["ppl"] - base_ppl if base_ppl is not None else 0.0
        )
        results["synthetic_latent_padding"][str(fraction)] = metrics
        print(
            f"[synthetic-latent-pad={fraction:.2f}] "
            f"PPL={metrics['ppl']:.6f} delta={metrics['ppl_delta_vs_zero']:+.6f}"
        )
    model.synthetic_latent_pad_fraction = 0.0
    attention_collector.close()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"saved -> {args.output_json}")


if __name__ == "__main__":
    main()
