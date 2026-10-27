"""Hidden-states mechanistic analysis of the ViT ablation (vit / random_vit / none).

Goal: understand *why* the pretrained visual prior does not transfer, i.e. why the
`vit`, `random_vit` and `none` arms end up at (near-)matched accuracy after training.

For each arm we load the final cr=16 checkpoint, push a SHARED batch of PwC eval
documents through `_encode_vision(..., return_intermediates=True)` and extract three
representations at matched geometry ([B, V, .]):
    vit_input     : projected + positional pooled text  (backend-INDEPENDENT input)
    vit_output    : after the ViT transformer (or bypass for 'none')
    vision_embeds : decoder-facing embeddings (after vision_projector)

We then compute:
  (1) Linear CKA between arms on vit_output and vision_embeds
      -> "do the arms end up representing the same thing?"
  (2) Transform magnitude ||vit_output - vit_input|| / ||vit_input|| per arm
      -> "how much does the ViT actually move the representation?"
  (3) Pretrained-prior weight drift: trained vision_encoder vs its init
      (vit -> ImageNet init; random_vit -> seed-0 re-init) as relative L2 per tensor
      -> "did training overwrite the ViT's pretrained weights?"
  (4) Effective rank / anisotropy of vit_output and vision_embeds per arm
      -> "does the ViT add representational capacity, or collapse it?"

Outputs: PNG plots + CSV/JSON tables under vit_abl_metrics/repr/.
Run inside the cvlm env; example command in the trailing __main__ note.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

from modeling import CVLM, ModelArguments, TrainingArguments  # noqa: E402
from cvlm_dataset import CvlmTrainDataset, make_collate_fn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

# ---- run config: must mirror the ablation training/eval config exactly ----
BASE = "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm"
OUT = os.path.join(HERE, "repr")
ARMS = {
    "vit": "abl_vit_20260723_000740",
    "random_vit": "abl_randomvit_20260723_125137",
    "none": "abl_none_20260723_202733",
}
CKPT_STEP = 37751  # final cr=16 checkpoint (model_step_37751.safetensors)
COLORS = {"vit": "#1f77b4", "random_vit": "#ff7f0e", "none": "#2ca02c"}

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TEXT_ENCODER = "answerdotai/ModernBERT-base"
VISION_ENCODER = "google/vit-base-patch16-224"
MAX_VISION_LEN = 1024
NUM_POOL_LATENTS = 4
USE_CHAT_TEMPLATE = True
CR = 16  # analyse at the headline compression where a real ViT would matter most

N_DOCS = 256    # shared eval docs to pool activations over
BATCH_SIZE = 4
DATASET = "sggetao/PwC"
SPLIT = "test"
MAX_PROMPT_LEN = 512
MAX_ANSWER_LEN = 2048
SEED = 42


def build_model(vision_backend):
    ma = ModelArguments(train=False)
    ma.model_name_or_path = MODEL_NAME
    ma.text_encoder_name = TEXT_ENCODER
    ma.vision_encoder_name = VISION_ENCODER
    ma.max_vision_len = MAX_VISION_LEN
    ma.compression_rate = CR
    ma.num_pool_latents = NUM_POOL_LATENTS
    ma.projector_use_input_rmsnorm = True  # LEGACY_PROJECTOR=0
    ma.vision_backend = vision_backend
    ta = TrainingArguments(output_dir="/tmp/repr_dummy")
    ta.bf16 = torch.cuda.is_available()
    model = CVLM(ma, ta)
    return model, ma, ta


def load_ckpt(model, arm):
    path = f"{BASE}/{ARMS[arm]}/model_step_{CKPT_STEP}.safetensors"
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  [{arm}] loaded {path}: missing={len(missing)} unexpected={len(unexpected)}")
    return sd


def make_loader(model):
    ds = CvlmTrainDataset(
        hf_dataset_name=DATASET,
        hf_split=SPLIT,
        decoder_tokenizer_name=MODEL_NAME,
        encoder_tokenizer_name=TEXT_ENCODER,
        max_prompt_len=MAX_PROMPT_LEN,
        max_answer_len=MAX_ANSWER_LEN,
        max_source_len=CR * MAX_VISION_LEN,
        max_samples=N_DOCS,
        use_chat_template=USE_CHAT_TEMPLATE,
    )
    collate = make_collate_fn(
        dec_pad_id=model.tokenizer.pad_token_id,
        enc_pad_id=model.encoder_tokenizer.pad_token_id,
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, collate_fn=collate, drop_last=False)


@torch.no_grad()
def extract(model, loader, device):
    """Run every batch through _encode_vision, collect masked token rows per tensor.

    Returns dict[name] -> float32 numpy array [N_tokens, dim] (only real vision tokens).
    """
    keep = {"vit_input": [], "vit_output": [], "vision_embeds": []}
    for batch in loader:
        sids = batch["source_ids"].to(device)
        smask = batch["source_attention_mask"].to(device)
        _, vmask, inter = model._encode_vision(sids, smask, return_intermediates=True)
        m = vmask.bool()  # [B, V] true where a real vision token sits
        for name in keep:
            t = inter[name].float()          # [B, V, D]
            keep[name].append(t[m].cpu().numpy())  # [n_real, D]
    return {k: np.concatenate(v, axis=0) for k, v in keep.items()}


# ---------------------- metric primitives ----------------------
def linear_cka(X, Y):
    """Linear CKA in [0,1]; rotation/scale-invariant. X,Y: [N, d] (same N rows)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC via Gram: ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    xty = X.T @ Y
    num = float((xty * xty).sum())
    xtx = X.T @ X
    yty = Y.T @ Y
    den = float(np.sqrt((xtx * xtx).sum()) * np.sqrt((yty * yty).sum())) + 1e-12
    return num / den


def transform_magnitude(vit_input, vit_output):
    """Per-row relative L2 movement, averaged. vit_input/vit_output: [N, d].

    Magnitude answers "how much" the ViT moves the vector, but conflates
    rescaling with reorientation (see cosine_alignment for the pure-angle view).
    """
    num = np.linalg.norm(vit_output - vit_input, axis=1)
    den = np.linalg.norm(vit_input, axis=1) + 1e-12
    r = num / den
    return float(r.mean()), float(r.std())


def cosine_alignment(vit_input, vit_output):
    """Per-row cosine similarity between input and output directions. [N, d].

    Scale-invariant: isolates *angular* reorientation. cos==1 => the ViT only
    rescales / passes the direction through (identity-in-direction); low cos =>
    the ViT genuinely reorients the representation. For the 'none' arm this is
    exactly 1.0 by construction (vit_output == vit_input).

    Returns (mean, std, per-row array) so we can plot the full distribution.
    """
    a = vit_input.astype(np.float64)
    b = vit_output.astype(np.float64)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    cos = (a * b).sum(1) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)
    return float(cos.mean()), float(cos.std()), cos


def cross_arm_cosine(Xa, Xb):
    """Per-token cosine between two arms' outputs on ALIGNED rows. Xa,Xb: [N, d].

    Rows are aligned across arms because extract() concatenates in a fixed order
    (shuffle=False, identical loader, identical cr=16 mask geometry), so row i is
    the same (document, vision position) in every arm. This is the raw-direction
    counterpart to CKA: CKA is rotation-invariant ("same geometry up to a basis
    change?"), this asks "do the arms literally point the same way per token?".

    Returns (mean, std, per-row array).
    """
    n = min(Xa.shape[0], Xb.shape[0])
    a = Xa[:n].astype(np.float64)
    b = Xb[:n].astype(np.float64)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    cos = (a * b).sum(1) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)
    return float(cos.mean()), float(cos.std()), cos


def effective_rank(X, max_rows=8000):
    """Effective rank (entropy of normalised singular-value spectrum, exp)."""
    if X.shape[0] > max_rows:
        idx = np.random.RandomState(0).choice(X.shape[0], max_rows, replace=False)
        X = X[idx]
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-9]
    if s.size == 0:
        return 0.0, 0.0
    p = (s ** 2) / (s ** 2).sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))
    # anisotropy = fraction of variance in the top component (higher => more collapsed)
    aniso = float(p[0])
    return erank, aniso


def weight_drift(trained_sd, arm, ma, ta):
    """Relative L2 drift of vision_encoder weights vs their training init.

    vit        -> init = ImageNet-pretrained google/vit-base-patch16-224.
    random_vit -> init = seed-0 re-init (reconstructed by building a fresh model).
    none       -> no ViT transformer weights are used; skip.
    """
    if arm == "none":
        return {}
    # Fresh model reproduces the exact init the run started from (vit: pretrained;
    # random_vit: deterministic seed-0 re-init inside CVLM.__init__).
    fresh, _, _ = build_model(arm)
    init_sd = fresh.state_dict()
    drift = {}
    for k, w0 in init_sd.items():
        if not k.startswith("vision_encoder.encoder."):
            continue
        if k not in trained_sd:
            continue
        w1 = trained_sd[k].float()
        w0 = w0.float()
        if w0.shape != w1.shape:
            continue
        n0 = float(torch.linalg.norm(w0))
        # Skip tensors initialised to (near-)zero — e.g. biases and LayerNorm
        # under _init_weights zero-init in random_vit. Relative drift ‖w1-w0‖/‖w0‖
        # explodes (divide-by-~0) and is meaningless. The informative comparison
        # is the weight matrices, which have nonzero init in both arms.
        if n0 < 1e-6:
            continue
        d = float(torch.linalg.norm(w1 - w0) / n0)
        drift[k] = d
    del fresh
    return drift


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reps = {}       # arm -> {tensor_name: np[N,d]}
    trained_sds = {}
    for arm in ARMS:
        print(f"\n=== arm: {arm} ===")
        model, ma, ta = build_model(arm)
        model.to(device)
        sd = load_ckpt(model, arm)
        trained_sds[arm] = sd
        model.eval()
        loader = make_loader(model)
        reps[arm] = extract(model, loader, device)
        n = reps[arm]["vit_output"].shape[0]
        print(f"  collected {n} real vision-token rows")
        del model
        torch.cuda.empty_cache()

    # sanity: vit_input must be IDENTICAL across arms (backend-independent).
    # (Row counts/order match because the loader is deterministic + shuffle=False.)
    ref = reps["vit"]["vit_input"]
    for arm in ("random_vit", "none"):
        a = reps[arm]["vit_input"]
        if a.shape == ref.shape:
            diff = float(np.abs(a - ref).max())
            print(f"[sanity] max|vit_input[{arm}] - vit_input[vit]| = {diff:.3e} "
                  f"(NOT expected ~0: projector weights are trained per-arm)")

    results = {"cr": CR, "n_docs": N_DOCS, "arms": {}, "cka": {}}
    cos_dists = {}  # arm -> per-row cosine array (for the distribution plot)

    # (2) transform magnitude + cosine alignment + (4) effective rank/anisotropy, per arm
    for arm in ARMS:
        r = reps[arm]
        mag_mean, mag_std = transform_magnitude(r["vit_input"], r["vit_output"])
        cos_mean, cos_std, cos_arr = cosine_alignment(r["vit_input"], r["vit_output"])
        cos_dists[arm] = cos_arr
        er_out, an_out = effective_rank(r["vit_output"])
        er_emb, an_emb = effective_rank(r["vision_embeds"])
        er_in, an_in = effective_rank(r["vit_input"])
        results["arms"][arm] = {
            "transform_mag_mean": mag_mean,
            "transform_mag_std": mag_std,
            "cos_align_mean": cos_mean,
            "cos_align_std": cos_std,
            "erank_vit_input": er_in,
            "erank_vit_output": er_out,
            "erank_vision_embeds": er_emb,
            "aniso_vit_input": an_in,
            "aniso_vit_output": an_out,
            "aniso_vision_embeds": an_emb,
        }
        print(f"[{arm}] transform_mag={mag_mean:.4f}±{mag_std:.4f} "
              f"cos(in,out)={cos_mean:.4f}±{cos_std:.4f} "
              f"erank(in/out/emb)={er_in:.1f}/{er_out:.1f}/{er_emb:.1f}")

    # (1) CKA between arms on vit_output and vision_embeds
    arms = list(ARMS)
    for tname in ("vit_output", "vision_embeds"):
        results["cka"][tname] = {}
        for i in range(len(arms)):
            for j in range(i, len(arms)):
                a, b = arms[i], arms[j]
                Xa, Xb = reps[a][tname], reps[b][tname]
                n = min(Xa.shape[0], Xb.shape[0])
                v = linear_cka(Xa[:n], Xb[:n])
                results["cka"][tname][f"{a}__{b}"] = v
                print(f"[CKA/{tname}] {a} vs {b}: {v:.4f}")

    # (1b) cross-arm per-token cosine on vit_output (raw-direction counterpart
    # to CKA). Off-diagonal arm pairs only; aligned rows.
    results["cross_arm_cos"] = {}
    xcos_dists = {}  # "a__b" -> per-row cosine array (for distribution plot)
    for tname in ("vit_output", "vision_embeds"):
        results["cross_arm_cos"][tname] = {}
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                a, b = arms[i], arms[j]
                m, s, arr = cross_arm_cosine(reps[a][tname], reps[b][tname])
                results["cross_arm_cos"][tname][f"{a}__{b}"] = {"mean": m, "std": s}
                if tname == "vit_output":
                    xcos_dists[f"{a}__{b}"] = arr
                print(f"[xCOS/{tname}] {a} vs {b}: {m:.4f}±{s:.4f}")

    # (3) weight drift (summary stats per arm)
    drift_summary = {}
    for arm in ("vit", "random_vit"):
        d = weight_drift(trained_sds[arm], arm, None, None)
        if d:
            vals = np.array(list(d.values()))
            drift_summary[arm] = {
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "n_tensors": int(vals.size),
                "per_tensor": d,
            }
            print(f"[drift/{arm}] mean={vals.mean():.4f} median={np.median(vals):.4f} "
                  f"range=[{vals.min():.4f},{vals.max():.4f}] over {vals.size} tensors")
    results["weight_drift"] = drift_summary

    with open(f"{OUT}/repr_metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {OUT}/repr_metrics.json")

    _plots(results, drift_summary, cos_dists, xcos_dists)
    _csv(results)


def _plots(results, drift_summary, cos_dists, xcos_dists):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(ARMS)

    # A: transform magnitude bar
    fig, ax = plt.subplots(figsize=(5, 4))
    mags = [results["arms"][a]["transform_mag_mean"] for a in arms]
    errs = [results["arms"][a]["transform_mag_std"] for a in arms]
    ax.bar(arms, mags, yerr=errs, color=[COLORS[a] for a in arms], capsize=4)
    ax.set_ylabel(r"$\||vit\_output - vit\_input\|| / \||vit\_input\||$")
    ax.set_title(f"ViT transform magnitude (cr={CR})")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/transform_magnitude.png", dpi=140); plt.close(fig)

    # A2: cosine alignment — bar (mean±std) + per-row distribution
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    cmeans = [results["arms"][a]["cos_align_mean"] for a in arms]
    cstds = [results["arms"][a]["cos_align_std"] for a in arms]
    ax0.bar(arms, cmeans, yerr=cstds, color=[COLORS[a] for a in arms], capsize=4)
    ax0.set_ylabel("cosine(vit_input, vit_output)")
    ax0.set_title(f"Directional alignment before/after ViT (cr={CR})")
    ax0.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax0.grid(True, axis="y", alpha=0.3)
    for arm in arms:
        c = cos_dists[arm]
        ax1.hist(c, bins=60, range=(min(-0.1, float(c.min())), 1.0),
                 histtype="step", lw=1.6, color=COLORS[arm], label=arm, density=True)
    ax1.set_xlabel("per-token cosine(in, out)"); ax1.set_ylabel("density")
    ax1.set_title("Cosine distribution per arm")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/cosine_alignment.png", dpi=140); plt.close(fig)

    # B: effective rank grouped bar (in/out/emb)
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = ["erank_vit_input", "erank_vit_output", "erank_vision_embeds"]
    labels = ["vit_input", "vit_output", "vision_embeds"]
    x = np.arange(len(labels)); w = 0.25
    for k, arm in enumerate(arms):
        vals = [results["arms"][arm][kk] for kk in keys]
        ax.bar(x + (k - 1) * w, vals, w, label=arm, color=COLORS[arm])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("effective rank"); ax.set_title(f"Effective rank per stage (cr={CR})")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/effective_rank.png", dpi=140); plt.close(fig)

    # C: CKA heatmaps (vit_output, vision_embeds)
    for tname in ("vit_output", "vision_embeds"):
        M = np.eye(len(arms))
        for i, a in enumerate(arms):
            for j, b in enumerate(arms):
                key = f"{a}__{b}" if f"{a}__{b}" in results["cka"][tname] else f"{b}__{a}"
                M[i, j] = results["cka"][tname][key]
        fig, ax = plt.subplots(figsize=(4.6, 4))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=30, ha="right")
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms)
        for i in range(len(arms)):
            for j in range(len(arms)):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=9)
        ax.set_title(f"Linear CKA: {tname} (cr={CR})")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(f"{OUT}/cka_{tname}.png", dpi=140); plt.close(fig)

    # C2: cross-arm per-token cosine on vit_output — bar (mean±std) + distribution
    pairs = list(xcos_dists)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4))
    means = [results["cross_arm_cos"]["vit_output"][p]["mean"] for p in pairs]
    stds = [results["cross_arm_cos"]["vit_output"][p]["std"] for p in pairs]
    labels = [p.replace("__", "\nvs ") for p in pairs]
    ax0.bar(labels, means, yerr=stds, color="#6a51a3", capsize=4)
    ax0.axhline(0.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax0.set_ylabel("per-token cosine(vit_output)")
    ax0.set_title(f"Cross-arm directional agreement (cr={CR})")
    ax0.grid(True, axis="y", alpha=0.3)
    for p in pairs:
        c = xcos_dists[p]
        ax1.hist(c, bins=60, range=(-1.0, 1.0), histtype="step", lw=1.6,
                 label=p.replace("__", " vs "), density=True)
    ax1.axvline(0.0, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.set_xlabel("per-token cross-arm cosine"); ax1.set_ylabel("density")
    ax1.set_title("Cross-arm cosine distribution")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/cross_arm_cosine.png", dpi=140); plt.close(fig)

    # D: weight drift bar
    if drift_summary:
        fig, ax = plt.subplots(figsize=(5, 4))
        names = list(drift_summary)
        means = [drift_summary[a]["mean"] for a in names]
        mins = [drift_summary[a]["min"] for a in names]
        maxs = [drift_summary[a]["max"] for a in names]
        yerr = [[m - lo for m, lo in zip(means, mins)],
                [hi - m for m, hi in zip(means, maxs)]]
        ax.bar(names, means, yerr=yerr, color=[COLORS[a] for a in names], capsize=4)
        ax.set_ylabel("relative L2 weight drift vs init")
        ax.set_title("ViT weight drift after training")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{OUT}/weight_drift.png", dpi=140); plt.close(fig)

    print(f"wrote plots to {OUT}/*.png")


def _csv(results):
    with open(f"{OUT}/repr_metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "transform_mag_mean", "transform_mag_std",
                    "cos_align_mean", "cos_align_std",
                    "erank_vit_input", "erank_vit_output", "erank_vision_embeds",
                    "aniso_vit_input", "aniso_vit_output", "aniso_vision_embeds"])
        for arm, r in results["arms"].items():
            w.writerow([arm] + [round(r[k], 4) for k in (
                "transform_mag_mean", "transform_mag_std",
                "cos_align_mean", "cos_align_std",
                "erank_vit_input", "erank_vit_output", "erank_vision_embeds",
                "aniso_vit_input", "aniso_vit_output", "aniso_vision_embeds")])
    print(f"wrote {OUT}/repr_metrics.csv")


if __name__ == "__main__":
    # PYTHONNOUSERSITE=1 TRACKIO_DISABLE=1 \
    #   /home/jovyan/.mlspace/envs/cvlm/bin/python vit_abl_metrics/repr_analysis.py
    main()
