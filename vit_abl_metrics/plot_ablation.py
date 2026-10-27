"""Plot ViT-ablation eval metrics (vit / random_vit / none) vs compression.

For every metric we plot the best checkpoint per compression level (cr),
picking the checkpoint with the lowest avg_loss at each cr, and put the three
arms on one axis so they can be compared at matched compression.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt

BASE = "/home/jovyan/shares/SR008.fs2/gigachat_checkpoints/rl/ckpts/MoE-losses/cvlm"
OUT = "/home/jovyan/shares/SR008.fs2/acherepanov/compress_project_new/CVLM/vit_abl_metrics"
ARMS = {
    "vit": "abl_vit_20260723_000740",
    "random_vit": "abl_randomvit_20260723_125137",
    "none": "abl_none_20260723_202733",
}
COLORS = {"vit": "#1f77b4", "random_vit": "#ff7f0e", "none": "#2ca02c"}
# (json key, human title, "higher is better"?)
METRICS = [
    ("token_accuracy", "Token accuracy", True),
    ("bleu4", "BLEU-4", True),
    ("rougeL", "ROUGE-L", True),
    ("rouge1", "ROUGE-1", True),
    ("rouge2", "ROUGE-2", True),
    ("exact_match", "Exact match", True),
    ("avg_loss", "Avg loss (NLL)", False),
    ("perplexity", "Perplexity", False),
    ("bits_per_source_token", "Bits / source token", False),
]


def step_of(f):
    m = re.search(r"step(\d+)\.json", f)
    return int(m.group(1)) if m else 0


def load_arm(name):
    """Return {cr: best-json} keyed by cr, best = lowest avg_loss."""
    rows = {}
    for f in glob.glob(f"{BASE}/{name}/eval_cvlm_native_step*.json"):
        if "samples" in f:
            continue
        j = json.load(open(f))
        cr = j.get("compression_rate")
        if cr is None or j.get("avg_loss") is None:
            continue
        if cr not in rows or j["avg_loss"] < rows[cr]["avg_loss"]:
            rows[cr] = j
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {arm: load_arm(name) for arm, name in ARMS.items()}

    # x-axis: effective context reduction (real end-to-end compression).
    def xy(arm, key):
        rows = data[arm]
        pts = sorted(
            (r["effective_context_reduction"], r[key])
            for r in rows.values()
            if key in r
        )
        return [p[0] for p in pts], [p[1] for p in pts]

    for key, title, higher in METRICS:
        plt.figure(figsize=(6.5, 4.5))
        for arm in ARMS:
            xs, ys = xy(arm, key)
            if not xs:
                continue
            plt.plot(xs, ys, "o-", color=COLORS[arm], label=arm, linewidth=2, markersize=6)
        plt.xlabel("Effective context reduction (x)")
        plt.ylabel(title)
        arrow = "higher = better" if higher else "lower = better"
        plt.title(f"{title} vs compression  ({arrow})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = f"{OUT}/{key}.png"
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"wrote {out}")

    # A combined 3x3 grid dashboard.
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (key, title, higher) in zip(axes.flat, METRICS):
        for arm in ARMS:
            xs, ys = xy(arm, key)
            if xs:
                ax.plot(xs, ys, "o-", color=COLORS[arm], label=arm, linewidth=1.8, markersize=5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("eff. compression (x)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("ViT ablation: vit vs random_vit vs none (best ckpt per cr)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{OUT}/dashboard.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}/dashboard.png")

    # dump the underlying table as CSV for the paper.
    import csv
    crs = sorted({cr for arm in ARMS for cr in data[arm]})
    with open(f"{OUT}/metrics_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cr", "eff_ctx_reduction", "arm"] + [k for k, _, _ in METRICS])
        for cr in crs:
            for arm in ARMS:
                r = data[arm].get(cr)
                if not r:
                    continue
                w.writerow(
                    [cr, round(r["effective_context_reduction"], 3), arm]
                    + [round(r.get(k, float("nan")), 4) for k, _, _ in METRICS]
                )
    print(f"wrote {OUT}/metrics_table.csv")


if __name__ == "__main__":
    main()
