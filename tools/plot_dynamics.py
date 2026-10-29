#!/usr/bin/env python3
"""Figure 1 статьи: loss-динамика четырёх веток с границами куррикулума.

  python tools/plot_dynamics.py --runs_root <runs> --stamp 20260729_171933 \
      --boundaries 1500 3000 4500 6000 --out fig_dynamics
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe)
STYLE = {
    "vit":       ("#D55E00", "pretrained ViT"),
    "frozenvit": ("#CC79A7", "frozen ViT"),
    "randomvit": ("#009E73", "random-init ViT"),
    "novit":     ("#0072B2", "no ViT"),
}


def load_csv(path, window=25):
    # RAW loss + rolling mean: loss_avg в metrics.csv — КУМУЛЯТИВНОЕ среднее
    # эпохи (AverageMeter), оно маскирует раннюю динамику — не использовать.
    steps, loss = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(float(row["step"])))
                loss.append(float(row["loss"]))
            except (KeyError, ValueError):
                continue
    sm = []
    for i in range(len(loss)):
        lo = max(0, i - window + 1)
        sm.append(sum(loss[lo:i + 1]) / (i + 1 - lo))
    return steps, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--stamp", default="")
    ap.add_argument("--boundaries", type=int, nargs="*", default=[1500, 3000, 4500, 6000])
    ap.add_argument("--out", default="fig_dynamics")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    for br, (color, label) in STYLE.items():
        dirs = sorted(glob.glob(os.path.join(args.runs_root, f"abl_{br}_*{args.stamp}*")))
        if not dirs:
            print(f"warn: нет {br}")
            continue
        p = os.path.join(dirs[-1], "metrics.csv")
        if not os.path.exists(p):
            print(f"warn: нет {p}")
            continue
        s, l = load_csv(p)
        ax.plot(s, l, color=color, label=label, lw=1.2)

    crs = [1, 2, 4, 8, 16]
    for i, b in enumerate(args.boundaries):
        ax.axvline(b, color="#999999", lw=0.6, ls="--", zorder=0)
    # подписи стадий cr по центрам интервалов
    edges = [0] + list(args.boundaries) + [ax.get_xlim()[1]]
    for i in range(min(len(crs), len(edges) - 1)):
        mid = (edges[i] + edges[i + 1]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, f"cr={crs[i]}", ha="center", va="top",
                fontsize=7, color="#666666")

    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (running avg)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}")
        print(f"saved {args.out}.{ext}")


if __name__ == "__main__":
    main()
