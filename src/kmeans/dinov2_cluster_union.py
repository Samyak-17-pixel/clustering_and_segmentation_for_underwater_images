"""k=2..4 clustering of DINOv2 patch features (PCA RGB and raw 768-D) with fg/bg union search.

For each image and feature set: k-means for k in {2,3,4}, k* chosen by silhouette score, and every
way of merging the clusters into one fg group + one bg group is scored against the GT (oracle).
A no-GT border-rule union is reported alongside.
"""
import argparse
import csv
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from gt_utils import GT_MODES, drop_degenerate, load_gt

from dinov2_seg import ROOT, load_model, metrics, patch_features, pca_masks, upsample

KS = (2, 3, 4)


def subsets(k):
    """All non-empty proper subsets of {0..k-1} (candidate fg unions): 2^k - 2 of them."""
    return [s for r in range(1, k) for s in combinations(range(k), r)]


def border_ring(shape):
    b = np.zeros(shape, bool)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return b


def border_rule_union(labels, k):
    """No-GT union: cluster is fg if its border fraction is below the global border fraction."""
    border = border_ring(labels.shape)
    frac = np.array([(border & (labels == c)).sum() / max((labels == c).sum(), 1) for c in range(k)])
    fg = [c for c in range(k) if frac[c] < border.mean()]
    return tuple(fg) if 0 < len(fg) < k else (int(frac.argmin()),)


def union_mask(labels, fg):
    return np.isin(labels, fg)


def best_union(labels, k, gt):
    best = None
    for s in subsets(k):
        prob = upsample(union_mask(labels, s), gt.shape)
        iou, f, mae = metrics(prob, gt)
        if best is None or iou > best["iou"]:
            best = dict(iou=iou, f=f, mae=mae, fg=s, prob=prob)
    return best


def process_feature(X, grid, gt):
    """k-means for each k; returns per-k labels, silhouettes, oracle results, and the k* summary."""
    labels, sil, oracle = {}, {}, {}
    for k in KS:
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X)
        labels[k] = lab.reshape(grid)
        sil[k] = float(silhouette_score(X, lab))
        oracle[k] = best_union(labels[k], k, gt)
    k_star = max(KS, key=sil.get)
    s = border_rule_union(labels[k_star], k_star)
    border = metrics(upsample(union_mask(labels[k_star], s), gt.shape), gt)
    return dict(labels=labels, sil=sil, oracle=oracle, k_star=k_star, border_fg=s, border=border)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "USOD")
    ap.add_argument("--out", default=ROOT / "results" / "kmeans" / "usod" / "cluster_union_stride14_k4")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=588)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", choices=GT_MODES, default="usod")
    ap.add_argument("--min-water-frac", type=float, default=0.0,
                    help="suim_nonwater only: skip images with less water than this (all-foreground GT)")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    for sub in ("panels", "masks_rgb", "masks_768"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    paths = sorted((data / "images").glob("*.jpg"))
    paths = drop_degenerate(paths, data, args.gt, args.min_water_frac, out)
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    for p in tqdm(paths):
        img = Image.open(p).convert("RGB")
        gt = load_gt(data, p.stem, args.gt)
        feats = patch_features(model, img, (args.width, args.height), device)
        grid = feats.shape[:2]
        _, rgb = pca_masks(feats)
        sets = {"rgb": rgb.reshape(-1, 3), "768": feats.reshape(-1, feats.shape[-1])}

        res = {name: process_feature(X, grid, gt) for name, X in sets.items()}
        for name, r in res.items():
            o = r["oracle"][r["k_star"]]
            Image.fromarray(((o["prob"] > 0.5) * 255).astype(np.uint8)).save(out / f"masks_{name}" / f"{p.stem}.png")
            row = {"image": p.name, "feat": name, "k_star": r["k_star"]}
            for k in KS:
                row[f"sil{k}"] = r["sil"][k]
                row[f"oracle_iou_k{k}"] = r["oracle"][k]["iou"]
            row.update(oracle_iou=o["iou"], oracle_f=o["f"], oracle_mae=o["mae"],
                       oracle_fg="+".join(map(str, o["fg"])),
                       border_iou=r["border"][0], border_f=r["border"][1], border_mae=r["border"][2])
            rows.append(row)

        fig, axs = plt.subplots(1, 7, figsize=(28, 4))
        axs[0].imshow(img), axs[0].set_title("image")
        axs[1].imshow(rgb), axs[1].set_title("PCA RGB")
        for j, name in enumerate(("rgb", "768")):
            r = res[name]
            o = r["oracle"][r["k_star"]]
            axs[2 + 2 * j].imshow(r["labels"][r["k_star"]], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
            axs[2 + 2 * j].set_title(f"{name} clusters k*={r['k_star']}")
            axs[3 + 2 * j].imshow(o["prob"] > 0.5, cmap="gray")
            axs[3 + 2 * j].set_title(f"{name} best union {'+'.join(map(str, o['fg']))} IoU={o['iou']:.2f}\n"
                                     f"(border rule IoU={r['border'][0]:.2f})")
        axs[6].imshow(gt, cmap="gray"), axs[6].set_title("GT")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "panels" / f"{p.stem}.png", dpi=60)
        plt.close(fig)

    with open(out / "metrics_union.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    for name in ("rgb", "768"):
        rs = [r for r in rows if r["feat"] == name]
        m = lambda key: float(np.mean([r[key] for r in rs]))
        med = lambda key: float(np.median([r[key] for r in rs]))
        print(f"[{name}] oracle IoU mean {m('oracle_iou'):.4f} median {med('oracle_iou'):.4f} | "
              f"border-rule IoU mean {m('border_iou'):.4f} median {med('border_iou'):.4f} | mean k* {m('k_star'):.2f}")
        print(f"[{name}] oracle IoU by fixed k: " + ", ".join(f"k={k}: {m(f'oracle_iou_k{k}'):.4f}" for k in KS))
        ranked = sorted(rs, key=lambda r: -r["oracle_iou"])
        print(f"[{name}] top5", [(r["image"], round(r["oracle_iou"], 2)) for r in ranked[:5]])
        print(f"[{name}] bottom5", [(r["image"], round(r["oracle_iou"], 2)) for r in ranked[-5:]])


if __name__ == "__main__":
    main()
