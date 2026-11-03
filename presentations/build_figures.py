"""Figure tiles for the Review 2 deck (presentations/Review-2-...pptx). Writes rounded PNG tiles + meta.json to presentations/figures/.

Every tile is a separate picture so that captions stay editable text in the deck. Numbers are taken from the
same code paths as the experiments (k-means: src/dinov2_cluster_union_stride.py, dendrograms: src/dendrogram/).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import dendrogram, linkage

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "dendrogram")]

from dendro_cluster import build_tree, gap_k                                          # noqa: E402
from dinov2_dendro import FeatureContext, evaluate_tree                               # noqa: E402
from dinov2_cluster_union_stride import (border_rule_union, dense_features,           # noqa: E402
                                         process_feature, upsample_maps)
from dinov2_seg import PATCH, load_model, pca_masks                                    # noqa: E402
from gt_utils import load_gt                                                          # noqa: E402

OUT = REPO / "presentations" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = (588, 448)
TILE = (640, 480)
BG = (13, 11, 36)
PALETTE = [(111, 211, 242), (242, 196, 111), (123, 224, 168), (232, 120, 138), (183, 156, 242), (169, 180, 200)]
META = {}


def rounded(img, radius=30):
    img = img.convert("RGBA").resize(TILE, Image.NEAREST if img.size[0] < TILE[0] else Image.LANCZOS)
    mask = Image.new("L", TILE, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, TILE[0] - 1, TILE[1] - 1], radius=radius, fill=255)
    img.putalpha(mask)
    return img


def save(img, name):
    rounded(img).save(OUT / f"{name}.png")


def rgb_tile(arr01):
    return Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))


def mask_tile(mask):
    a = np.zeros(mask.shape + (3,), np.uint8)
    a[:] = (8, 8, 24)
    a[mask] = (240, 244, 250)
    return Image.fromarray(a)


def label_tile(labels):
    return Image.fromarray(np.array(PALETTE, np.uint8)[labels % len(PALETTE)])


def prob_mask(grid_mask, gt_shape, stride):
    return upsample_maps(grid_mask[None].astype(np.float32), gt_shape, stride, SIZE)[0] > 0.5


def iou(pred, gt):
    u = (pred | gt).sum()
    return float((pred & gt).sum() / u) if u else 1.0


def kmeans_run(X, grid, gt, stride, kmax):
    res = process_feature(X, grid, gt, SimpleNamespace(kmax=kmax, stride=stride), SIZE)
    k = res["k_star"]
    lab = res["labels"][k]
    fg = border_rule_union(lab, k, round(PATCH / stride))
    real = prob_mask(np.isin(lab, fg), gt.shape, stride)
    orc = res["oracle"][k]
    return dict(k=k, labels=lab, real=real, real_iou=float(res["border"][0]), oracle=orc["prob"] > 0.5,
                oracle_iou=float(orc["iou"]), oracle_by_k={kk: float(v["iou"]) for kk, v in res["oracle"].items()})


def cached_mask(stem, stride, key, gt_shape):
    z = np.load(REPO / "results" / "dendrogram" / "_cache" / "usod" / f"{stem}_s{stride}.npz")
    gh, gw = z["grid"]
    m = np.unpackbits(z[key])[: gh * gw].reshape(gh, gw).astype(bool)
    return prob_mask(m, gt_shape, stride)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("dinov2_vitb14", device)
    data = REPO / "data" / "USOD"

    def load(stem):
        return Image.open(data / "images" / f"{stem}.jpg").convert("RGB"), load_gt(data, stem, "usod")

    # ---- PCA-RGB stride 14 vs 3 (slides 4, 6) ----
    for stem in ("usod_test5", "usod_test205", "usod_test106"):
        img, gt = load(stem)
        save(img, f"{stem}_photo")
        save(mask_tile(gt), f"{stem}_gt")
        for s in (14, 3):
            dense, _ = dense_features(model, img, SIZE, s, device)
            save(rgb_tile(pca_masks(dense)[1]), f"{stem}_pca{s}")

    # ---- pipeline strip (slide 4) and IoU explanation (slide 5): usod_test5, stride 14, k<=4 ----
    img, gt = load("usod_test5")
    dense, _ = dense_features(model, img, SIZE, 14, device)
    grid = dense.shape[:2]
    gimg = img.resize(TILE).copy()
    d = ImageDraw.Draw(gimg)
    for i in range(1, 42):
        d.line([(i * TILE[0] / 42, 0), (i * TILE[0] / 42, TILE[1])], fill=(255, 255, 255), width=1)
    for j in range(1, 32):
        d.line([(0, j * TILE[1] / 32), (TILE[0], j * TILE[1] / 32)], fill=(255, 255, 255), width=1)
    save(gimg, "pipeline_patches")
    km = kmeans_run(dense.reshape(-1, dense.shape[-1]), grid, gt, 14, 4)
    save(label_tile(km["labels"]), "pipeline_clusters")
    save(mask_tile(km["real"]), "pipeline_mask")
    tp, fp, fn = km["real"] & gt, km["real"] & ~gt, ~km["real"] & gt
    ov = np.zeros(gt.shape + (3,), np.uint8)
    ov[:] = (8, 8, 24)
    ov[tp], ov[fp], ov[fn] = PALETTE[2], PALETTE[3], PALETTE[1]
    save(Image.fromarray(ov), "iou_overlap")
    META["iou_example"] = dict(image="usod_test5", iou=km["real_iou"], overlap=int(tp.sum()), union=int((km["real"] | gt).sum()))

    # ---- k-means k<=4 vs k<=6, stride 14 (slides 7, 8) ----
    META["kmeans14"] = {}
    for stem in ("usod_test279", "usod_test81"):
        img, gt = load(stem)
        dense, _ = dense_features(model, img, SIZE, 14, device)
        grid = dense.shape[:2]
        X = dense.reshape(-1, dense.shape[-1])
        save(img, f"{stem}_photo")
        save(mask_tile(gt), f"{stem}_gt")
        META["kmeans14"][stem] = {}
        for kmax in (4, 6):
            r = kmeans_run(X, grid, gt, 14, kmax)
            save(label_tile(r["labels"]), f"{stem}_km14_k{kmax}_clusters")
            save(mask_tile(r["real"]), f"{stem}_km14_k{kmax}_mask")
            save(mask_tile(r["oracle"]), f"{stem}_km14_k{kmax}_oracle")
            META["kmeans14"][stem][kmax] = dict(k=r["k"], real_iou=r["real_iou"], oracle_iou=r["oracle_iou"])

    # ---- stride 3, k<=6: gap slide and wins (slides 9, 11, 13) ----
    META["s3"] = {}
    for stem in ("usod_test205", "usod_test53", "usod_test2", "usod_test153", "usod_test107", "usod_test106"):
        img, gt = load(stem)
        dense, _ = dense_features(model, img, SIZE, 3, device)
        grid = dense.shape[:2]
        r = kmeans_run(dense.reshape(-1, dense.shape[-1]), grid, gt, 3, 6)
        save(img, f"{stem}_photo")
        save(mask_tile(gt), f"{stem}_gt")
        save(mask_tile(r["real"]), f"{stem}_km3_real")
        save(mask_tile(r["oracle"]), f"{stem}_km3_oracle")
        meta = dict(km_real=r["real_iou"], km_oracle=r["oracle_iou"], dendro={})
        for lk in ("ward", "average", "complete", "single"):
            m = cached_mask(stem, 3, f"768|free|{lk}|gap_border", gt.shape)
            save(mask_tile(m), f"{stem}_dendro_{lk}_gap")
            meta["dendro"][lk] = iou(m, gt)
        META["s3"][stem] = meta

    # ---- real image, one tree cut at k = 2, 3, 4 (slide 10): usod_test153 stride 14 ----
    img, gt = load("usod_test153")
    dense, _ = dense_features(model, img, SIZE, 14, device)
    grid = dense.shape[:2]
    ctx = FeatureContext(dense.reshape(-1, dense.shape[-1]), grid, device)
    Z = build_tree(dense.reshape(-1, dense.shape[-1]), grid, "average", False, "768")
    res = evaluate_tree(Z, ctx, False, 1)
    for k in (2, 3, 4):
        save(label_tile(res["labels"][k]), f"tree_cut_k{k}")

    # ---- failure case (slide 14): usod_test40, stride 3, free average tree, 768-D ----
    stem = "usod_test40"
    img, gt = load(stem)
    dense, _ = dense_features(model, img, SIZE, 3, device)
    grid = dense.shape[:2]
    X = dense.reshape(-1, dense.shape[-1])
    ctx = FeatureContext(X, grid, device)
    Zs = build_tree(X[ctx.sub] if ctx.subsampled else X, grid, "average", False, "768")
    res = evaluate_tree(Zs, ctx, False, round(PATCH / 3))
    sizes = {k: (np.bincount(res["labels"][k].ravel(), minlength=k) / res["labels"][k].size).round(4).tolist() for k in (2, 3, 4)}
    counts = {k: np.bincount(res["labels"][k].ravel(), minlength=k).tolist() for k in (2, 3, 4)}
    heights = Zs[::-1, 2][:6].tolist()
    META["failure"] = dict(k_sil=res["k_sil"], k_gap=res["k_gap"], sizes=sizes, counts=counts, n_patches=int(res["labels"][2].size), heights_top6=heights,
                           iou_gap=iou(cached_mask(stem, 3, "768|free|average|gap_border", gt.shape), gt),
                           iou_sil=iou(cached_mask(stem, 3, "768|free|average|sil_border", gt.shape), gt))
    save(img, f"{stem}_photo")
    save(mask_tile(gt), f"{stem}_gt")
    for tag, key in (("gap", "gap_border"), ("sil", "sil_border")):
        save(mask_tile(cached_mask(stem, 3, f"768|free|average|{key}", gt.shape)), f"{stem}_{tag}_mask")
    save(label_tile(res["labels"][2]), f"{stem}_k2_clusters")
    save(label_tile(res["labels"][4]), f"{stem}_k4_clusters")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dendrogram(Zs, truncate_mode="lastp", p=12, no_labels=True, ax=ax, color_threshold=0, above_threshold_color="#6FD3F2")
    dd = Zs[::-1, 2]
    ax.axhline((dd[0] + dd[1]) / 2, color="#E8788A", lw=2, ls="--")
    ax.axhline((dd[res["k_sil"] - 2] + dd[res["k_sil"] - 1]) / 2, color="#7BE0A8", lw=2)
    ax.text(0.99, (dd[0] + dd[1]) / 2, "gap cut (k=2)", color="#E8788A", ha="right", va="bottom", transform=ax.get_yaxis_transform(), fontsize=13)
    ax.text(0.99, (dd[res["k_sil"] - 2] + dd[res["k_sil"] - 1]) / 2, f"silhouette cut (k={res['k_sil']})", color="#7BE0A8", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("top 12 branches shown, lower ones collapsed\n(tree built on 3,000 sampled patches; the rest join the nearest one)", color="#A9B4C8", fontsize=11)
    ax.annotate(f"only {min(counts[2])} patches", xy=(5, 0.55), xytext=(14, 0.97), color="white", fontsize=14,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    style_dark(fig, ax, "merge height")
    fig.savefig(OUT / "failure_tree.png", dpi=170, transparent=True)
    plt.close(fig)

    # ---- toy tree (slide 10) ----
    X = np.array([[1.0], [1.1], [1.2], [3.0], [3.1], [8.0]])
    Zt = linkage(X, "average")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dendrogram(Zt, labels=["water", "water", "water", "reef", "reef", "fish"], ax=ax, color_threshold=0,
               above_threshold_color="#6FD3F2", leaf_font_size=14)
    ax.axhline(4.0, color="#E8788A", lw=2, ls="--")
    ax.axhline(1.0, color="#7BE0A8", lw=2)
    ax.text(0.01, 4.1, "cut here: 2 groups", color="#E8788A", transform=ax.get_yaxis_transform(), fontsize=13)
    ax.text(0.01, 1.1, "cut here: 3 groups", color="#7BE0A8", transform=ax.get_yaxis_transform(), fontsize=13)
    style_dark(fig, ax, "merge height")
    ax.tick_params(axis="x", colors="#E6EAF2", labelsize=13)
    fig.savefig(OUT / "toy_tree.png", dpi=170, transparent=True)
    plt.close(fig)

    (OUT / "meta.json").write_text(json.dumps(META, indent=1, default=float))
    print(json.dumps(META, indent=1, default=float)[:3000])


def style_dark(fig, ax, ylabel):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#6B7394")
    ax.tick_params(colors="#E6EAF2", labelsize=12)
    ax.set_ylabel(ylabel, color="#E6EAF2", fontsize=13)
    ax.set_xticks([]) if ax.get_xticklabels() == [] else None
    fig.tight_layout()


if __name__ == "__main__":
    main()
