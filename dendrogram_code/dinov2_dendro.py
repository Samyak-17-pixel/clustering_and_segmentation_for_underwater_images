"""Dendrogram (hierarchical) clustering of frozen DINOv2 patch features with label-free foreground selection.

For each image the backbone runs once; then every combination of feature set (768-D / PCA-RGB), spatial constraint
(free / grid-connectivity), linkage (ward / average / complete / single) builds one tree, and four label-free rules
turn it into a foreground mask. The GT is used only for the oracle and for the metrics.
"""
import argparse
import csv
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.cluster.hierarchy import dendrogram
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dendro_cluster import (KS, LINKAGES, RULES, build_tree, cut_labels, gap_k, make_sampling_grid, metrics_batch,
                            node_stats, rule_score, rule_subtree, silhouette_gpu, subset_matrix,
                            subsample_index, upsample_gpu)
from dinov2_cluster_union_stride import border_rule_union, dense_features, process_feature as kmeans_process
from dinov2_seg import PATCH, load_model, pca_masks
from gt_utils import GT_MODES, drop_degenerate, load_gt

FIELDS = ["image", "stride", "feat", "spatial", "linkage", "rule", "k", "iou", "f", "mae"]


class FeatureContext:
    """Per (image, feature set) quantities shared by all trees: subsample, distance matrix, nearest-leaf map."""

    def __init__(self, X, grid, device):
        self.grid, self.N = grid, len(X)
        self.X = torch.from_numpy(X).float().to(device)
        self.sub = subsample_index(self.N)
        self.subsampled = len(self.sub) < self.N
        Xs = self.X[torch.from_numpy(self.sub).to(device)]
        self.D = torch.cdist(Xs, Xs)
        self.sub_t = torch.from_numpy(self.sub).to(device)
        self.nn = None
        if self.subsampled:
            self.nn = torch.cat([torch.cdist(c, Xs).argmin(1) for c in self.X.split(8192)]).cpu().numpy()
        gh, gw = grid
        self.py, self.px = np.divmod(np.arange(self.N), gw)
        self.device = device


def evaluate_tree(Z, ctx, spatial, ring_r):
    """Cuts and masks of one tree. Returns per-k full labels (grid), silhouettes, and the four rule masks."""
    gh, gw = ctx.grid
    leaves = np.arange(ctx.N) if (spatial or not ctx.subsampled) else ctx.sub
    py, px = ctx.py[leaves], ctx.px[leaves]
    ring_leaf = (py < ring_r) | (py >= gh - ring_r) | (px < ring_r) | (px >= gw - ring_r)

    st = node_stats(Z, py, px, ring_leaf)
    cuts = cut_labels(Z, st)                                    # (n_leaves, len(KS))
    if spatial or not ctx.subsampled:
        full, lab_sub = cuts, cuts[ctx.sub]
    else:
        full, lab_sub = cuts[ctx.nn], cuts
    sil = {}
    for j, k in enumerate(KS):
        lab = torch.from_numpy(lab_sub[:, j]).long().to(ctx.device)
        sil[k] = silhouette_gpu(ctx.D, lab, k)
    k_sil = max(KS, key=sil.get)
    k_gap = gap_k(Z)
    labels = {k: full[:, j].reshape(gh, gw) for j, k in enumerate(KS)}

    to_full = (lambda m: m) if (spatial or not ctx.subsampled) else (lambda m: m[ctx.nn])
    masks = {}
    for name, k in (("sil_border", k_sil), ("gap_border", k_gap)):
        fg = border_rule_union(labels[k], k, ring_r)
        masks[name] = np.isin(labels[k], fg)
    lab2 = cuts[:, 0]
    masks["subtree"] = to_full(rule_subtree(st, Z, lab2, ring_leaf)).reshape(gh, gw)
    masks["score"] = to_full(rule_score(st, ctx.grid, lab2, ring_leaf)).reshape(gh, gw)
    return dict(labels=labels, sil=sil, k_sil=k_sil, k_gap=k_gap, masks=masks)


def score_tree(res, sampling_grid, gts, subsets, device):
    """Metrics rows (without identifiers) of one tree against every GT. gts: {name: bool tensor (P,)}."""
    onehots = [torch.from_numpy(res["labels"][k] == c).float() for k in KS for c in range(k)]
    rule_maps = [torch.from_numpy(res["masks"][r]).float() for r in RULES]
    maps = torch.stack(onehots + rule_maps).to(device)
    up = upsample_gpu(maps, sampling_grid).flatten(1)            # (20 + 4, P)
    out = {name: [] for name in gts}
    off, per_k = 0, {}
    for k in KS:
        per_k[k] = up[off:off + k]
        off += k
    rules_up = up[off:]
    for name, gt in gts.items():
        for i, r in enumerate(RULES):
            iou, f, mae = metrics_batch(rules_up[i:i + 1], gt)
            k = res["k_sil"] if r == "sil_border" else res["k_gap"] if r == "gap_border" else 0
            out[name].append((r, k, float(iou[0]), float(f[0]), float(mae[0])))
        for k in KS:
            iou, f, mae = metrics_batch(subsets[k] @ per_k[k], gt)
            j = int(iou.argmax())
            out[name].append(("oracle_k", k, float(iou[j]), float(f[j]), float(mae[j])))
            for tag, kk in (("oracle_sil", res["k_sil"]), ("oracle_gap", res["k_gap"])):
                if k == kk:
                    out[name].append((tag, k, float(iou[j]), float(f[j]), float(mae[j])))
    return out


def compare_figure(path, img, gt, km_mask, km_iou, masks, ious, feat, linkages, spatials, grid_hw):
    """Rows = rules, columns = image / GT / k-means / every (spatial, linkage) variant."""
    variants = list(product(spatials, linkages))
    fig, axs = plt.subplots(len(RULES), 3 + len(variants), figsize=(2.6 * (3 + len(variants)), 2.2 * len(RULES)))
    for ax in axs.ravel():
        ax.axis("off")
    axs[0, 0].imshow(img), axs[0, 0].set_title("image", fontsize=8)
    axs[0, 1].imshow(gt, cmap="gray"), axs[0, 1].set_title("GT", fontsize=8)
    axs[0, 2].imshow(km_mask, cmap="gray"), axs[0, 2].set_title(f"k-means+border\nIoU={km_iou:.2f}", fontsize=8)
    for j, (sp, lk) in enumerate(variants):
        for i, r in enumerate(RULES):
            ax = axs[i, 3 + j]
            ax.imshow(masks[feat, sp, lk, r], cmap="gray", interpolation="nearest")
            ax.set_title(f"{'constr' if sp else 'free'} {lk}\n{r} IoU={ious[feat, sp, lk, r]:.2f}", fontsize=7)
    fig.suptitle(f"feature set: {feat}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=50)
    plt.close(fig)


def dendrogram_figure(path, trees, feat, linkages, spatials):
    fig, axs = plt.subplots(len(spatials), len(linkages), figsize=(4 * len(linkages), 3 * len(spatials)), squeeze=False)
    for i, sp in enumerate(spatials):
        for j, lk in enumerate(linkages):
            Z, k_sil, k_gap = trees[feat, sp, lk]
            ax = axs[i, j]
            dendrogram(Z, truncate_mode="lastp", p=25, no_labels=True, ax=ax, color_threshold=0)
            d = Z[::-1, 2]
            ax.axhline((d[k_sil - 2] + d[k_sil - 1]) / 2, color="r", lw=1, label=f"silhouette k*={k_sil}")
            ax.axhline((d[k_gap - 2] + d[k_gap - 1]) / 2, color="g", lw=1, ls="--", label=f"gap k={k_gap}")
            ax.set_title(f"{'constrained' if sp else 'free'} {lk}", fontsize=9)
            ax.legend(fontsize=6)
    fig.suptitle(f"truncated dendrograms ({feat})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=REPO / "USOD")
    ap.add_argument("--gts", nargs="+", choices=GT_MODES, default=["usod"])
    ap.add_argument("--tag", default=None, help="cache name shared by the GTs of one image set (default: first GT)")
    ap.add_argument("--out-root", default=REPO / "dendrogram_results")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=588)
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--features", nargs="+", default=["768", "rgb"], choices=["768", "rgb"])
    ap.add_argument("--linkages", nargs="+", default=list(LINKAGES), choices=LINKAGES)
    ap.add_argument("--spatial", nargs="+", default=["free", "constrained"], choices=["free", "constrained"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-water-frac", type=float, default=0.02)
    ap.add_argument("--sample", type=int, default=10, help="number of fixed sample images for comparison figures")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data, out_root = Path(args.data), Path(args.out_root)
    size = (args.width, args.height)
    spatials = [s == "constrained" for s in args.spatial]
    ring_r = round(PATCH / args.stride)
    tag = args.tag or args.gts[0]
    cache_dir = out_root / "_cache" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {g: out_root / g for g in args.gts}
    keep = {}
    all_paths = sorted((data / "images").glob("*.jpg"))
    for g, d in out_dirs.items():
        for sub in ("compare", "dendrograms"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        keep[g] = {p.name for p in drop_degenerate(all_paths, data, g, args.min_water_frac, d)}
    paths = [p for p in all_paths if any(p.name in keep[g] for g in args.gts)]
    sample_every = max(len(paths) // args.sample, 1)
    sample_names = {p.name for p in paths[::sample_every][: args.sample]}
    if args.limit:
        paths = paths[: args.limit]
        sample_names &= {p.name for p in paths}

    csv_paths = {g: out_dirs[g] / f"metrics_s{args.stride}.csv" for g in args.gts}
    done = {g: set() for g in args.gts}
    for g, cp in csv_paths.items():
        if args.resume and cp.exists():
            done[g] = {r["image"] for r in csv.DictReader(open(cp))}
        else:
            with open(cp, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    subsets = {k: subset_matrix(k, device) for k in KS}
    km_args = SimpleNamespace(kmax=6, stride=args.stride)

    for p in tqdm(paths):
        active = [g for g in args.gts if p.name in keep[g] and p.name not in done[g]]
        if not active:
            continue
        img = Image.open(p).convert("RGB")
        gts_np = {g: load_gt(data, p.stem, g) for g in active}
        out_hw = next(iter(gts_np.values())).shape
        gts = {g: torch.from_numpy(a).flatten().to(device) for g, a in gts_np.items()}
        dense, _ = dense_features(model, img, size, args.stride, device)
        grid = dense.shape[:2]
        _, rgb = pca_masks(dense)
        sets = {"768": dense.reshape(-1, dense.shape[-1]), "rgb": rgb.reshape(-1, 3)}
        sampling_grid = make_sampling_grid(out_hw, size, grid, args.stride, device)
        is_sample = p.name in sample_names

        rows = {g: [] for g in active}
        cache, masks_all, ious_all = {}, {}, {g: {} for g in active}
        trees = {}
        for feat in args.features:
            ctx = FeatureContext(sets[feat], grid, device)
            for sp, lk in product(spatials, args.linkages):
                n_leaves = ctx.N if (sp or not ctx.subsampled) else len(ctx.sub)
                Z = build_tree(sets[feat][ctx.sub] if (not sp and ctx.subsampled) else sets[feat], grid, lk, sp, feat)
                assert len(Z) == n_leaves - 1
                res = evaluate_tree(Z, ctx, sp, ring_r)
                scored = score_tree(res, sampling_grid, gts, subsets, device)
                sp_name = "constrained" if sp else "free"
                for g in active:
                    for rule, k, iou, f, mae in scored[g]:
                        rows[g].append(dict(image=p.name, stride=args.stride, feat=feat, spatial=sp_name,
                                            linkage=lk, rule=rule, k=k, iou=iou, f=f, mae=mae))
                        if rule in RULES:
                            ious_all[g][feat, sp, lk, rule] = iou
                for r in RULES:
                    cache[f"{feat}|{sp_name}|{lk}|{r}"] = np.packbits(res["masks"][r])
                    masks_all[feat, sp, lk, r] = res["masks"][r]
                if is_sample:
                    trees[feat, sp, lk] = (Z, res["k_sil"], res["k_gap"])
        np.savez_compressed(cache_dir / f"{p.stem}_s{args.stride}.npz", grid=np.array(grid), **cache)

        if is_sample:
            for feat in args.features:
                for g in active:
                    dendrogram_figure(out_dirs[g] / "dendrograms" / f"{p.stem}_s{args.stride}_{feat}.png",
                                      trees, feat, args.linkages, spatials)
                labels_km = kmeans_process(sets[feat], grid, next(iter(gts_np.values())), km_args, size)
                ks = labels_km["k_star"]
                km_mask = np.isin(labels_km["labels"][ks], border_rule_union(labels_km["labels"][ks], ks, ring_r))
                km_up = upsample_gpu(torch.from_numpy(km_mask).float()[None].to(device), sampling_grid).flatten(1)
                for g in active:
                    km_iou = float(metrics_batch(km_up, gts[g])[0][0])
                    compare_figure(out_dirs[g] / "compare" / f"{p.stem}_s{args.stride}_{feat}.png", img, gts_np[g],
                                   km_mask, km_iou, masks_all, ious_all[g], feat, args.linkages, spatials, grid)

        for g in active:
            with open(csv_paths[g], "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerows(rows[g])


if __name__ == "__main__":
    main()
