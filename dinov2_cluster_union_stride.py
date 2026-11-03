"""Dense (overlapping-window) DINOv2 features + k=2..kmax clustering with oracle fg/bg union search.

Stride < 14 is obtained by running the unchanged DINOv2 on shifted crops (offsets 0, s, 2s, ... < 14 on both
axes) and interleaving the patch tokens on a grid whose cells are `stride` pixels apart.
stride=14 gives the original single-pass grid.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from gt_utils import GT_MODES, drop_degenerate, load_gt

from dinov2_cluster_union import subsets
from dinov2_seg import MEAN, PATCH, ROOT, STD, load_model, metrics, pca_masks

KFIT = 5000   # k-means is fit on at most this many patches, then all patches are assigned
SIL = 3000    # silhouette uses at most this many patches


@torch.no_grad()
def dense_features(model, img, size, stride, device):
    """Returns dense (Hd, Wd, D) float32 features with cell centres at 7 + stride*m, and the (0,0)-offset grid."""
    W, H = size
    x = torch.from_numpy(np.array(img.resize((W, H), Image.BICUBIC))).permute(2, 0, 1)[None].float() / 255
    x = ((x - MEAN) / STD).half().to(device)
    offsets = list(range(0, PATCH, stride))
    feats = {}
    for a, dy in enumerate(offsets):
        for b, dx in enumerate(offsets):
            h, w = (H - dy) // PATCH * PATCH, (W - dx) // PATCH * PATCH
            t = model.forward_features(x[:, :, dy:dy + h, dx:dx + w])["x_norm_patchtokens"][0]
            feats[a, b] = t.float().cpu().numpy().reshape(h // PATCH, w // PATCH, -1)

    def axis_map(n_out, length):
        centres = PATCH // 2 + stride * np.arange(n_out)
        best_d, best_i, best_err = np.zeros(n_out, int), np.zeros(n_out, int), np.full(n_out, np.inf)
        for a, d in enumerate(offsets):
            n_tok = (length - d) // PATCH
            i = np.clip(np.round((centres - PATCH // 2 - d) / PATCH).astype(int), 0, n_tok - 1)
            err = np.abs(d + PATCH // 2 + PATCH * i - centres)
            better = err < best_err
            best_d[better], best_i[better], best_err[better] = a, i[better], err[better]
        return best_d, best_i

    hd, wd = (H - PATCH) // stride + 1, (W - PATCH) // stride + 1
    ay, ty = axis_map(hd, H)
    ax, tx = axis_map(wd, W)
    dense = np.zeros((hd, wd, feats[0, 0].shape[-1]), np.float32)
    for (a, b), f in feats.items():
        rows, cols = np.where(ay == a)[0], np.where(ax == b)[0]
        if len(rows) and len(cols):
            dense[np.ix_(rows, cols)] = f[ty[rows][:, None], tx[cols][None, :]]
    return dense, feats[0, 0]


def upsample_maps(maps, out_hw, stride, size):
    """Bilinear sampling of (k, gh, gw) grids (cell centres at 7 + stride*m in the resized image) at GT pixels."""
    W, H = size
    k, gh, gw = maps.shape
    oh, ow = out_hw

    def coords(n_out, length, n_grid):
        m = ((np.arange(n_out) + 0.5) * length / n_out - PATCH / 2) / stride
        m = np.clip(m, 0, n_grid - 1)
        return 2 * m / max(n_grid - 1, 1) - 1

    gy, gx = np.meshgrid(coords(oh, H, gh), coords(ow, W, gw), indexing="ij")
    grid = torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32))[None]
    t = torch.from_numpy(maps.astype(np.float32))[None]
    return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)[0].numpy()


def border_ring(shape, r):
    b = np.zeros(shape, bool)
    b[:r, :] = b[-r:, :] = b[:, :r] = b[:, -r:] = True
    return b


def border_rule_union(labels, k, r):
    ring = border_ring(labels.shape, r)
    frac = np.array([(ring & (labels == c)).sum() / max((labels == c).sum(), 1) for c in range(k)])
    fg = [c for c in range(k) if frac[c] < ring.mean()]
    return tuple(fg) if 0 < len(fg) < k else (int(frac.argmin()),)


def cluster(X, k):
    if len(X) <= KFIT:
        return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X)
    idx = np.random.default_rng(0).choice(len(X), KFIT, replace=False)
    return KMeans(n_clusters=k, n_init=4, random_state=0).fit(X[idx]).predict(X)


def process_feature(X, grid, gt, args, size):
    ks = range(2, args.kmax + 1)
    labels, sil, oracle = {}, {}, {}
    for k in ks:
        lab = cluster(X, k)
        labels[k] = lab.reshape(grid)
        sil[k] = float(silhouette_score(X, lab, sample_size=SIL if len(X) > SIL else None, random_state=0))
        onehot = np.stack([labels[k] == c for c in range(k)])
        cmaps = upsample_maps(onehot, gt.shape, args.stride, size)
        best = None
        for s in subsets(k):
            prob = cmaps[list(s)].sum(0)
            iou, f, mae = metrics(prob, gt)
            if best is None or iou > best["iou"]:
                best = dict(iou=iou, f=f, mae=mae, fg=s, prob=prob)
        oracle[k] = best
    k_star = max(ks, key=sil.get)
    s = border_rule_union(labels[k_star], k_star, round(PATCH / args.stride))
    prob = upsample_maps(np.isin(labels[k_star], s)[None], gt.shape, args.stride, size)[0]
    return dict(labels=labels, sil=sil, oracle=oracle, k_star=k_star, border=metrics(prob, gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "USOD")
    ap.add_argument("--out", default=ROOT / "dinov2_results" / "cluster_union_stride3")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=588)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt", choices=GT_MODES, default="usod")
    ap.add_argument("--min-water-frac", type=float, default=0.0,
                    help="suim_nonwater only: skip images with less water than this (all-foreground GT)")
    args = ap.parse_args()
    ks = list(range(2, args.kmax + 1))
    size = (args.width, args.height)

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
        dense, base = dense_features(model, img, size, args.stride, device)
        grid = dense.shape[:2]
        _, rgb = pca_masks(dense)
        _, rgb14 = pca_masks(base)
        sets = {"rgb": rgb.reshape(-1, 3), "768": dense.reshape(-1, dense.shape[-1])}

        res = {name: process_feature(X, grid, gt, args, size) for name, X in sets.items()}
        for name, r in res.items():
            o = r["oracle"][r["k_star"]]
            Image.fromarray(((o["prob"] > 0.5) * 255).astype(np.uint8)).save(out / f"masks_{name}" / f"{p.stem}.png")
            row = {"image": p.name, "feat": name, "k_star": r["k_star"]}
            for k in ks:
                row[f"sil{k}"] = r["sil"][k]
                row[f"oracle_iou_k{k}"] = r["oracle"][k]["iou"]
            row.update(oracle_iou=o["iou"], oracle_f=o["f"], oracle_mae=o["mae"],
                       oracle_fg="+".join(map(str, o["fg"])),
                       border_iou=r["border"][0], border_f=r["border"][1], border_mae=r["border"][2])
            rows.append(row)

        fig, axs = plt.subplots(1, 8, figsize=(32, 4))
        axs[0].imshow(img), axs[0].set_title("image")
        axs[1].imshow(rgb14), axs[1].set_title("PCA RGB (stride 14)")
        axs[2].imshow(rgb), axs[2].set_title(f"PCA RGB (stride {args.stride})")
        for j, name in enumerate(("rgb", "768")):
            r = res[name]
            o = r["oracle"][r["k_star"]]
            axs[3 + 2 * j].imshow(r["labels"][r["k_star"]], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
            axs[3 + 2 * j].set_title(f"{name} clusters k*={r['k_star']}")
            axs[4 + 2 * j].imshow(o["prob"] > 0.5, cmap="gray")
            axs[4 + 2 * j].set_title(f"{name} best union {'+'.join(map(str, o['fg']))} IoU={o['iou']:.2f}\n"
                                     f"(border rule IoU={r['border'][0]:.2f})")
        axs[7].imshow(gt, cmap="gray"), axs[7].set_title("GT")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "panels" / f"{p.stem}.png", dpi=55)
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
        print(f"[{name}] oracle IoU by fixed k: " + ", ".join(f"k={k}: {m(f'oracle_iou_k{k}'):.4f}" for k in ks))
        ranked = sorted(rs, key=lambda r: -r["oracle_iou"])
        print(f"[{name}] top5", [(r["image"], round(r["oracle_iou"], 2)) for r in ranked[:5]])
        print(f"[{name}] bottom5", [(r["image"], round(r["oracle_iou"], 2)) for r in ranked[-5:]])


if __name__ == "__main__":
    main()
