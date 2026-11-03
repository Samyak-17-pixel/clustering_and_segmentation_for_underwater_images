"""Side-by-side tables, statistics vs the k-means baseline, and best-variant panels for the dendrogram runs.

    python src/dendrogram/summarize_dendro.py --gt usod --strides 14 3
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "common"))

from gt_utils import load_gt

RULES = ("sil_border", "gap_border", "subtree", "score")
BASELINE_DIR = {"usod": "usod", "suim_saliency": "suim_saliency",
                "suim_nonwater": "suim"}
DEFAULT_DATA = {"usod": REPO / "data" / "USOD", "suim_saliency": REPO / "data" / "SUIM" / "TEST", "suim_nonwater": REPO / "data" / "SUIM" / "TEST"}


def read_rows(path):
    return list(csv.DictReader(open(path)))


def baseline(gt, stride):
    d = REPO / "results" / "kmeans" / BASELINE_DIR[gt] / f"cluster_union_stride{stride}_k6"
    out = defaultdict(dict)
    for r in read_rows(d / "metrics_union.csv"):
        out[r["feat"]][r["image"]] = (float(r["border_iou"]), float(r["oracle_iou"]))
    return out, d


def paired_bootstrap(diff, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def summarize(gt, stride, out_dir):
    rows = read_rows(out_dir / f"metrics_s{stride}.csv")
    base, _ = baseline(gt, stride)
    per = defaultdict(dict)                    # (feat, spatial, linkage, rule) -> {image: (iou, f, mae)}
    for r in rows:
        rule = r["rule"] if r["rule"] != "oracle_k" else f"oracle_k{r['k']}"
        per[r["feat"], r["spatial"], r["linkage"], rule][r["image"]] = (float(r["iou"]), float(r["f"]), float(r["mae"]))

    cols = list(RULES) + ["oracle_sil", "oracle_gap"]
    table, stats = [], []
    for feat in ("768", "rgb"):
        for sp in ("free", "constrained"):
            for lk in ("ward", "average", "complete", "single"):
                if (feat, sp, lk, RULES[0]) not in per:
                    continue
                row = dict(stride=stride, feat=feat, spatial=sp, linkage=lk)
                for c in cols:
                    d = per[feat, sp, lk, c]
                    row[f"{c}_iou"] = np.mean([v[0] for v in d.values()])
                    if c in RULES:
                        row[f"{c}_f"] = np.mean([v[1] for v in d.values()])
                        row[f"{c}_mae"] = np.mean([v[2] for v in d.values()])
                row["n"] = len(per[feat, sp, lk, RULES[0]])
                table.append(row)
                for c in RULES + ("oracle_sil",):
                    d = per[feat, sp, lk, c]
                    imgs = [i for i in d if i in base[feat]]
                    new = np.array([d[i][0] for i in imgs])
                    old = np.array([base[feat][i][1 if c.startswith("oracle") else 0] for i in imgs])
                    diff = new - old
                    lo, hi = paired_bootstrap(diff)
                    try:
                        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                    except ValueError:
                        p = 1.0
                    stats.append(dict(stride=stride, feat=feat, spatial=sp, linkage=lk, rule=c, n=len(imgs),
                                      mean_iou=new.mean(), kmeans_iou=old.mean(), diff=diff.mean(),
                                      ci_lo=lo, ci_hi=hi, wilcoxon_p=p))
    for name, data in (("summary", table), ("stats", stats)):
        with open(out_dir / f"{name}_s{stride}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=data[0].keys())
            w.writeheader()
            w.writerows(data)

    km = {f: (np.mean([v[0] for v in base[f].values()]), np.mean([v[1] for v in base[f].values()])) for f in base}
    lines = [f"# {gt}, stride {stride}: mean IoU (dendrogram variants vs k-means baseline)", "",
             f"k-means baseline (k<=6, border rule / oracle at silhouette k*): "
             + ", ".join(f"{f}: {a:.3f} / {b:.3f}" for f, (a, b) in km.items()), "",
             "| feat | spatial | linkage | n | " + " | ".join(cols) + " |", "|" + "---|" * (4 + len(cols))]
    for r in table:
        lines.append(f"| {r['feat']} | {r['spatial']} | {r['linkage']} | {r['n']} | "
                     + " | ".join(f"{r[c + '_iou']:.3f}" for c in cols) + " |")
    (out_dir / f"summary_s{stride}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    best = max((r for r in stats if r["rule"] in RULES), key=lambda r: r["mean_iou"])
    print(f"best label-free variant: {best['feat']} {best['spatial']} {best['linkage']} {best['rule']} "
          f"IoU={best['mean_iou']:.3f} (k-means {best['kmeans_iou']:.3f}, diff {best['diff']:+.3f} "
          f"[{best['ci_lo']:+.3f}, {best['ci_hi']:+.3f}], Wilcoxon p={best['wilcoxon_p']:.3g})")
    return best, per


def best_panels(gt, stride, out_dir, best, per, data, tag, cache_root):
    panel_dir = out_dir / "panels_best" / f"s{stride}_{best['feat']}_{best['spatial']}_{best['linkage']}_{best['rule']}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    _, base_dir = baseline(gt, stride)
    key = f"{best['feat']}|{best['spatial']}|{best['linkage']}|{best['rule']}"
    scores = per[best["feat"], best["spatial"], best["linkage"], best["rule"]]
    for name, (iou, _, _) in tqdm(scores.items(), desc="best-variant panels"):
        stem = Path(name).stem
        cache = cache_root / tag / f"{stem}_s{stride}.npz"
        if not cache.exists():
            continue
        z = np.load(cache)
        gh, gw = z["grid"]
        mask = np.unpackbits(z[key])[: gh * gw].reshape(gh, gw).astype(bool)
        img = Image.open(data / "images" / name).convert("RGB")
        gt_mask = load_gt(data, stem, gt)
        pred = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(gt_mask.shape[::-1], Image.NEAREST)) > 0
        km_path = base_dir / f"masks_{best['feat']}" / f"{stem}.png"
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))
        panels = [(img, "image"), (gt_mask, "GT"),
                  (pred, f"{best['linkage']} {best['spatial']} / {best['rule']}\nIoU={iou:.2f}")]
        if km_path.exists():
            panels.append((np.asarray(Image.open(km_path).convert("L")) > 127, "k-means oracle best union"))
        for ax, (im, title) in zip(axs, panels):
            ax.imshow(im, cmap=None if im is img else "gray")
            ax.set_title(title, fontsize=9)
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(panel_dir / f"{stem}.png", dpi=55)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, choices=list(BASELINE_DIR))
    ap.add_argument("--strides", nargs="+", type=int, default=[14])
    ap.add_argument("--out-root", default=REPO / "results" / "dendrogram")
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-panels", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    out_dir = out_root / args.gt
    data = Path(args.data) if args.data else DEFAULT_DATA[args.gt]
    tag = args.tag or ("usod" if args.gt == "usod" else "suim")
    for stride in args.strides:
        best, per = summarize(args.gt, stride, out_dir)
        if not args.no_panels:
            best_panels(args.gt, stride, out_dir, best, per, data, tag, out_root / "_cache")


if __name__ == "__main__":
    main()
