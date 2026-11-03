"""Quantitative and visual summary of the existing DINOv2 clustering results (CPU only, no model needed).

Reads results/usod/metrics.csv (E1) and results/usod/*/metrics_union.csv (E2-E4) plus the USOD ground-truth
masks, and writes figures to docs/figures/analysis/ and tables to results/usod/analysis/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = {  # run id -> (csv directory, short description)
    "E2": ("cluster_union_stride14_k4", "stride 14, k<=4"),
    "E4": ("cluster_union_stride14_k6", "stride 14, k<=6"),
    "E3": ("cluster_union_stride3_k6", "stride 3, k<=6"),
}
FEATS = {"rgb": "PCA-RGB", "768": "768-D"}
CASES = ("usod_test174", "usod_test205", "usod_test219", "usod_test278", "usod_test54", "usod_test106")
CATS = ("success", "selection failure", "cluster failure")
CAT_COLORS = {"success": "tab:green", "selection failure": "tab:orange", "cluster failure": "tab:red"}
SIZE_BINS = [0, 0.02, 0.10, 0.30, 1.01]
SIZE_LABELS = ["tiny (<2%)", "small (2-10%)", "medium (10-30%)", "large (>30%)"]
N_BOOT = 2000


def load_long(res):
    """One row per (image, method): run, feat, mode, iou, f, mae, k_star."""
    rows = []
    e1 = pd.read_csv(res / "metrics.csv")
    for name, label in (("pca", "PC1>0"), ("kmeans", "k-means k=2")):
        rows.append(pd.DataFrame(dict(image=e1["image"], run="E1", feat=label, mode="border",
                                      iou=e1[f"{name}_iou"], f=e1[f"{name}_f"], mae=e1[f"{name}_mae"], k_star=2)))
    for run, (sub, _) in RUNS.items():
        df = pd.read_csv(res / sub / "metrics_union.csv", dtype={"feat": str})
        for mode in ("oracle", "border"):
            rows.append(pd.DataFrame(dict(image=df["image"], run=run, feat=df["feat"].map(FEATS), mode=mode,
                                          iou=df[f"{mode}_iou"], f=df[f"{mode}_f"], mae=df[f"{mode}_mae"],
                                          k_star=df["k_star"])))
    out = pd.concat(rows, ignore_index=True)
    out["image"] = out["image"].str.replace(".jpg", "", regex=False)
    out["method"] = out["run"] + " " + out["feat"] + " " + out["mode"]
    return out


def load_wide(res):
    """Per (run, feat) frame with the per-k oracle columns and silhouettes."""
    frames = {}
    for run, (sub, _) in RUNS.items():
        df = pd.read_csv(res / sub / "metrics_union.csv", dtype={"feat": str})
        df["image"] = df["image"].str.replace(".jpg", "", regex=False)
        for feat, label in FEATS.items():
            frames[run, label] = df[df["feat"] == feat].set_index("image").sort_index()
    return frames


def gt_attributes(mask_dir, images):
    """Ground-truth difficulty factors: object area fraction, touches the image edge, connected components."""
    rows = []
    for im in images:
        g = np.asarray(Image.open(mask_dir / f"{im}.png").convert("L")) > 128
        edge = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
        rows.append(dict(image=im, area=g.mean(), touches_border=bool(edge.any()),
                         edge_frac=float(edge.mean()), n_components=int(ndimage.label(g)[1])))
    df = pd.DataFrame(rows).set_index("image")
    df["size_bin"] = pd.cut(df["area"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    return df


def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def category(oracle, border, thr=0.5):
    return np.where(border >= thr, CATS[0], np.where(oracle >= thr, CATS[1], CATS[2]))


def method_order(long):
    order = ["E1 PC1>0 border", "E1 k-means k=2 border"]
    for run in RUNS:
        for feat in FEATS.values():
            for mode in ("border", "oracle"):
                order.append(f"{run} {feat} {mode}")
    return [m for m in order if m in set(long["method"])]


def summary_table(long, rng):
    rows = []
    for m in method_order(long):
        d = long[long["method"] == m]
        row = dict(method=m, run=d["run"].iat[0], feat=d["feat"].iat[0], mode=d["mode"].iat[0], n=len(d))
        for k in ("iou", "f", "mae"):
            lo, hi = boot_ci(d[k], rng)
            row.update({f"{k}_mean": d[k].mean(), f"{k}_ci_lo": lo, f"{k}_ci_hi": hi})
        row.update(iou_median=d["iou"].median(), frac_iou_gt_0_5=(d["iou"] > 0.5).mean(),
                   frac_iou_lt_0_2=(d["iou"] < 0.2).mean(), mean_k_star=d["k_star"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired(long, a, b, rng, metric="iou"):
    x = long[long["method"] == a].set_index("image")[metric].sort_index()
    y = long[long["method"] == b].set_index("image")[metric].sort_index()
    d = (x - y).to_numpy()
    lo, hi = boot_ci(d, rng)
    try:
        p = float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        p = float("nan")
    return dict(a=a, b=b, metric=metric, mean_a=x.mean(), mean_b=y.mean(), mean_diff=d.mean(),
                diff_ci_lo=lo, diff_ci_hi=hi, wilcoxon_p=p, n_a_better=int((d > 1e-9).sum()),
                n_b_better=int((d < -1e-9).sum()), n_tied=int((np.abs(d) <= 1e-9).sum()))


def significance_table(long, rng):
    pairs = [
        ("E3 768-D border", "E3 PCA-RGB border"), ("E3 768-D oracle", "E3 PCA-RGB oracle"),
        ("E2 768-D border", "E2 PCA-RGB border"),
        ("E3 768-D border", "E4 768-D border"), ("E3 768-D oracle", "E4 768-D oracle"),
        ("E3 PCA-RGB border", "E4 PCA-RGB border"), ("E3 PCA-RGB oracle", "E4 PCA-RGB oracle"),
        ("E4 768-D border", "E2 768-D border"), ("E4 768-D oracle", "E2 768-D oracle"),
        ("E4 PCA-RGB border", "E2 PCA-RGB border"),
        ("E3 768-D oracle", "E3 768-D border"),
        ("E2 768-D border", "E1 k-means k=2 border"), ("E3 768-D border", "E1 k-means k=2 border"),
        ("E1 k-means k=2 border", "E1 PC1>0 border"),
    ]
    return pd.DataFrame([paired(long, a, b, rng) for a, b in pairs])


# ---------------------------------------------------------------- figures

def fig_method_comparison(summ, out):
    colors = {"border": "tab:blue", "oracle": "tab:gray"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    y = np.arange(len(summ))[::-1]
    for ax, (k, title) in zip(axs, (("iou", "Mean IoU (higher is better)"), ("f", "Mean F-measure, beta^2=0.3 (higher is better)"),
                                    ("mae", "Mean MAE (lower is better)"))):
        mean = summ[f"{k}_mean"].to_numpy()
        err = np.vstack([mean - summ[f"{k}_ci_lo"], summ[f"{k}_ci_hi"] - mean])
        c = [colors[m] if r != "E1" else "tab:purple" for m, r in zip(summ["mode"], summ["run"])]
        ax.barh(y, mean, xerr=err, color=c, alpha=0.85, capsize=3)
        for yi, v, hi in zip(y, mean, summ[f"{k}_ci_hi"]):
            ax.text(hi + 0.01, yi, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlim(0, 1 if k != "mae" else 0.45)
        ax.grid(axis="x", alpha=0.3)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(summ["method"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("tab:purple", "tab:blue", "tab:gray")]
    fig.legend(handles, ["E1 baseline (border rule)", "border rule (no ground truth)", "oracle union (uses ground truth)"],
               loc="lower center", ncol=3)
    fig.suptitle("All methods on USOD (300 images): mean with 95% bootstrap CI over images")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(out / "method_comparison.png", dpi=110)
    plt.close(fig)


def fig_iou_distribution(long, out):
    methods = ["E1 k-means k=2 border", "E3 PCA-RGB border", "E3 768-D border", "E3 PCA-RGB oracle", "E3 768-D oracle"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    t = np.linspace(0, 1, 201)
    for m in methods:
        v = long.loc[long["method"] == m, "iou"].to_numpy()
        ax1.plot(t, [(v >= x).mean() for x in t], label=m, lw=2, ls="--" if "oracle" in m else "-")
    ax1.set_xlabel("IoU threshold t")
    ax1.set_ylabel("Fraction of images with IoU >= t")
    ax1.set_title("Survival curve of per-image IoU")
    ax1.grid(alpha=0.3)
    ax1.legend()
    data = [long.loc[long["method"] == m, "iou"].to_numpy() for m in methods]
    parts = ax2.violinplot(data, showmedians=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, 1):
        ax2.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=4, alpha=0.4, color="k")
    ax2.set_xticks(range(1, len(methods) + 1))
    ax2.set_xticklabels([m.replace(" ", "\n", 2) for m in methods], fontsize=9)
    ax2.set_ylabel("Per-image IoU")
    ax2.set_title("Distribution of per-image IoU (bar = median, dots = images)")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "iou_distribution.png", dpi=110)
    plt.close(fig)


def fig_oracle_vs_border(frames, out, run="E3"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, feat in zip(axs, FEATS.values()):
        d = frames[run, feat]
        cat = category(d["oracle_iou"].to_numpy(), d["border_iou"].to_numpy())
        for c in CATS:
            sel = cat == c
            ax.scatter(d["oracle_iou"][sel], d["border_iou"][sel], s=14, alpha=0.7, color=CAT_COLORS[c],
                       label=f"{c} ({sel.sum()})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.axvline(0.5, color="gray", lw=0.6, ls=":")
        for im in CASES:
            if im in d.index:
                ax.annotate(im.replace("usod_", ""), (d.at[im, "oracle_iou"], d.at[im, "border_iou"]), fontsize=8,
                            xytext=(4, 4), textcoords="offset points")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Oracle IoU (best fg/bg union, uses ground truth)")
        ax.set_ylabel("Border-rule IoU (no ground truth)")
        ax.set_title(f"{run} ({RUNS[run][1]}), {feat}: gap = {(d['oracle_iou'] - d['border_iou']).mean():.3f}")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle("Each dot is one image. Below the diagonal = IoU lost by choosing the foreground without labels")
    fig.tight_layout()
    fig.savefig(out / "oracle_vs_border.png", dpi=110)
    plt.close(fig)


def fig_failure_breakdown(frames, out):
    labels, counts = [], []
    for run in RUNS:
        for feat in FEATS.values():
            d = frames[run, feat]
            cat = category(d["oracle_iou"].to_numpy(), d["border_iou"].to_numpy())
            labels.append(f"{run} {feat}\n({RUNS[run][1]})")
            counts.append([(cat == c).sum() for c in CATS])
    counts = np.array(counts)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    left = np.zeros(len(labels))
    y = np.arange(len(labels))[::-1]
    for j, c in enumerate(CATS):
        ax.barh(y, counts[:, j], left=left, color=CAT_COLORS[c], label=c)
        for yi, l, v in zip(y, left, counts[:, j]):
            if v >= 12:
                ax.text(l + v / 2, yi, str(v), ha="center", va="center", color="white", fontsize=9)
        left += counts[:, j]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of images (out of 300)")
    ax.set_xlim(0, 300)
    ax.set_title("Why images fail: success = border-rule IoU >= 0.5; selection failure = oracle >= 0.5 but border rule < 0.5;\n"
                 "cluster failure = even the best union of clusters has IoU < 0.5", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "failure_breakdown.png", dpi=110)
    plt.close(fig)


def fig_oracle_vs_k(frames, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    styles = {"E2": ":", "E4": "--", "E3": "-"}
    colors = {"PCA-RGB": "tab:orange", "768-D": "tab:blue"}
    for (run, feat), d in frames.items():
        ks = sorted(int(c.split("k")[-1]) for c in d.columns if c.startswith("oracle_iou_k"))
        ax1.plot(ks, [d[f"oracle_iou_k{k}"].mean() for k in ks], marker="o", ls=styles[run], color=colors[feat],
                 label=f"{run} {feat} oracle at fixed k")
    for feat, c in colors.items():
        ax1.axhline(frames["E3", feat]["border_iou"].mean(), color=c, lw=1, alpha=0.6, ls="-.",
                    label=f"E3 {feat} border rule at k* (realistic)")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Mean IoU over 300 images")
    ax1.set_title("Oracle IoU grows with k (more unions to choose from); realistic score does not")
    ax1.set_xticks(range(2, 7))
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    width = 0.13
    keys = list(frames)
    for i, (run, feat) in enumerate(keys):
        vc = frames[run, feat]["k_star"].value_counts().reindex(range(2, 7), fill_value=0)
        ax2.bar(np.arange(2, 7) + (i - (len(keys) - 1) / 2) * width, vc.values, width, color=colors[feat],
                alpha=[0.35, 0.65, 1.0][list(RUNS).index(run)], label=f"{run} {feat}")
    ax2.set_xlabel("k* chosen by silhouette")
    ax2.set_ylabel("Number of images")
    ax2.set_title("Silhouette choice of k*: PCA-RGB prefers 4, 768-D is bimodal (2 or 6)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "oracle_vs_k.png", dpi=110)
    plt.close(fig)


def fig_stride_paired(frames, out):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, feat in enumerate(FEATS.values()):
        for j, mode in enumerate(("oracle", "border")):
            a, b = frames["E3", feat][f"{mode}_iou"], frames["E4", feat][f"{mode}_iou"]
            d = (a - b).to_numpy()
            p = stats.wilcoxon(a, b).pvalue
            ax = axs[i, j]
            ax.hist(d, bins=np.linspace(-0.6, 0.6, 49), color="tab:blue" if feat == "768-D" else "tab:orange", alpha=0.8)
            ax.axvline(0, color="k", lw=1)
            ax.axvline(d.mean(), color="tab:red", lw=1.5, ls="--", label=f"mean {d.mean():+.3f}")
            ax.set_title(f"{feat}, {mode}: stride 3 better on {(d > 0).mean():.0%} of images, Wilcoxon p={p:.2g}",
                         fontsize=10)
            ax.set_ylabel("Number of images")
            ax.legend(fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel("Per-image IoU difference: stride 3 (E3) minus stride 14 (E4), both k<=6")
    fig.suptitle("Overlapping windows (stride 3) vs single pass (stride 14), paired per image: most images change by < 0.02 IoU")
    fig.tight_layout()
    fig.savefig(out / "stride_paired.png", dpi=110)
    plt.close(fig)


def fig_difficulty(frames, attrs, out, run="E3", feat="768-D"):
    d = frames[run, feat].join(attrs)
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.8))
    ax = axs[0]
    ax.scatter(d["area"], d["oracle_iou"], s=12, alpha=0.6, color="tab:gray", label="oracle")
    ax.scatter(d["area"], d["border_iou"], s=12, alpha=0.6, color="tab:blue", label="border rule")
    ax.set_xscale("log")
    ax.set_xlabel("Ground-truth object area (fraction of image, log scale)")
    ax.set_ylabel("Per-image IoU")
    ax.set_title(f"IoU vs object size ({run} {feat})")
    ax.legend()
    ax.grid(alpha=0.3)

    def grouped(ax, key, cats, title, xlabel):
        x = np.arange(len(cats))
        for off, mode, c in ((-0.2, "oracle", "tab:gray"), (0.2, "border", "tab:blue")):
            means = [d.loc[d[key] == g, f"{mode}_iou"].mean() for g in cats]
            ax.bar(x + off, means, 0.4, color=c, label=mode)
            for xi, v in zip(x + off, means):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ns = [(d[key] == g).sum() for g in cats]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(cats, ns)], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean IoU")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    grouped(axs[1], "size_bin", SIZE_LABELS, "Mean IoU by object size", "Ground-truth area bin")
    grouped(axs[2], "touches_border", [False, True], "Mean IoU by whether the object touches the image edge",
            "Ground-truth object touches the image edge")
    fig.suptitle("Difficulty factors computed from the USOD ground truth masks")
    fig.tight_layout()
    fig.savefig(out / "difficulty_factors.png", dpi=110)
    plt.close(fig)


def fig_selection_by_kstar(frames, out, run="E3"):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, feat in zip(axs, FEATS.values()):
        d = frames[run, feat]
        ks = sorted(d["k_star"].unique())
        x = np.arange(len(ks))
        for off, mode, c in ((-0.2, "oracle", "tab:gray"), (0.2, "border", "tab:blue")):
            means = [d.loc[d["k_star"] == k, f"{mode}_iou"].mean() for k in ks]
            ax.bar(x + off, means, 0.4, color=c, label=mode)
            for xi, v in zip(x + off, means):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"k*={k}\n(n={(d['k_star'] == k).sum()})" for k in ks])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"{run} {feat}: IoU by the k* chosen by silhouette")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("The border rule is near the oracle at k*=2 but breaks down when it has to merge 3 or more clusters")
    fig.tight_layout()
    fig.savefig(out / "selection_by_kstar.png", dpi=110)
    plt.close(fig)


def fig_galleries(frames, res, out, run="E3", feat="768-D", n=6):
    d = frames[run, feat].sort_values("border_iou")
    mid = len(d) // 2
    picks = {"worst": d.index[:n], "median": d.index[mid - n // 2: mid + n - n // 2], "best": d.index[-n:][::-1]}
    panel_dir = res / RUNS[run][0] / "panels"
    for name, ims in picks.items():
        tiles = [Image.open(panel_dir / f"{im}.png").convert("RGB") for im in ims]
        w = max(t.width for t in tiles)
        canvas = Image.new("RGB", (w, sum(t.height for t in tiles)), "white")
        y = 0
        for t in tiles:
            canvas.paste(t, (0, y))
            y += t.height
        canvas.save(out / f"gallery_{name}.png", optimize=True)


# ---------------------------------------------------------------- main

def export_json(frames, attrs, summ, sig, path):
    per_image = []
    e3r, e3d = frames["E3", "PCA-RGB"], frames["E3", "768-D"]
    for im in attrs.index:
        a = attrs.loc[im]
        per_image.append(dict(
            image=im, area=round(float(a["area"]), 4), touches_border=bool(a["touches_border"]),
            n_components=int(a["n_components"]),
            oracle_768=round(float(e3d.at[im, "oracle_iou"]), 3), border_768=round(float(e3d.at[im, "border_iou"]), 3),
            k_768=int(e3d.at[im, "k_star"]),
            oracle_rgb=round(float(e3r.at[im, "oracle_iou"]), 3), border_rgb=round(float(e3r.at[im, "border_iou"]), 3),
            k_rgb=int(e3r.at[im, "k_star"]),
            category=str(category(np.array([e3d.at[im, "oracle_iou"]]), np.array([e3d.at[im, "border_iou"]]))[0])))
    oracle_k = {f"{run} {feat}": [round(float(d[f"oracle_iou_k{k}"].mean()), 4)
                                  for k in range(2, 7) if f"oracle_iou_k{k}" in d]
                for (run, feat), d in frames.items()}
    breakdown = {}
    for (run, feat), d in frames.items():
        cat = category(d["oracle_iou"].to_numpy(), d["border_iou"].to_numpy())
        breakdown[f"{run} {feat}"] = {c: int((cat == c).sum()) for c in CATS}
    payload = dict(summary=json.loads(summ.round(4).to_json(orient="records")),
                   significance=json.loads(sig.round(5).to_json(orient="records")),
                   oracle_by_k=oracle_k, breakdown=breakdown, per_image=per_image)
    path.write_text(json.dumps(payload, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=ROOT / "results" / "usod")
    ap.add_argument("--data", default=ROOT / "data" / "USOD")
    ap.add_argument("--fig-out", default=ROOT / "docs" / "figures" / "analysis")
    ap.add_argument("--export-json", action="store_true", help="also write analysis/summary.json for dashboards")
    args = ap.parse_args()

    res, fig_out = Path(args.results), Path(args.fig_out)
    tab_out = res / "analysis"
    fig_out.mkdir(parents=True, exist_ok=True)
    tab_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    long = load_long(res)
    frames = load_wide(res)
    attrs = gt_attributes(Path(args.data) / "masks", sorted(long["image"].unique()))

    summ = summary_table(long, rng)
    sig = significance_table(long, rng)
    summ.to_csv(tab_out / "summary.csv", index=False, float_format="%.4f")
    sig.to_csv(tab_out / "significance.csv", index=False, float_format="%.5g")
    per_image = frames["E3", "768-D"][["oracle_iou", "border_iou", "k_star"]].add_prefix("e3_768_").join(
        frames["E3", "PCA-RGB"][["oracle_iou", "border_iou", "k_star"]].add_prefix("e3_rgb_")).join(attrs)
    per_image["e3_768_category"] = category(per_image["e3_768_oracle_iou"].to_numpy(), per_image["e3_768_border_iou"].to_numpy())
    per_image.to_csv(tab_out / "per_image.csv", float_format="%.4f")

    fig_method_comparison(summ, fig_out)
    fig_iou_distribution(long, fig_out)
    fig_oracle_vs_border(frames, fig_out)
    fig_failure_breakdown(frames, fig_out)
    fig_oracle_vs_k(frames, fig_out)
    fig_stride_paired(frames, fig_out)
    fig_difficulty(frames, attrs, fig_out)
    fig_selection_by_kstar(frames, fig_out)
    fig_galleries(frames, res, fig_out)
    if args.export_json:
        export_json(frames, attrs, summ, sig, tab_out / "summary.json")

    cols = ["method", "iou_mean", "iou_ci_lo", "iou_ci_hi", "iou_median", "f_mean", "mae_mean", "frac_iou_gt_0_5"]
    print(summ[cols].round(3).to_string(index=False))
    print()
    print(sig[["a", "b", "mean_diff", "diff_ci_lo", "diff_ci_hi", "wilcoxon_p", "n_a_better", "n_b_better"]]
          .round(4).to_string(index=False))
    print()
    d = per_image
    print("E3 768-D categories:", d["e3_768_category"].value_counts().to_dict())
    print("mean border IoU by size bin:", d.groupby("size_bin", observed=False)["e3_768_border_iou"].mean().round(3).to_dict())
    print("mean border IoU by touches_border:", d.groupby("touches_border")["e3_768_border_iou"].mean().round(3).to_dict())
    print("mean border IoU by k*:", d.groupby("e3_768_k_star")["e3_768_border_iou"].mean().round(3).to_dict())
    print(f"figures -> {fig_out}, tables -> {tab_out}")


if __name__ == "__main__":
    main()
