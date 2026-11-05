"""Cross-dataset results: tables + per-image side-by-side galleries against every earlier method.

  X1: trained on USOD only -> tested on SUIM test (GT 'not-water', and SUIM Saliency GT)
  X2: trained on SUIM only -> tested on all 300 USOD images
Outputs in results/cross_dataset/.
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from baseline_masks import get_masks

XR = ROOT / "results" / "cross_dataset"
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
evpos = {i: k for k, i in enumerate(ev)}
IND = {s: np.load(pred_path(f"stage_{s}", "evalset")).astype(np.float32) for s in "ABC"}
IND_THR = {s: json.load(open(RES / f"stage_{s}" / "config.json"))["thr"] for s in "ABC"}
R = ROOT / "results"


def saliency_gt(r):
    return load_gt(r["data"], r["stem"], "suim_saliency")


def base_tab(path, feat="768", **flt):
    d = pd.read_csv(path); d["image"] = d["image"].str.replace(".jpg", "", regex=False)
    d = d[d.feat.astype(str) == feat] if "feat" in d else d
    for k, v in flt.items(): d = d[d[k] == v]
    return d.set_index("image")

def dendro(path, rule="gap_border"):
    return base_tab(path, spatial="free", linkage="average", rule=rule)

BASE = {  # setting -> {method: Series(image -> iou)}
    "usod": {"k-means (border rule)": base_tab(R / "kmeans/usod/cluster_union_stride3_k6/metrics_union.csv")["border_iou"],
             "dendrogram (gap rule)": dendro(R / "dendrogram/usod/metrics_s3.csv")["iou"],
             "k-means oracle (uses GT)": base_tab(R / "kmeans/usod/cluster_union_stride3_k6/metrics_union.csv")["oracle_iou"]},
    "suim_nonwater": {"k-means (border rule)": base_tab(R / "kmeans/suim/cluster_union_stride3_k6/metrics_union.csv")["border_iou"],
                      "dendrogram (gap rule)": dendro(R / "dendrogram/suim_nonwater/metrics_s3.csv")["iou"],
                      "k-means oracle (uses GT)": base_tab(R / "kmeans/suim/cluster_union_stride3_k6/metrics_union.csv")["oracle_iou"]},
    "suim_saliency": {"k-means (border rule)": base_tab(R / "kmeans/suim_saliency/cluster_union_stride3_k6/metrics_union.csv")["border_iou"],
                      "dendrogram (gap rule)": dendro(R / "dendrogram/suim_saliency/metrics_s3.csv")["iou"],
                      "k-means oracle (uses GT)": base_tab(R / "kmeans/suim_saliency/cluster_union_stride3_k6/metrics_union.csv")["oracle_iou"]},
}

SETTINGS = {  # name: (tag, source ds, target dataset, GT mode, baseline key, title)
    "X1_nonwater": ("x1", "usod", "suim", "suim_nonwater", "suim_nonwater", "X1: train USOD -> test SUIM (GT: not-water)"),
    "X1_saliency": ("x1", "usod", "suim", "suim_saliency", "suim_saliency", "X1: train USOD -> test SUIM (GT: SUIM Saliency)"),
    "X2_usod": ("x2", "suim", "usod", "usod", "usod", "X2: train SUIM -> test USOD (all 300)"),
}
rows_all, tables, per_image = [], [], {}
for name, (tag, src, tgt, gtm, bkey, title) in SETTINGS.items():
    idx = np.load(OUT / "preds" / f"cross_{tag}_index.npy")
    P = {s: np.load(OUT / "preds" / f"cross_{tag}_{s}.npy").astype(np.float32) for s in "ABC"}
    thr = {s: json.load(open(XR / tag / f"stage_{s}" / "config.json"))["thr"][src] for s in "ABC"}
    rec = []
    for k, i in enumerate(idx):
        r = recs[i]
        gt = load_gt(r["data"], r["stem"], gtm)
        if gtm == "suim_saliency" and not gt.any(): continue
        row = dict(image=r["stem"], split=r["split"], area=float(gt.mean()))
        for s in "ABC":
            row[f"cross {s}"] = mask_metrics(prob_to_mask(P[s][k], gt.shape, thr[s]), gt)["iou"]
        if gtm != "suim_saliency":      # in-domain reference (trained on the target dataset) exists for these GTs
            for s in "ABC":
                if r["split"] in ("val", "test") and r["split"] == "test":
                    ch = CH[r["ds"]]
                    row[f"in-domain {s}"] = mask_metrics(prob_to_mask(IND[s][evpos[i]][ch], gt.shape, IND_THR[s][r["ds"]]), gt)["iou"]
        for m, ser in BASE[bkey].items():
            row[m] = ser.get(r["stem"], np.nan)
        rec.append(row)
    df = pd.DataFrame(rec)
    df["size"] = pd.cut(df.area, [0, 0.1, 0.3, 1.01], labels=["small (<10%)", "medium (10-30%)", "large (>30%)"])
    # images the baselines could be scored on
    df["has_base"] = df["dendrogram (gap rule)"].notna()
    per_image[name] = df
    df.to_csv(XR / f"per_image_{name}.csv", index=False)

order = ["k-means (border rule)", "dendrogram (gap rule)", "k-means oracle (uses GT)", "in-domain A", "in-domain B", "in-domain C", "cross A", "cross B", "cross C"]
md = ["# Cross-dataset generalisation\n",
      "Source-only training: every choice (hyper-parameters, threshold, checkpoint) used the source dataset's validation split only. Baselines are training-free. All methods scored on the same images, native GT resolution. *in-domain* = Section 21 models (trained on the target dataset's own training split) on the target's test split; only available where that split is held out.\n"]
summary = []
for name, (tag, src, tgt, gtm, bkey, title) in SETTINGS.items():
    df = per_image[name]; sub = df[df.has_base]
    md += [f"## {title}\n", f"n = {len(sub)} images with baseline results ({len(df)} scored for cross models)\n", "| Method | n | mean IoU [95% CI] | IoU>0.5 | IoU<0.2 |", "|---|---|---|---|---|"]
    for m in order:
        if m not in sub or sub[m].notna().sum() == 0: continue
        v = sub[m].dropna(); lo, hi = bootstrap_ci(v.values)
        md.append(f"| {m} | {len(v)} | **{v.mean():.3f}** [{lo:.3f}, {hi:.3f}] | {(v > .5).mean():.0%} | {(v < .2).mean():.0%} |")
        summary.append(dict(setting=name, method=m, n=len(v), iou=v.mean(), lo=lo, hi=hi))
    # by size
    md += ["", "Mean IoU by object size (cross C vs best label-free baseline):\n", "| size | n | k-means border | dendrogram gap | cross A | cross B | cross C |", "|---|---|---|---|---|---|---|"]
    for b, g in sub.groupby("size", observed=True):
        md.append(f"| {b} | {len(g)} | " + " | ".join(f"{g[m].mean():.3f}" for m in ["k-means (border rule)", "dendrogram (gap rule)", "cross A", "cross B", "cross C"]) + " |")
    d = (sub["cross C"] - sub["dendrogram (gap rule)"]).values; lo, hi = bootstrap_ci(d)
    md += ["", f"Paired gain cross C minus dendrogram: **{d.mean():+.3f}** [{lo:+.3f}, {hi:+.3f}], cross C better on {(d > 0).mean():.0%} of images.\n"]
    if name == "X2_usod":
        t = sub[sub.split == "test"]
        md += ["USOD held-out test subset (100 images, same as Section 21):\n", "| Method | mean IoU |", "|---|---|"]
        for m in order:
            if m in t and t[m].notna().any(): md.append(f"| {m} | {t[m].mean():.3f} |")
        md.append("")
pd.DataFrame(summary).to_csv(XR / "summary.csv", index=False)

# ------------------------------------------------------------------ figures
INK, GREY = "#1f2937", "#9ca3af"
COL = {"k-means (border rule)": GREY, "dendrogram (gap rule)": "#6b7280", "k-means oracle (uses GT)": "#d97706",
       "in-domain A": "#99f6e4", "in-domain B": "#2dd4bf", "in-domain C": "#0f766e",
       "cross A": "#bfdbfe", "cross B": "#60a5fa", "cross C": "#1d4ed8"}
S = pd.DataFrame(summary)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharex=True)
for ax, (name, v) in zip(axs, SETTINGS.items()):
    g = S[S.setting == name].iloc[::-1]
    ax.barh(range(len(g)), g.iou, color=[COL[m] for m in g.method], height=0.62, xerr=[g.iou - g.lo, g.hi - g.iou], error_kw=dict(ecolor=INK, lw=1, capsize=2))
    for y, val in enumerate(g.iou): ax.text(min(val + 0.015, 0.93), y, f"{val:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.method, fontsize=8); ax.set_xlim(0, 1)
    ax.set_title(v[5], fontsize=9, loc="left"); ax.set_xlabel("mean IoU (95% CI)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(XR / "summary_bars.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (name, v) in zip(axs, SETTINGS.items()):
    d = per_image[name]; d = d[d.has_base]
    ax.scatter(d["dendrogram (gap rule)"], d["cross C"], s=14, color="#1d4ed8", alpha=.7, edgecolor="white", lw=.4)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("dendrogram (gap rule) IoU"); ax.set_ylabel("cross C IoU"); ax.set_title(v[5], fontsize=9, loc="left")
    ax.text(.03, .94, f"cross C better on {(d['cross C'] > d['dendrogram (gap rule)']).mean():.0%}", fontsize=9, color=INK, va="top")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(XR / "scatter_vs_dendrogram.png", dpi=150); plt.close()

# ------------------------------------------------------------------ galleries
rng = np.random.RandomState(0)
def choose(df, n_extra=12):
    d = df[df.has_base].copy()
    if "in-domain C" in d: d = d[d["in-domain C"].notna()]     # only images held out from in-domain training, so the columns are comparable
    picks, why = [], {}
    def add(stem, reason):
        if stem not in picks and len(picks) < n_extra: picks.append(stem); why[stem] = reason
    for b, g in d.groupby("size", observed=True):                  # median-IoU image of each size bucket
        if len(g): add(g.iloc[(g["cross C"] - g["cross C"].median()).abs().argsort().iloc[0]].image, f"{b}, typical")
    for s in d.sort_values("cross C").image.iloc[-3:][::-1]: add(s, "best cross C")
    for s in d.sort_values("cross C").image.iloc[:3]: add(s, "worst cross C")
    best_base = d[["k-means (border rule)", "dendrogram (gap rule)"]].max(1)
    for s in d.assign(g=best_base - d["cross C"]).sort_values("g").image.iloc[-3:][::-1]: add(s, "baseline beats cross C")
    return picks, why

def draw(name, picks, why, df, per_fig=6):
    tag, src, tgt, gtm, bkey, title = SETTINGS[name]
    items = []
    for st in picks:
        r = next(x for x in recs if x["stem"] == st and x["ds"] == tgt and (tgt == "usod" or x["split"] == "test"))
        items.append(dict(r, gt_mode=gtm))
    masks = get_masks(items)
    idx = list(np.load(OUT / "preds" / f"cross_{tag}_index.npy"))
    P = {s: np.load(OUT / "preds" / f"cross_{tag}_{s}.npy").astype(np.float32) for s in "ABC"}
    thr = {s: json.load(open(XR / tag / f"stage_{s}" / "config.json"))["thr"][src] for s in "ABC"}
    has_ind = "in-domain C" in df
    cols = ["image", "ground truth", "k-means\\n(border rule)", "dendrogram\\n(gap rule)", "k-means\\noracle (uses GT)"] + (["in-domain C\\n(trained on target)"] if has_ind else []) + ["cross A\\n(linear probe)", "cross B\\n(conv head)", "cross C\\n(+LoRA)"]
    files = []
    for part, lo in enumerate(range(0, len(items), per_fig), 1):
        chunk = items[lo:lo + per_fig]
        fig, axs = plt.subplots(len(chunk), len(cols), figsize=(2.15 * len(cols), 1.75 * len(chunk) + 0.5), squeeze=False)
        for a, it in enumerate(chunk):
            gt = load_gt(it["data"], it["stem"], gtm); k = idx.index(recs.index(next(x for x in recs if x["stem"] == it["stem"] and x["ds"] == tgt and (tgt == "usod" or x["split"] == "test"))))
            ri = df[df.image == it["stem"]].iloc[0]
            panels = [(np.asarray(Image.open(it["img"]).convert("RGB").resize(gt.shape[::-1])), None), (gt, None)]
            for m in ["k-means (border rule)", "dendrogram (gap rule)", "k-means oracle (uses GT)"]:
                mk, iou = masks[it["stem"]][m]; panels.append((mk, iou))
            if has_ind:
                i_rec = recs.index(next(x for x in recs if x["stem"] == it["stem"] and x["ds"] == tgt and (tgt == "usod" or x["split"] == "test")))
                panels.append((prob_to_mask(IND["C"][evpos[i_rec]][CH[tgt]], gt.shape, IND_THR["C"][tgt]), ri["in-domain C"]))
            for s in "ABC": panels.append((prob_to_mask(P[s][k], gt.shape, thr[s]), ri[f"cross {s}"]))
            for b, (im, iou) in enumerate(panels):
                ax = axs[a, b]; ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_visible(False)
                if iou is not None:
                    ax.text(.03, .04, f"IoU {iou:.2f}", transform=ax.transAxes, color="white", fontsize=8, va="bottom", bbox=dict(fc="black", alpha=.6, lw=0, pad=1.5))
                if a == 0: ax.set_title(cols[b].replace("\\n", "\n"), fontsize=8)
            axs[a, 0].set_ylabel(f"{it['stem']}\n{why[it['stem']]}", fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
        plt.suptitle(title, fontsize=10, x=0.01, ha="left"); plt.tight_layout()
        f = XR / f"gallery_{name}_{part}.png"; plt.savefig(f, dpi=120); plt.close(); files.append(f.name)
    return files

md += ["## Galleries (same image, every method)\n", "Selection rule (not cherry-picked): the median-IoU image of each object-size bucket, the 3 best and 3 worst cross-C images, and the 3 images where the best label-free baseline beats cross C by most; capped at 12 per setting, images held out from in-domain training only. IoU is printed on each mask.\n"]
for name in SETTINGS:
    picks, why = choose(per_image[name])
    files = draw(name, picks, why, per_image[name])
    md += [f"* **{SETTINGS[name][5]}**: " + ", ".join(f"`{f}`" for f in files)]
md += ["", "Other figures: `summary_bars.png` (all methods, 3 settings), `scatter_vs_dendrogram.png` (per-image cross C vs dendrogram).\n"]
open(XR / "report.md", "w").write("\n".join(md))
print("\n".join(md[:80]))
