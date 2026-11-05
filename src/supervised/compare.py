"""Final comparison: baselines (k-means, dendrogram, optional SVAM-Net) vs Stages A-D on the SAME held-out images.

  python compare.py
Metrics at native GT resolution. Test split only: 100 held-out USOD images, 110 SUIM test images.
Writes results/comparison/{comparison.csv,comparison.md,per_image.csv, *.png} and results/supervised/stage_X/metrics.json.
"""
import numpy as np, pandas as pd, torch, torch.nn.functional as F, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

CMP = ROOT / "results" / "comparison"; CMP.mkdir(parents=True, exist_ok=True)
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
STAGES = {"A": "A: linear probe (frozen)", "B": "B: conv head (frozen)", "C": "C: + LoRA fine-tune", "D": "D: + SAM refinement"}
preds = {s: np.load(pred_path(f"stage_{s}", "evalset")).astype(np.float32) for s in STAGES if pred_path(f"stage_{s}", "evalset").exists()}
thr = {s: json.load(open(RES / f"stage_{s}" / "config.json")).get("thr", {"usod": 0.5, "suim": 0.5}) for s in preds}
thr["D"] = {"usod": 0.5, "suim": 0.5}

rows = []
for k, i in enumerate(ev):
    r = recs[i]
    if r["split"] != "test": continue
    gt = load_gt(r["data"], r["stem"], r["gt_mode"])
    ch = CH[r["ds"]]
    row = dict(ds=r["ds"], image=r["stem"], area=float(gt.mean()))
    for s, P in preds.items():
        pr = prob_to_mask(P[k, ch], gt.shape, thr[s][r["ds"]])
        m = mask_metrics(pr, gt)
        for kk in ("iou", "fbeta", "mae"): row[f"{s}_{kk}"] = m[kk]
    rows.append(row)
df = pd.DataFrame(rows)

# ---------------- baselines (computed earlier, no training; evaluated per image on the same stems)
def base_kmeans(path):
    d = pd.read_csv(path); d = d[d.feat.astype(str) == "768"].copy(); d["image"] = d["image"].str.replace(".jpg", "", regex=False)
    return d.set_index("image")

def base_dendro(path, linkage="average", rule="gap_border"):
    d = pd.read_csv(path); d["image"] = d["image"].str.replace(".jpg", "", regex=False)
    d = d[(d.feat.astype(str) == "768") & (d.spatial == "free") & (d.linkage == linkage) & (d.rule == rule)]
    return d.set_index("image")

base = {}
R = ROOT / "results"
km_u = base_kmeans(R / "kmeans/usod/cluster_union_stride3_k6/metrics_union.csv")
km_s = base_kmeans(R / "kmeans/suim/cluster_union_stride3_k6/metrics_union.csv")
dn_u = base_dendro(R / "dendrogram/usod/metrics_s3.csv"); dn_s = base_dendro(R / "dendrogram/suim_nonwater/metrics_s3.csv")
dn_u2 = base_dendro(R / "dendrogram/usod/metrics_s3.csv", "average", "sil_border"); dn_s2 = base_dendro(R / "dendrogram/suim_nonwater/metrics_s3.csv", "average", "sil_border")
BASE = {
    "k-means, border rule (stride 3, k<=6)": (km_u, km_s, "border_iou", "border_f", "border_mae"),
    "dendrogram, silhouette+border": (dn_u2, dn_s2, "iou", "f", "mae"),
    "dendrogram, gap rule (best label-free)": (dn_u, dn_s, "iou", "f", "mae"),
    "k-means ORACLE (uses GT)": (km_u, km_s, "oracle_iou", "oracle_f", "oracle_mae"),
}
for name, (u, s, ci, cf, cm) in BASE.items():
    for ds, tab in (("usod", u), ("suim", s)):
        sel = df.ds == ds
        for met, col in (("iou", ci), ("fbeta", cf), ("mae", cm)):
            df.loc[sel, f"{name}__{met}"] = df.loc[sel, "image"].map(tab[col]).values

# optional SVAM-Net (USOD only), produced by svam_baseline.py
sv = OUT / "svam_usod.csv"
if sv.exists():
    d = pd.read_csv(sv).set_index("image")
    for met in ("iou", "fbeta", "mae"):
        df.loc[df.ds == "usod", f"SVAM-Net (pretrained, USOD)__{met}"] = df.loc[df.ds == "usod", "image"].map(d[met]).values

methods = [c[:-5] for c in df.columns if c.endswith("__iou")] + [f"{s}_" for s in preds]
order = list(BASE) + ([m for m in methods if m.startswith("SVAM")]) + [s for s in preds]
def col(m, met): return f"{m}__{met}" if "__iou" in "".join(df.columns) and f"{m}__{met}" in df.columns else f"{m}_{met}"
name_of = lambda m: STAGES[m] if m in STAGES else m

# restrict SUIM to images the baselines evaluated (non-degenerate GT)
valid_s = set(km_s.index) & set(dn_s.index)
df["in_base"] = (df.ds == "usod") | df.image.isin(valid_s)
out = []
for ds, label in (("usod", "USOD held-out (100 imgs, GT = salient object)"), ("suim", "SUIM test (GT = not-water)")):
    sub = df[(df.ds == ds) & df.in_base]
    for m in order:
        c = col(m, "iou")
        if c not in sub or sub[c].isna().all(): continue
        v = sub[c].dropna(); lo, hi = bootstrap_ci(v.values)
        out.append(dict(dataset=label, method=name_of(m), n=len(v), iou=v.mean(), iou_lo=lo, iou_hi=hi,
                        fbeta=sub[col(m, "fbeta")].mean(), mae=sub[col(m, "mae")].mean(),
                        frac_iou_gt_0_5=(v > 0.5).mean(), frac_iou_lt_0_2=(v < 0.2).mean()))
tab = pd.DataFrame(out)
tab.to_csv(CMP / "comparison.csv", index=False)
df.to_csv(CMP / "per_image.csv", index=False)
# stage-all-110 extra rows
extra = []
for s in preds:
    sub = df[df.ds == "suim"]; v = sub[f"{s}_iou"]
    extra.append(dict(method=name_of(s), n=len(v), iou=v.mean()))
pd.DataFrame(extra).to_csv(CMP / "suim_all110_stages.csv", index=False)

# per-stage metrics.json
for s in preds:
    json.dump({ds: {met: float(df[df.ds == ds][f"{s}_{met}"].mean()) for met in ("iou", "fbeta", "mae")} | {"n": int((df.ds == ds).sum())}
               for ds in ("usod", "suim")}, open(RES / f"stage_{s}" / "metrics.json", "w"), indent=1)

# ---------------- markdown
def md(t):
    lines = []
    for ds, g in t.groupby("dataset", sort=False):
        lines += [f"### {ds}", "", "| Method | n | mean IoU [95% CI] | F-beta | MAE | IoU>0.5 | IoU<0.2 |", "|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.method} | {r.n} | **{r.iou:.3f}** [{r.iou_lo:.3f}, {r.iou_hi:.3f}] | {r.fbeta:.3f} | {r.mae:.3f} | {r.frac_iou_gt_0_5:.0%} | {r.frac_iou_lt_0_2:.0%} |")
        lines.append("")
    return "\n".join(lines)

# size buckets
df["size"] = pd.cut(df.area, [0, 0.1, 0.3, 1.01], labels=["small (<10%)", "medium (10-30%)", "large (>30%)"])
sz = []
for ds in ("usod", "suim"):
    sub = df[(df.ds == ds) & df.in_base]
    for b, g in sub.groupby("size", observed=True):
        for m in order:
            c = col(m, "iou")
            if c in g and g[c].notna().any(): sz.append(dict(ds=ds, size=b, method=name_of(m), n=int(g[c].notna().sum()), iou=g[c].mean()))
sz = pd.DataFrame(sz); sz.to_csv(CMP / "iou_by_object_size.csv", index=False)

# paired stage gains
pg = []
chain = [s for s in "ABCD" if s in preds]
ref = "dendrogram, gap rule (best label-free)"
for ds in ("usod", "suim"):
    sub = df[(df.ds == ds) & df.in_base]
    prev_name, prev = ref, sub[f"{ref}__iou"]
    for s in chain:
        cur = sub[f"{s}_iou"]; d = (cur - prev).dropna().values
        lo, hi = bootstrap_ci(d)
        pg.append(dict(ds=ds, step=f"{prev_name} -> {name_of(s)}", mean_gain=d.mean(), ci_lo=lo, ci_hi=hi, frac_images_improved=(d > 0).mean()))
        prev_name, prev = name_of(s), cur
pg = pd.DataFrame(pg); pg.to_csv(CMP / "stage_gains_paired.csv", index=False)
with open(CMP / "comparison.md", "w") as f:
    f.write("# Comparison on held-out test images\n\nAll methods scored on the same images at native GT resolution. Supervised stages never saw these images; thresholds and all hyper-parameters were chosen on a separate validation split.\n\n")
    f.write(md(tab)); f.write("\n### Mean IoU by object size\n\n" + sz.pivot_table(index=["ds", "method"], columns="size", values="iou", observed=True, sort=False).round(3).to_markdown() + "\n")
    f.write("\n### Paired IoU gain, step by step (bootstrap 95% CI over images)\n\n" + pg.round(3).to_markdown(index=False) + "\n")
    f.write("\n### SUIM, Stages on all 110 test images (baselines excluded degenerate GT)\n\n" + pd.DataFrame(extra).round(3).to_markdown(index=False) + "\n")

# ---------------- figures
INK, GREY = "#1f2937", "#9ca3af"
blues = ["#bfdbfe", "#7fb2f0", "#3b82f6", "#1d4ed8"]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharex=False)
for ax, (ds, g) in zip(axs, tab.groupby("dataset", sort=False)):
    g = g.iloc[::-1]
    colors = [blues[chain.index(s)] if (s := next((k for k, v in STAGES.items() if v == m), None)) else ("#d97706" if "ORACLE" in m else GREY) for m in g.method]
    ax.barh(range(len(g)), g.iou, color=colors, height=0.62, xerr=[g.iou - g.iou_lo, g.iou_hi - g.iou], error_kw=dict(ecolor=INK, lw=1, capsize=2))
    for y, (v) in enumerate(g.iou): ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.method, fontsize=8.5); ax.set_xlim(0, 1.0)
    ax.set_title(ds, fontsize=10, loc="left"); ax.set_xlabel("mean IoU (95% bootstrap CI)")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(CMP / "method_comparison.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, ds in zip(axs, ("usod", "suim")):
    g = sz[sz.ds == ds]; sizes = ["small (<10%)", "medium (10-30%)", "large (>30%)"]
    ms = [m for m in dict.fromkeys(g.method)]
    w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        s = next((k for k, v in STAGES.items() if v == m), None)
        c = blues[chain.index(s)] if s else ("#d97706" if "ORACLE" in m else GREY if "k-means" in m else "#6b7280")
        vals = [g[(g.method == m) & (g["size"] == b)].iou.mean() for b in sizes]
        ax.bar(np.arange(3) + j * w, vals, w * 0.9, color=c, label=m)
    ax.set_xticks(np.arange(3) + 0.4 - w / 2); ax.set_xticklabels([f"{b}\n(n={int(g[(g['size']==b)].n.max())})" for b in sizes], fontsize=9)
    ax.set_title(f"{ds.upper()}: IoU by object size", loc="left", fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axs[0].set_ylabel("mean IoU"); axs[1].legend(fontsize=6.5, loc="lower right", frameon=False)
plt.tight_layout(); plt.savefig(CMP / "iou_by_object_size.png", dpi=150); plt.close()

# qualitative gallery: 4 USOD + 2 SUIM test images (spread of baseline-failure cases)
last = chain[-1]
pick = []
for ds, n in (("usod", 5), ("suim", 3)):
    sub = df[(df.ds == ds) & df.in_base].sort_values(f"{ref}__iou")
    pick += list(sub.image.iloc[np.linspace(0, len(sub) - 1, n).astype(int)])
cols = ["image", "GT"] + [f"stage {s}" for s in chain]
fig, axs = plt.subplots(len(pick), len(cols), figsize=(2.3 * len(cols), 1.8 * len(pick)))
evmap = {recs[i]["stem"] + recs[i]["ds"]: k for k, i in enumerate(ev)}
for a, stem in enumerate(pick):
    r = next(x for x in recs if x["stem"] == stem and x["split"] == "test")
    img = np.asarray(load_image(r["img"], (308, 231))); gt = load_gt(r["data"], r["stem"], r["gt_mode"])
    ims = [img, np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize((308, 231)))]
    k = evmap[stem + r["ds"]]; ch = CH[r["ds"]]
    for s in chain: ims.append(preds[s][k, ch] > thr[s][r["ds"]])
    for b, im in enumerate(ims):
        ax = axs[a, b]; ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.axis("off")
        if a == 0: ax.set_title(cols[b], fontsize=8)
        if b >= 2: ax.text(4, 222, f"IoU {df.loc[(df.image==stem), f'{chain[b-2]}_iou'].iloc[0]:.2f}", color="yellow", fontsize=7, va="bottom", bbox=dict(fc="black", alpha=.5, lw=0, pad=1))
plt.tight_layout(); plt.savefig(CMP / "qualitative.png", dpi=130); plt.close()
print(md(tab)); print(sz.pivot_table(index=["ds", "method"], columns="size", values="iou", observed=True, sort=False).round(3)); print(pg.round(3).to_string())
