"""Tables, figures and per-image collages of the GMM study (Sec. 27): in-domain, X1 (USOD source -> SUIM test), X2 (SUIM source -> all USOD).

  python compare_gmm.py     # needs per_image_usod.csv / per_image_suim.csv (gmm_eval.py) and config_*.json / global_*.pkl
Baselines (k-means border rule and oracle, dendrogram gap rule) are the stored per-image numbers of the earlier studies. Outputs in results/gmm/.
"""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gmm_common import *
from baseline_masks import get_masks
from common import bootstrap_ci

R = ROOT / "results"
sets, _ = image_sets()
CFG = {s: json.load(open(RES / f"config_{s}.json")) for s in ("usod", "suim")}


def base_tab(path, flt=None):
    d = pd.read_csv(path); d["image"] = d["image"].str.replace(".jpg", "", regex=False)
    if "feat" in d: d = d[d.feat.astype(str) == "768"]
    for k, v in (flt or {}).items(): d = d[d[k] == v]
    return d.set_index("image")


def baselines(gt):
    kd = {"usod": "usod", "suim_nonwater": "suim", "suim_saliency": "suim_saliency"}[gt]; dd = {"usod": "usod", "suim_nonwater": "suim_nonwater", "suim_saliency": "suim_saliency"}[gt]
    km = base_tab(R / "kmeans" / kd / "cluster_union_stride3_k6" / "metrics_union.csv")
    dg = base_tab(R / "dendrogram" / dd / "metrics_s3.csv", dict(spatial="free", linkage="average", rule="gap_border"))
    return {"k-means (border rule)": km["border_iou"], "k-means oracle (uses GT)": km["oracle_iou"], "dendrogram (gap rule)": dg["iou"]}


def gt_area(data, stem, gt):
    return float(load_gt(data, stem, gt).mean())


per = {"usod": pd.read_csv(RES / "per_image_usod.csv", dtype={"image": str}), "suim": pd.read_csv(RES / "per_image_suim.csv", dtype={"image": str})}
test_usod = set(sets["usod_test"][1])
SETTINGS = [  # name, image table, GT, which config ('own'/'cross'), restrict-to
    ("In-domain USOD (test 100)", "usod", "usod", "own", "test100"),
    ("X2: SUIM settings -> all 300 USOD", "usod", "usod", "cross", "all"),
    ("X2: SUIM settings -> USOD test 100", "usod", "usod", "cross", "test100"),
    ("In-domain SUIM, not-water (test)", "suim", "suim_nonwater", "own", "all"),
    ("In-domain SUIM, saliency (test)", "suim", "suim_saliency", "own", "all"),
    ("X1: USOD settings -> SUIM test, not-water", "suim", "suim_nonwater", "cross", "all"),
    ("X1: USOD settings -> SUIM test, saliency", "suim", "suim_saliency", "cross", "all"),
]
md = ["# Gaussian mixture models: results (Sec. 27)\n", "Per-image GMM = DINOv2 patches (stride 3) -> PCA -> GMM -> border rule; dataset-level GMM = one GMM fitted on the source dataset's training patches, components labelled by the border rule. "
      "Settings and fits use only the source dataset's training images; the test images are scored once. k-means and dendrogram are the stored per-image numbers of the earlier studies (same images). Label definitions differ between USOD (salient object) and SUIM (not-water / saliency), see Sec. 22.\n",
      f"Chosen per-image settings: USOD source {CFG['usod']}; SUIM source {CFG['suim']}.\n"]
rows_all, tables = [], {}
for name, key, gt, which, restrict in SETTINGS:
    df = per[key].copy(); B = baselines(gt)
    df["iou_pi"], df["iou_gl"] = df[f"iou_pi_{which}_{gt}"], df[f"iou_gl_{which}_{gt}"]
    if which == "own": df["iou_oracle"] = df[f"oracle_{gt}"]
    if restrict == "test100": df = df[df.image.isin(test_usod)]
    for b, ser in B.items(): df[b] = df.image.map(ser)
    df = df[df["dendrogram (gap rule)"].notna()]
    if gt == "suim_saliency": df = df[[load_gt(SUIM_TE, s, gt).any() for s in df.image]]
    data = USOD if key == "usod" else SUIM_TE; df["area"] = [gt_area(data, s, gt) for s in df.image]
    tables[name] = df
    cols = [("k-means (border rule)", "k-means (border rule)"), ("dendrogram (gap rule)", "dendrogram (gap rule)"), ("iou_pi", "GMM per-image"), ("iou_gl", "GMM dataset-level"), ("k-means oracle (uses GT)", "k-means oracle (uses GT)")]
    if which == "own": cols.append(("iou_oracle", "GMM oracle (uses GT)"))
    md += [f"## {name}\n", f"n = {len(df)} images\n", "| method | mean IoU [95% CI] | IoU>0.5 | IoU<0.2 |", "|---|---|---|---|"]
    for c, lab in cols:
        v = df[c].dropna(); lo, hi = bootstrap_ci(v.values); md.append(f"| {lab} | **{v.mean():.3f}** [{lo:.3f}, {hi:.3f}] | {(v > .5).mean():.0%} | {(v < .2).mean():.0%} |")
        rows_all.append(dict(setting=name, method=lab, n=len(v), iou=v.mean(), lo=lo, hi=hi))
    for c, lab in (("iou_pi", "GMM per-image"), ("iou_gl", "GMM dataset-level")):
        for b in ("k-means (border rule)", "dendrogram (gap rule)"):
            d = (df[c] - df[b]).values; lo, hi = bootstrap_ci(d); md.append(f"* {lab} minus {b}: **{d.mean():+.3f}** [{lo:+.3f}, {hi:+.3f}], better on {(d > 0).mean():.0%} of images")
    df["size"] = pd.cut(df.area, [0, 0.1, 0.3, 1.01], labels=["small (<10%)", "medium (10-30%)", "large (>30%)"])
    md += ["", "| size | n | k-means | dendrogram | GMM per-image | GMM dataset-level |", "|---|---|---|---|---|---|"]
    for b_, g in df.groupby("size", observed=True): md.append(f"| {b_} | {len(g)} | {g['k-means (border rule)'].mean():.3f} | {g['dendrogram (gap rule)'].mean():.3f} | {g['iou_pi'].mean():.3f} | {g['iou_gl'].mean():.3f} |")
    md.append("")
pd.DataFrame(rows_all).to_csv(RES / "summary.csv", index=False)
# K distribution and run time
tm = ["## Chosen number of components and run time\n", "| dataset | settings from | mean K | per-image GMM fit (s) | dataset-level GMM assignment (s) |", "|---|---|---|---|---|"]
for key in ("usod", "suim"):
    for which in ("own", "cross"):
        d = per[key]; tm.append(f"| {key} test images | {which} | {d[f'K_{which}'].mean():.2f} | {d[f'secs_pi_{which}'].mean():.2f} | {d[f'secs_gl_{which}'].mean():.3f} |")
md += tm + [""]
# figure: mean IoU per setting
S = pd.DataFrame(rows_all); COL = {"k-means (border rule)": "#9ca3af", "dendrogram (gap rule)": "#6b7280", "GMM per-image": "#7c3aed", "GMM dataset-level": "#c084fc", "k-means oracle (uses GT)": "#d97706", "GMM oracle (uses GT)": "#fbbf24"}
fig, axs = plt.subplots(1, len(SETTINGS), figsize=(3.1 * len(SETTINGS), 4.6), sharex=True)
for ax, (name, *_r) in zip(axs, SETTINGS):
    g = S[S.setting == name].iloc[::-1]
    ax.barh(range(len(g)), g.iou, color=[COL[m] for m in g.method], height=.62, xerr=[g.iou - g.lo, g.hi - g.iou], error_kw=dict(ecolor="#1f2937", lw=1, capsize=2))
    for y, v in enumerate(g.iou): ax.text(min(v + .015, .93), y, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.method if ax is axs[0] else [], fontsize=7); ax.set_xlim(0, 1); ax.set_title(name, fontsize=7, loc="left", wrap=True)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
plt.tight_layout(); plt.savefig(RES / "summary_bars.png", dpi=140); plt.close()
# per-image collages: one PNG per image
model = load_model("dinov2_vitb14", DEV); masks = {k: pickle.load(open(OUT / f"eval_{k}.pkl", "rb")) for k in ("usod", "suim")}
unp = lambda t: np.unpackbits(t[0])[: int(np.prod(t[1]))].reshape(t[1]).astype(bool)
recs = build_records(); rec_of = {(r["ds"], r["stem"], r["split"] == "test" or r["ds"] == "usod"): r for r in recs}
nitems = 0
for name, key, gt, which, restrict in SETTINGS:
    if "test 100" in name and "X2" in name: continue            # same images as the all-300 X2 table
    df = tables[name]; df = df.sort_values("iou_pi"); out = RES / "examples" / name.split(":")[0].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "") / gt
    best_base = df[["k-means (border rule)", "dendrogram (gap rule)"]].max(1)
    sel = {"good": list(df.image[::-1][:10]), "bad": list(df.image[:10]), "gmm_beats_baselines": list(df.assign(g=df.iou_pi - best_base).sort_values("g", ascending=False).image[:5])}
    data = USOD if key == "usod" else SUIM_TE; split_rec = lambda s: next(r for r in recs if r["ds"] == key and r["stem"] == s and (key == "usod" or r["split"] == "test"))
    items = {s: dict(split_rec(s), gt_mode=gt) for cat in sel.values() for s in cat}
    bm = get_masks(list(items.values())); cfg = CFG[("usod" if key == "usod" else "suim") if which == "own" else ("suim" if key == "usod" else "usod")]
    for cat, stems in sel.items():
        (out / cat).mkdir(parents=True, exist_ok=True)
        for stem in stems:
            row = df[df.image == stem].iloc[0]; it = items[stem]; G = load_gt(data, stem, gt); img = np.asarray(Image.open(it["img"]).convert("RGB").resize(G.shape[::-1]))
            X = dense(model, it["img"]); grid = X.shape[:2]; Z = pca64(X.reshape(-1, X.shape[-1]))
            _, _, prob, _, _ = per_image_gmm(Z, grid, cfg, G.shape, return_all=True)
            mk = masks[key][stem]
            panels = [("image", img, None), ("ground truth", G, None)] + [(n, bm[stem][n][0], bm[stem][n][1]) for n in ("k-means (border rule)", "dendrogram (gap rule)")]
            panels += [("GMM per-image", unp(mk[f"pi_{which}"]), row["iou_pi"]), ("GMM dataset-level", unp(mk[f"gl_{which}"]), row["iou_gl"]), ("GMM foreground probability", prob, None)]
            fig, axs = plt.subplots(1, 7, figsize=(21, 3.0))
            for ax, (t, im, v) in zip(axs, panels):
                ax.imshow(im, cmap=None if im.ndim == 3 else ("viridis" if im.dtype != bool and t.endswith("probability") else "gray")); ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values(): s_.set_visible(False)
                if v is not None: ax.text(.03, .04, f"IoU {v:.2f}", transform=ax.transAxes, color="white", fontsize=9, bbox=dict(fc="black", alpha=.65, lw=0, pad=1.5))
            fig.suptitle(f"{name} | {stem} | {cat}", fontsize=8, x=0.01, ha="left"); plt.tight_layout()
            plt.savefig(out / cat / f"{stem}_iou{row['iou_pi']:.2f}.png", dpi=80); plt.close(); nitems += 1
    print(name, "collages done", flush=True)
md += ["## Per-image examples\n", f"{nitems} PNGs, one per image, in `examples/<setting>/<GT>/{{good,bad,gmm_beats_baselines}}/` (10 best, 10 worst, 5 largest gains over the better label-free baseline for the per-image GMM). "
       "Panels: image, ground truth, k-means, dendrogram, GMM per-image, GMM dataset-level, GMM foreground probability.", "", "`summary_bars.png` shows the mean IoU of all methods per setting; `per_image_*.csv` hold every number."]
(RES / "report.md").write_text("\n".join(md)); print("\n".join(md[:60]))
