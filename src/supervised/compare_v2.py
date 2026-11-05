"""Unified-label ("subject") results: pooled model, cross-dataset X1/X2, tables, bars, per-image collages.

Settings (all scored against the subject GT, native resolution):
  pool_usod : pooled model on the 100 held-out USOD test images
  pool_suim : pooled model on the 110 SUIM test images
  X1        : trained on USOD only -> 110 SUIM test images
  X2        : trained on SUIM only -> all 300 USOD images (and the 100-image test subset)
Baselines are label-free: k-means border rule and dendrogram gap rule (USOD IoUs from the stored tables,
SUIM IoUs regenerated against the subject GT and cached).
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from baseline_masks import get_masks
from export_examples import export

V2 = ROOT / "results" / "subject_v2"
R = ROOT / "results"
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
evpos = {i: k for k, i in enumerate(ev)}
KM, DG = "k-means (border rule)", "dendrogram (gap rule)"


def stored_base(ds_dir, dd_dir):
    km = pd.read_csv(R / "kmeans" / ds_dir / "cluster_union_stride3_k6" / "metrics_union.csv")
    km = km[km.feat.astype(str) == "768"] if "feat" in km else km
    km["image"] = km["image"].str.replace(".jpg", "", regex=False)
    dg = pd.read_csv(R / "dendrogram" / dd_dir / "metrics_s3.csv"); dg["image"] = dg["image"].str.replace(".jpg", "", regex=False)
    dg = dg[(dg.feat.astype(str) == "768") & (dg.spatial == "free") & (dg.linkage == "average") & (dg.rule == "gap_border")]
    return km.set_index("image")["border_iou"], dg.set_index("image")["iou"]


def suim_base():
    """k-means / dendrogram IoU against the subject GT for the 110 SUIM test images (cached)."""
    f = V2 / "baseline_iou_suim_test.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"image": str}).set_index("image")
    items = [dict(r, gt_mode="suim_subject") for r in recs if r["ds"] == "suim" and r["split"] == "test"]
    rows = []
    for k, it in enumerate(items):
        m = get_masks([it])[it["stem"]]
        rows.append(dict(image=it["stem"], **{KM: m[KM][1], DG: m[DG][1]}))
        if k % 20 == 0: print("baseline", k, flush=True)
    d = pd.DataFrame(rows); d.to_csv(f, index=False)
    return d.set_index("image")


def load_preds(setting):
    """-> (records indices, {stage: [n,TH,TW] float32}, {stage: thr})"""
    if setting.startswith("pool"):
        P = {s: np.load(pred_path(f"v2_pool_{s}", "evalset")).astype(np.float32)[:, 0] for s in "ABC"}
        thr = {s: json.load(open(V2 / "pool" / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
        idx = ev
        pos = [evpos[i] for i in idx]
        return idx, {s: P[s][pos] for s in P}, thr
    tag = {"X1": "x1", "X2": "x2"}[setting]
    idx = list(np.load(OUT / "preds" / f"v2_{tag}_index.npy"))
    P = {s: np.load(OUT / "preds" / f"v2_{tag}_{s}.npy").astype(np.float32) for s in "ABC"}
    thr = {s: json.load(open(V2 / tag / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
    return idx, P, thr


def per_image(setting, ds, split_filter):
    idx, P, thr = load_preds(setting)
    kb, db = stored_base("usod", "usod") if ds == "usod" else (None, None)
    sb = suim_base() if ds == "suim" else None
    rows = []
    for k, i in enumerate(idx):
        r = recs[i]
        if r["ds"] != ds or not split_filter(r): continue
        gt = load_gt(r["data"], r["stem"], "suim_subject" if ds == "suim" else "usod")
        row = dict(image=r["stem"], split=r["split"], area=float(gt.mean()), rec=i, k=k)
        for s in "ABC":
            row[f"model {s}"] = mask_metrics(prob_to_mask(P[s][k], gt.shape, thr[s]), gt)["iou"]
        if ds == "usod":
            row[KM], row[DG] = kb.get(r["stem"], np.nan), db.get(r["stem"], np.nan)
        else:
            row[KM], row[DG] = sb.loc[r["stem"], KM], sb.loc[r["stem"], DG]
        rows.append(row)
    return pd.DataFrame(rows), P, thr


SETTINGS = {
    "pool_usod": dict(ds="usod", filt=lambda r: r["split"] == "test", title="pooled model, USOD test (100)"),
    "pool_suim": dict(ds="suim", filt=lambda r: r["split"] == "test", title="pooled model, SUIM test (110)"),
    "X1": dict(ds="suim", filt=lambda r: r["split"] == "test", title="X1: train USOD only -> SUIM test (110)"),
    "X2": dict(ds="usod", filt=lambda r: True, title="X2: train SUIM only -> all USOD (300)"),
}
ORDER = [KM, DG, "model A", "model B", "model C"]

if __name__ == "__main__":
    md = ["# Unified subject label: results\n",
          "Subject = fish/divers/robots/wrecks if present, else reef/plants (SUIM); USOD salient object unchanged. One output channel, one threshold chosen on validation. Baselines are label-free. IoU at native GT resolution.\n"]
    summ, tabs = [], {}
    for name, sp in SETTINGS.items():
        df, P, thr = per_image(name, sp["ds"], sp["filt"])
        df.to_csv(V2 / f"per_image_{name}.csv", index=False)
        tabs[name] = (df, P, thr)
        d = df[df[DG].notna()]
        md += [f"## {sp['title']}\n", f"n = {len(d)} images with baseline results ({len(df)} scored for models)\n",
               "| Method | mean IoU [95% CI] | IoU>0.5 | IoU<0.2 |", "|---|---|---|---|"]
        for m in ORDER:
            v = d[m].dropna(); lo, hi = bootstrap_ci(v.values)
            md.append(f"| {m} | **{v.mean():.3f}** [{lo:.3f}, {hi:.3f}] | {(v > .5).mean():.0%} | {(v < .2).mean():.0%} |")
            summ.append(dict(setting=name, method=m, n=len(v), iou=v.mean(), lo=lo, hi=hi))
        g = (d["model C"] - d[DG]).values; lo, hi = bootstrap_ci(g)
        md += ["", f"Paired gain model C minus dendrogram: **{g.mean():+.3f}** [{lo:+.3f}, {hi:+.3f}], better on {(g > 0).mean():.0%} of images.\n"]
        if name == "X2":
            t = d[d.split == "test"]
            md += [f"X2 on the 100-image USOD test subset: " + ", ".join(f"{m} {t[m].mean():.3f}" for m in ORDER) + "\n"]
    pd.DataFrame(summ).to_csv(V2 / "summary.csv", index=False)

    S = pd.DataFrame(summ)
    COL = {KM: "#9ca3af", DG: "#6b7280", "model A": "#bfdbfe", "model B": "#60a5fa", "model C": "#1d4ed8"}
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2), sharex=True)
    for ax, (name, sp) in zip(axs, SETTINGS.items()):
        g = S[S.setting == name].iloc[::-1]
        ax.barh(range(len(g)), g.iou, color=[COL[m] for m in g.method], height=.62, xerr=[g.iou - g.lo, g.hi - g.iou], error_kw=dict(ecolor="#1f2937", lw=1, capsize=2))
        for y, v in enumerate(g.iou): ax.text(min(v + .015, .93), y, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(g))); ax.set_yticklabels(g.method, fontsize=8); ax.set_xlim(0, 1)
        ax.set_title(sp["title"], fontsize=8.5, loc="left"); ax.set_xlabel("mean IoU (95% CI)")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    plt.tight_layout(); plt.savefig(V2 / "summary_bars.png", dpi=150); plt.close()

    # per-image collages: one file per image, all methods in one row
    md += ["## Per-image examples\n", "One PNG per image under `examples/{good,bad,baseline_wins}/` (10 best, 10 worst for model C, 5 where a label-free baseline beats it). Each file shows image | ground truth | k-means | dendrogram | model A | B | C with IoU on every mask.\n"]
    for name, sp in SETTINGS.items():
        df, P, thr = tabs[name]
        d = df[df[DG].notna()]
        items = {recs[i]["stem"]: dict(recs[i], gt_mode="suim_subject" if sp["ds"] == "suim" else "usod") for i in d.rec}
        krow = d.set_index("image")

        def panel_fn(stem, m, gt, krow=krow, P=P, thr=thr):
            k = int(krow.loc[stem, "k"])
            return [(n, *m[n]) for n in (KM, DG)] + [(f"model {s}", prob_to_mask(P[s][k], gt.shape, thr[s]), krow.loc[stem, f"model {s}"]) for s in "ABC"]
        rows = export(name, V2 / "examples", items, lambda it, sp=sp: load_gt(it["data"], it["stem"], "suim_subject" if sp["ds"] == "suim" else "usod"), panel_fn, d, "model C")
        print(name, len(rows), "collages", flush=True)
    open(V2 / "report.md", "w").write("\n".join(md))
    print("\n".join(md))
