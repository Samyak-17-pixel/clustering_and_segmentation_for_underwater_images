"""Hard-case ablation: score every variant (Sec. 23 baseline "", TTA "_tta", size-weighted loss "_sw", ...) per size bucket
and on reef-only scenes; val for selection, test / cross-dataset for reporting.

  python hard_cases.py report            # ablation.csv + report.md over all variants found on disk
  python hard_cases.py collages <sfx>    # one PNG per image: Sec. 23 model C vs the variant's model C
Variant layout (same as a training variant with tag <base><sfx>): results/subject_v2/<base><sfx>/stage_<S>/config.json and
outputs/preds/v2_<base><sfx>_<S>_evalset.npy (pool) | v2_<base><sfx>_<S>.npy + v2_<base><sfx>_index.npy (cross).
"""
import json, sys
import numpy as np, pandas as pd
from common import *
from gt_utils import suim_class_map, suim_subject

V2 = ROOT / "results" / "subject_v2"
HC = V2 / "hard_cases"; HC.mkdir(exist_ok=True)
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
evpos = {i: k for k, i in enumerate(ev)}
BUCKETS = [0, 0.05, 0.1, 0.3, 1.01]
BLAB = ["<5%", "5-10%", "10-30%", ">30%"]
_meta = {}


def meta(i):
    """(gt mask, area, label-rule branch) at native resolution, cached."""
    if i not in _meta:
        r = recs[i]
        if r["ds"] == "usod":
            gt, br = load_gt(r["data"], r["stem"], "usod"), 0
        else:
            gt = load_gt(r["data"], r["stem"], "suim_subject")
            br = suim_subject(suim_class_map(r["data"] / "masks" / f"{r['stem']}.bmp"))[1]
        _meta[i] = (gt, float(gt.mean()), br)
    return _meta[i]


def stages(base, sfx):
    d = V2 / f"{base}{sfx}"
    return [s for s in "ABC" if (d / f"stage_{s}" / "config.json").exists()]


def exists(base, sfx):
    return len(stages(base, sfx)) >= 2


def variant_preds(base, sfx, s):
    """-> (record indices, probs [n,TH,TW], thr)"""
    tag = f"{base}{sfx}"
    thr = json.load(open(V2 / tag / f"stage_{s}" / "config.json"))["thr"]["all"]
    if base == "pool":
        P = np.load(pred_path(f"v2_{tag}_{s}", "evalset")).astype(np.float32)[:, 0]
        return ev, P, thr
    return list(np.load(OUT / "preds" / f"v2_{tag}_index.npy")), np.load(OUT / "preds" / f"v2_{tag}_{s}.npy").astype(np.float32), thr


def score(base, sfx):
    rows = []
    for s in stages(base, sfx):
        idx, P, thr = variant_preds(base, sfx, s)
        for k, i in enumerate(idx):
            r = recs[i]
            if base == "pool" and r["split"] not in ("val", "test"): continue
            if base != "pool" and r["ds"] == ("usod" if base == "x1" else "suim") : continue
            gt, area, br = meta(i)
            rows.append(dict(variant=sfx or "base", setting=base, split=r["split"] if base == "pool" else "target", stage=s, ds=r["ds"],
                             image=r["stem"], area=area, branch=br, iou=mask_metrics(prob_to_mask(P[k], gt.shape, thr), gt)["iou"]))
    return pd.DataFrame(rows)


def summarise(d):
    d = d.copy(); d["size"] = pd.cut(d.area, BUCKETS, labels=BLAB)
    out = dict(n=len(d), iou=d.iou.mean())
    for b in BLAB:
        g = d[d["size"] == b]; out[b] = g.iou.mean() if len(g) else np.nan
    g = d[d.branch == 2]; out["reef-only"] = g.iou.mean() if len(g) else np.nan; out["n_reef"] = len(g)
    return out


def report():
    frames = []
    for sfx in sorted({p.name[len(b):] for p in V2.iterdir() if p.is_dir() for b in ("pool", "x1", "x2") if p.name.startswith(b)}):
        for base in ("pool", "x1", "x2"):
            if exists(base, sfx):
                f = HC / f"scores_{base}{sfx or '_base'}.csv"
                if f.exists(): d = pd.read_csv(f, dtype={"image": str})
                else: d = score(base, sfx); d.to_csv(f, index=False)
                frames.append(d)
    A = pd.concat(frames)
    rows = []
    for (v, setting, split, stage), g in A.groupby(["variant", "setting", "split", "stage"]):
        rows.append(dict(variant=v, setting=setting, split=split, stage=stage, **summarise(g)))
    T = pd.DataFrame(rows); T.to_csv(HC / "ablation.csv", index=False)
    md = ["# Hard-case ablation\n", "Mean IoU per object-size bucket and on reef-only scenes. `val` rows are used for selection, `test` / `target` rows are reported. Variants: base = Sec. 23, tta = test-time augmentation, sw = size-aware loss, l4 = last-4-layer features, hr = higher input resolution, reef = reef oversampling.\n"]
    for (setting, split), g in T.groupby(["setting", "split"]):
        md += [f"## {setting} / {split}\n", "| variant | stage | n | mean IoU | <5% | 5-10% | 10-30% | >30% | reef-only (n) |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g[g.stage.isin(["B", "C"])].sort_values(["stage", "variant"]).iterrows():
            md.append(f"| {r.variant} | {r.stage} | {r.n} | **{r.iou:.3f}** | " + " | ".join("-" if pd.isna(r[b]) else f"{r[b]:.3f}" for b in BLAB) + f" | {'-' if pd.isna(r['reef-only']) else format(r['reef-only'], '.3f')} ({r.n_reef}) |")
        md.append("")
    (HC / "report.md").write_text("\n".join(md)); print("\n".join(md))


def collages(sfx):
    """Per-variant results in the Sec. 23 layout: results/subject_v2/hard_cases/<variant>/{per_image_*.csv, summary.csv, examples/}.
    One PNG per image: image | GT | k-means | dendrogram | model C (Sec. 23) | model C (variant). Folders: good, bad,
    baseline_wins (selected on the variant's model C), fixed (largest gain over Sec. 23), regressed."""
    from compare_v2 import stored_base, suim_base, KM, DG
    from baseline_masks import get_masks
    from export_examples import export
    name = sfx.strip("_") or "base"
    root = HC / name; root.mkdir(exist_ok=True)
    kb, db = stored_base("usod", "usod"); sb = suim_base()
    summ = []
    for setting, base, ds, split in [("pool_usod", "pool", "usod", "test"), ("pool_suim", "pool", "suim", "test"),
                                     ("X1", "x1", "suim", "target"), ("X2", "x2", "usod", "target")]:
        if not (HC / f"scores_{base}{sfx}.csv").exists() or "C" not in stages(base, sfx):
            print(setting, "skipped (variant not trained for this setting)"); continue
        new = pd.read_csv(HC / f"scores_{base}{sfx}.csv", dtype={"image": str}); old = pd.read_csv(HC / f"scores_{base}_base.csv", dtype={"image": str})
        sel = lambda d, s: d[(d.stage == s) & (d.ds == ds) & (d.split == split)].set_index("image")
        df = pd.DataFrame({"area": sel(new, "C").area, "model C (Sec. 23)": sel(old, "C").iou, "model C (new)": sel(new, "C").iou,
                           "model A (new)": sel(new, "A").iou, "model B (new)": sel(new, "B").iou})
        df[KM] = [kb.get(s, np.nan) if ds == "usod" else sb[KM].get(s, np.nan) for s in df.index]
        df[DG] = [db.get(s, np.nan) if ds == "usod" else sb[DG].get(s, np.nan) for s in df.index]
        df["gain"] = df["model C (new)"] - df["model C (Sec. 23)"]
        df = df.reset_index(); df.to_csv(root / f"per_image_{setting}.csv", index=False)
        for m in [KM, DG, "model C (Sec. 23)", "model A (new)", "model B (new)", "model C (new)"]:
            v = df[m].dropna(); lo, hi = bootstrap_ci(v.values)
            summ.append(dict(setting=setting, method=m, n=len(v), iou=v.mean(), lo=lo, hi=hi))
        d = df[df[DG].notna()]
        idx, Pn, tn = variant_preds(base, sfx, "C"); _, Po, to = variant_preds(base, "", "C")
        pos = {recs[i]["stem"]: k for k, i in enumerate(idx) if recs[i]["ds"] == ds}
        items = {recs[i]["stem"]: dict(recs[i], gt_mode="suim_subject" if ds == "suim" else "usod") for i in idx if recs[i]["ds"] == ds and recs[i]["stem"] in set(d.image)}
        def panel_fn(stem, m, gt, Pn=Pn, Po=Po, tn=tn, to=to, pos=pos, d=d.set_index("image")):
            k = pos[stem]
            return [(n, *m[n]) for n in (KM, DG)] + [("model C (Sec. 23)", prob_to_mask(Po[k], gt.shape, to), d.loc[stem, "model C (Sec. 23)"]),
                                                     ("model C (new)", prob_to_mask(Pn[k], gt.shape, tn), d.loc[stem, "model C (new)"])]
        g = d.sort_values("gain")
        extra = {"fixed": list(g.image[::-1][:10]), "regressed": list(g[g.gain < 0].image[:5])}
        export(setting, root / "examples", items, lambda it, ds=ds: load_gt(it["data"], it["stem"], "suim_subject" if ds == "suim" else "usod"),
               panel_fn, d, "model C (new)", extra)
        print(setting, "done", flush=True)
    pd.DataFrame(summ).to_csv(root / "summary.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    elif sys.argv[1] == "collages":
        collages(sys.argv[2] if len(sys.argv) > 2 else "")
