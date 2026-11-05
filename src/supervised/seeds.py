"""Seed variability of the pooled unified-label models: test IoU per seed, mean +/- std (across seeds)."""
import json
import numpy as np, pandas as pd
from common import *

V2 = ROOT / "results" / "subject_v2"
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
evpos = {i: k for k, i in enumerate(ev)}
rows = []
for s in "ABC":
    for seed in range(5):
        sd = f"_s{seed}" if seed else ""
        pf = pred_path(f"v2_pool_{s}{sd}", "evalset")
        cf = V2 / "pool" / f"stage_{s}{sd}" / "config.json"
        if not pf.exists() or not cf.exists(): continue
        P = np.load(pf).astype(np.float32)[:, 0]
        thr = json.load(open(cf))["thr"]["all"]
        row = dict(stage=s, seed=seed)
        for ds in ("usod", "suim"):
            v = []
            for i in ev:
                r = recs[i]
                if r["ds"] != ds or r["split"] != "test": continue
                gt = load_gt(r["data"], r["stem"], "suim_subject" if ds == "suim" else "usod")
                v.append(mask_metrics(prob_to_mask(P[evpos[i]], gt.shape, thr), gt)["iou"])
            row[f"iou_{ds}_test"] = float(np.mean(v))
        rows.append(row)
d = pd.DataFrame(rows); d.to_csv(V2 / "seeds_per_run.csv", index=False)
g = d.groupby("stage").agg(n_seeds=("seed", "count"), usod_mean=("iou_usod_test", "mean"), usod_std=("iou_usod_test", "std"),
                           suim_mean=("iou_suim_test", "mean"), suim_std=("iou_suim_test", "std"))
g.to_csv(V2 / "seeds_summary.csv")
md = ["| Stage | seeds | USOD test IoU | SUIM test IoU |", "|---|---|---|---|"]
for s, r in g.iterrows():
    md.append(f"| {s} | {int(r.n_seeds)} | {r.usod_mean:.3f} +/- {r.usod_std:.3f} | {r.suim_mean:.3f} +/- {r.suim_std:.3f} |")
(V2 / "seeds.md").write_text("\n".join(md))
print("\n".join(md)); print(d.to_string())
