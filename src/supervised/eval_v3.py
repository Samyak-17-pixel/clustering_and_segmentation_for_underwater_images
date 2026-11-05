"""In-domain regression check for the new label (Sec. 26): the Sec. 23 model (pool) and the new model (pool2) scored on the same test images
against the NEW ground truth (USOD salient object; SUIM animals/objects only, empty where there is none).
Writes results/subject_v3/in_domain.md and per_image_in_domain.csv."""
import json
import numpy as np, pandas as pd
from common import *

V3 = ROOT / "results" / "subject_v3"; V3.mkdir(parents=True, exist_ok=True)
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
evpos = {i: k for k, i in enumerate(ev)}
MODELS = {"Sec. 23 (pool, old rule)": ("pool", "subject_v2"), "Sec. 26 (pool2, new rule)": ("pool2", "subject_v3")}
rows = []
for name, (tag, rd) in MODELS.items():
    for s in "ABC":
        P = np.load(pred_path(f"v2_{tag}_{s}", "evalset")).astype(np.float32)[:, 0]
        thr = json.load(open(ROOT / "results" / rd / tag / f"stage_{s}" / "config.json"))["thr"]["all"]
        for i in ev:
            r = recs[i]
            if r["split"] != "test": continue
            gt = load_gt(r["data"], r["stem"], "usod" if r["ds"] == "usod" else "suim_subject2")
            pred = prob_to_mask(P[evpos[i]], gt.shape, thr)
            m = mask_metrics(pred, gt)
            rows.append(dict(model=name, stage=s, ds=r["ds"], image=r["stem"], has_subject=bool(gt.any()), iou=m["iou"], pred_share=float(pred.mean())))
d = pd.DataFrame(rows); d.to_csv(V3 / "per_image_in_domain.csv", index=False)
md = ["# In-domain check with the new label (test images, never used for any choice)\n",
      "Both models are scored against the NEW ground truth: USOD salient object; SUIM animals / divers / robots / wrecks only (images without one have an empty mask: IoU 1 if the model predicts nothing, else 0).\n",
      "| model | stage | USOD test IoU (100) | SUIM test IoU, images with a subject (107) | SUIM images without a subject (3): false alarms |", "|---|---|---|---|---|"]
for name in MODELS:
    for s in "ABC":
        g = d[(d.model == name) & (d.stage == s)]
        u = g[g.ds == "usod"].iou; su = g[(g.ds == "suim") & g.has_subject].iou; ne = g[(g.ds == "suim") & ~g.has_subject]
        md.append(f"| {name} | {s} | {u.mean():.3f} | {su.mean():.3f} | {(ne.pred_share > 0.01).sum()} of {len(ne)} |")
(V3 / "in_domain.md").write_text("\n".join(md)); print("\n".join(md))
