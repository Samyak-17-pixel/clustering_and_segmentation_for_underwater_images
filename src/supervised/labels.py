"""Unified "subject" labels (see gt_utils).

  python labels.py            # subject : salient objects, or the reef/plants when the scene has nothing else (Sec. 23)
  python labels.py subject2   # subject2: animals / divers / robots / wrecks only; reef, plants, sand, water = background (Sec. 26)

Writes outputs/features/gt_<label>.npy  uint8 [N,231,308] in build_records() order (USOD rows = the USOD mask),
results/subject_v2|v3/label_stats.json (branch counts, agreement with SUIM Saliency on the test images).
"""
import json, sys
import numpy as np
from common import *
from gt_utils import suim_class_map, suim_subject, suim_subject2  # noqa: E402

LABEL = sys.argv[1] if len(sys.argv) > 1 else "subject"
RULE = {"subject": suim_subject, "subject2": suim_subject2}[LABEL]

if __name__ == "__main__":
    recs = build_records()
    out = np.zeros((len(recs), TH, TW), np.uint8)
    branch = {}
    for i, r in enumerate(recs):
        if r["ds"] == "usod":
            out[i] = (gt_soft(r, "usod") * 255).round().astype(np.uint8)
            continue
        m, b = RULE(suim_class_map(r["data"] / "masks" / f"{r['stem']}.bmp"))
        branch[(r["split"], r["stem"])] = b
        out[i] = (np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((TW, TH), Image.BOX)))
    d = OUT / "features"
    np.save(d / f"gt_{LABEL}.npy", out)
    stats = {}
    for sp in ("train", "val", "test"):
        bs = [b for (s, _), b in branch.items() if s == sp]
        stats[sp] = {f"branch{k}": bs.count(k) for k in (1, 2, 3)}
    # agreement with the SUIM Saliency masks (110 test images; independent annotation of "what is salient")
    ious = []
    for r in recs:
        if r["ds"] == "suim" and r["split"] == "test":
            sal = load_gt(r["data"], r["stem"], "suim_saliency")
            if not sal.any():
                continue
            sub = load_gt(r["data"], r["stem"], "suim_" + LABEL)
            nonw = load_gt(r["data"], r["stem"], "suim_nonwater")
            ious.append((mask_metrics(sub, sal)["iou"], mask_metrics(nonw, sal)["iou"]))
    ious = np.array(ious)
    stats["agreement_with_saliency_test"] = dict(n=len(ious), subject_iou=float(ious[:, 0].mean()),
                                                 nonwater_iou=float(ious[:, 1].mean()))
    rd = ROOT / "results" / ("subject_v3" if LABEL == "subject2" else "subject_v2")
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "label_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
