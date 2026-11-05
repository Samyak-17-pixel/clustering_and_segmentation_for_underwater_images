"""Stage D: refine the Stage-C masks with SAM (facebook/sam-vit-base, no training).

For each connected component of the Stage-C mask we prompt SAM with a padded box + the component's
interior point, keep the 3 multimask candidates, and store them. Which candidate/fusion rule to use is
then chosen on the validation split only (rules in `fuse`).
  python sam_refine.py
Writes outputs/preds/stage_D_cands.pkl and outputs/preds/stage_D_evalset.npy (final probabilities).
"""
import pickle, itertools, csv, time
import numpy as np
import torch
from scipy import ndimage
from transformers import SamModel, SamProcessor
from tqdm import tqdm
from common import *

dev = "cuda"
recs = build_records()
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
gts = np.load(OUT / "features" / "gt.npy")
P = np.load(pred_path("stage_C", "evalset")).astype(np.float32)          # [400,2,TH,TW]
thr = json.load(open(RES / "stage_C" / "config.json"))["thr"]
SH, SW = H, W                                                            # SAM candidate resolution (462x616)
MAX_COMP, MIN_AREA = 6, 0.002

proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()


def up_to(a, shape):
    return F.interpolate(torch.from_numpy(a)[None, None].float(), size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()


CP = OUT / "preds" / "stage_D_cands.pkl"
cands = pickle.load(open(CP, "rb")) if CP.exists() else {}
t0 = time.time()
for k, i in enumerate(tqdm(ev) if not cands else []):
    r = recs[i]; ch = CH[r["ds"]]
    prob = up_to(P[k, ch], (SH, SW))
    m = prob > thr[r["ds"]]
    lab, n = ndimage.label(m)
    comps = []
    for c in range(1, n + 1):
        cm = lab == c
        if cm.mean() < MIN_AREA:
            continue
        comps.append((cm.sum(), cm))
    comps = [c[1] for c in sorted(comps, key=lambda t: -t[0])[:MAX_COMP]]
    out = []
    if comps:
        img = load_image(r["img"], (SW, SH))
        boxes, pts = [], []
        for cm in comps:
            ys, xs = np.where(cm)
            padx, pady = 0.05 * (xs.max() - xs.min() + 1), 0.05 * (ys.max() - ys.min() + 1)
            boxes.append([max(xs.min() - padx, 0), max(ys.min() - pady, 0), min(xs.max() + padx, SW - 1), min(ys.max() + pady, SH - 1)])
            dist = ndimage.distance_transform_edt(cm)
            py, px = np.unravel_index(dist.argmax(), dist.shape)
            pts.append([[px, py]])
        inp = proc(img, input_boxes=[boxes], input_points=[pts], input_labels=[[[1]] * len(comps)], return_tensors="pt").to(dev)
        with torch.no_grad():
            o = sam(**{k2: v for k2, v in inp.items() if k2 in ("pixel_values", "input_boxes", "input_points", "input_labels")}, multimask_output=True)
        masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]  # [n,3,SH,SW]
        scores = o.iou_scores.cpu()[0].numpy()                                                         # [n,3]
        for j, cm in enumerate(comps):
            out.append(dict(comp=np.packbits(cm), masks=np.packbits(masks[j].numpy(), axis=None), scores=scores[j]))
    cands[k] = out
print(f"SAM candidates in {time.time()-t0:.0f}s")
pickle.dump(cands, open(CP, "wb"))


def unpack(b, n):
    return np.unpackbits(b)[:n * SH * SW].reshape((n, SH, SW) if n > 1 else (SH, SW)).astype(bool)


def fuse(k, rule, probfull, thr_ds):
    """rule = (select, accept_iou, mix). Returns float prob [SH,SW]."""
    select, acc, mix = rule
    if acc > 1: return (probfull > thr_ds).astype(np.float32)      # SAM off: Stage-C mask unchanged
    base = probfull
    out = np.zeros((SH, SW), np.float32)
    coarse = base > thr_ds
    if not cands[k]:
        return base
    refined = np.zeros((SH, SW), bool)
    kept_coarse = coarse.copy()
    for c in cands[k]:
        cm = unpack(c["comp"], 1)
        ms = unpack(c["masks"], 3)
        ious = np.array([(m & cm).sum() / max((m | cm).sum(), 1) for m in ms])
        j = int(np.argmax(ious)) if select == "iou" else int(np.argmax(c["scores"]))
        if ious[j] >= acc:
            refined |= ms[j]
            kept_coarse &= ~cm
    final = refined | kept_coarse
    final = final.astype(np.float32)
    return final if mix == 1 else mix * final + (1 - mix) * base


def eval_rule(rule, split="val"):
    # returns (mean IoU on USOD val, mean IoU on SUIM val)
    sc = {"usod": [], "suim": []}
    for k, i in enumerate(ev):
        r = recs[i]
        if r["split"] != split: continue
        ch = CH[r["ds"]]
        pf = up_to(P[k, ch], (SH, SW))
        f = fuse(k, rule, pf, thr[r["ds"]])
        thr_f = 0.5 if rule[2] == 1 else (thr[r["ds"]] + 0.5) / 2
        g = up_to(gts[i].astype(np.float32) / 255, (SH, SW)) > 0.5
        sc[r["ds"]].append(mask_metrics(f > 0.5, g)["iou"])
    return float(np.mean(sc["usod"])), float(np.mean(sc["suim"]))


rows = []
base_val = None
for rule in itertools.product(["iou", "score"], [0.5, 0.7, 0.85, 0.9, 0.95, 1.5], [1.0, 0.5]):
    u, s = eval_rule(rule)
    rows.append(dict(select=rule[0], accept_iou=rule[1], mix=rule[2], val_iou_usod=u, val_iou_suim=s, val_mean=(u + s) / 2))
    print(rows[-1], flush=True)
# 'no SAM' reference on the same footing
res_dir = RES / "stage_D"; res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "tuning.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
bu = max(rows, key=lambda d: d["val_iou_usod"]); bs = max(rows, key=lambda d: d["val_iou_suim"])
rules = {"usod": (bu["select"], bu["accept_iou"], bu["mix"]), "suim": (bs["select"], bs["accept_iou"], bs["mix"])}
json.dump(dict(rule_per_dataset=rules, val_usod=bu["val_iou_usod"], val_suim=bs["val_iou_suim"], note="accept_iou>1 means SAM off (Stage-C mask kept)"), open(res_dir / "config.json", "w"), indent=1)
print("BEST", rules)

Pd = np.zeros_like(P)
for k, i in enumerate(ev):
    r = recs[i]; ch = CH[r["ds"]]
    pf = up_to(P[k, ch], (SH, SW))
    f = fuse(k, rules[r["ds"]], pf, thr[r["ds"]])
    # rescale so that 0.5 is the decision boundary for the evaluator: store as-is at half res
    Pd[k, ch] = up_to(f, (TH, TW))
    Pd[k, 1 - ch] = P[k, 1 - ch]
np.save(pred_path("stage_D", "evalset"), Pd.astype(np.float16))
print("saved", Pd.shape)
