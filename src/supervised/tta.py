"""E1: test-time augmentation for the unified-label models (no retraining).

Views: original, h-flip, 0.75x, 1.25x (network input scaled, logits resized back, probabilities averaged).
Thresholds are re-tuned on the SOURCE validation images only.

  python tta.py pool | x1 | x2
Writes the same layout as a training variant named <tag>_tta:
  outputs/preds/v2_<tag>_tta_<S>_evalset.npy (pool) or v2_<tag>_tta_<S>.npy + v2_<tag>_tta_index.npy (cross),
  results/subject_v2/<tag>_tta/stage_<S>/config.json
"""
import argparse, json
import numpy as np, torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from common import *

VIEWS = [(1.0, False), (1.0, True), (0.75, False), (1.25, False)]


def load_models(b):
    ck = {s: torch.load(OUT / "ckpt" / f"v2_{b}_{s}.pt") for s in "ABC"}
    mk = lambda c: Head(**{k: v for k, v in c.items() if k not in ("lr", "wd")}, out=1).cuda().eval()
    heads = {}
    for s in "AB":
        heads[s] = mk(ck[s]["cfg"]); heads[s].load_state_dict(ck[s]["state"])
    frozen = load_backbone()
    bb = get_peft_model(load_backbone(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["qkv", "proj"], bias="none")).eval()
    sd = bb.state_dict(); sd.update(ck["C"]["lora"]); bb.load_state_dict(sd)
    hc = mk(ck["C"]["head_cfg"]); hc.load_state_dict(ck["C"]["head"])
    return frozen, bb.base_model.model, heads, hc


@torch.no_grad()
def predict_tta(models, path, views=VIEWS):
    frozen, base, heads, hc = models
    x0 = to_tensor(load_image(path))[None].cuda()
    acc = {s: 0 for s in "ABC"}
    for sc, flip in views:
        hs, ws = round(H * sc / 14) * 14, round(W * sc / 14) * 14
        x = x0 if (hs, ws) == (H, W) else F.interpolate(x0, size=(hs, ws), mode="bilinear", align_corners=False, antialias=sc < 1)
        if flip: x = x.flip(-1)
        x = normalise(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = extract(frozen, x).float()
            t = base.forward_features(x)["x_norm_patchtokens"]
        fc = t.float().transpose(1, 2).reshape(1, -1, hs // 14, ws // 14)
        for s, lg in (("A", heads["A"](f)), ("B", heads["B"](f)), ("C", hc(fc))):
            p = torch.sigmoid(up(lg))[0, 0]
            acc[s] = acc[s] + (p.flip(-1) if flip else p)
    return {s: (acc[s] / len(views)).cpu().numpy() for s in acc}


def tune_thr(P, gts_low):
    best = (-1, 0.5)
    for t in np.arange(0.2, 0.81, 0.05):
        sc = np.mean([mask_metrics(P[k] > t, gts_low[k] > 127)["iou"] for k in range(len(P))])
        if sc > best[0]: best = (sc, float(t))
    return best[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tag", choices=["pool", "x1", "x2"]); b = ap.parse_args().tag
    recs = build_records()
    gts = np.load(OUT / "features" / "gt_subject.npy")
    models = load_models(b)
    if b == "pool":
        ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
        sets = {"eval": ev}
        val_ids = [i for i in ev if recs[i]["split"] == "val"]
    else:
        src = "usod" if b == "x1" else "suim"
        tgt = list(np.load(OUT / "preds" / f"v2_{b}_index.npy"))
        val_ids = [i for i, r in enumerate(recs) if r["ds"] == src and r["split"] == "val"]
        sets = {"eval": tgt}
    allids = sorted(set(val_ids) | set(sets["eval"]))
    cache = {i: predict_tta(models, recs[i]["img"]) for i in allids}
    print("predicted", len(allids), "images", flush=True)
    rd = ROOT / "results" / "subject_v2" / f"{b}_tta"
    for s in "ABC":
        Pv = np.stack([cache[i][s] for i in val_ids])
        thr = tune_thr(Pv, gts[val_ids])
        P = np.stack([cache[i][s] for i in sets["eval"]]).astype(np.float16)
        (rd / f"stage_{s}").mkdir(parents=True, exist_ok=True)
        json.dump(dict(thr={"all": thr}, variant="tta", views=VIEWS), open(rd / f"stage_{s}" / "config.json", "w"))
        if b == "pool":
            np.save(pred_path(f"v2_pool_tta_{s}", "evalset"), P[:, None])
        else:
            np.save(OUT / "preds" / f"v2_{b}_tta_{s}.npy", P)
            np.save(OUT / "preds" / f"v2_{b}_tta_index.npy", np.array(sets["eval"]))
        print(s, "thr", thr, flush=True)
