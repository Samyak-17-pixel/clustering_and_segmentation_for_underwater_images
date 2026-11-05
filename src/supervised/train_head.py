"""Stages A and B: train a head on cached, frozen DINOv2 features.

  python train_head.py A    # linear probe
  python train_head.py B    # small dilated-conv head
Hyper-parameters are chosen on the validation split only. Writes
results/supervised/stage_X/tuning.csv, outputs/ckpt/stage_X.pt and outputs/preds/stage_X_{val,test}.npy.
"""
import itertools, sys, csv, time
import numpy as np
import torch
from common import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["A", "B"])
ap.add_argument("--train", choices=["usod", "suim"], default=None, help="cross-dataset mode: train/tune on this dataset only")
ap.add_argument("--tag", default=None, help="output tag for cross-dataset mode (e.g. x1)")
ap.add_argument("--label", choices=["legacy", "subject", "subject2"], default="legacy",
                help="subject = unified salient-object/reef label, one output channel (results/subject_v2)")
ap.add_argument("--loss", choices=["plain", "sizeweight"], default="plain", help="sizeweight = per-image weight by subject area")
ap.add_argument("--feats", default="feats.npy", help="cached feature file in outputs/features (e.g. feats_l4.npy)")
ap.add_argument("--seed", type=int, default=0, help="seed != 0 re-trains the seed-0 best config only")
args = ap.parse_args()
stage, SRC = args.stage, args.train
SUBJ = args.label != "legacy"
SD = f"_s{args.seed}" if args.seed else ""
TAG = args.tag or "pool"
cid = (lambda ds: 0) if SUBJ else (lambda ds: CH[ds])
dev = "cuda"
recs = build_records()
feats = np.load(OUT / "features" / args.feats, mmap_mode="r")
gts = np.load(OUT / "features" / (f"gt_{args.label}.npy" if SUBJ else "gt.npy"))
idx = {s: np.array([i for i, r in enumerate(recs) if r["split"] == s and (SRC is None or r["ds"] == SRC)]) for s in ("train", "val", "test")}
chan = torch.tensor([cid(r["ds"]) for r in recs])
is_usod = np.array([r["ds"] == "usod" for r in recs])

tr = idx["train"]
Ftr = torch.from_numpy(np.ascontiguousarray(feats[tr[0]:tr[-1] + 1]) if False else np.stack([feats[i] for i in tr])).to(dev)
Gtr = torch.from_numpy(gts[tr]).to(dev)
Ctr = chan[tr].to(dev)
tr_usod = np.where(is_usod[tr])[0]
tr_suim = np.where(~is_usod[tr])[0]


def predict(model, ids, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            f = torch.from_numpy(np.stack([feats[j] for j in ids[i:i + bs]])).to(dev).float()
            out.append(torch.sigmoid(up(model(f))).cpu())
    return torch.cat(out)            # [n,2,TH,TW]


def val_score(model, split="val"):
    ids = idx[split]
    p = predict(model, ids)
    scores = {"usod": [], "suim": []}
    for k, i in enumerate(ids):
        ds = recs[i]["ds"]
        pr = p[k, cid(ds)].numpy() > 0.5
        g = gts[i] > 127
        scores[ds].append(mask_metrics(pr, g)["iou"])
    return (float(np.mean(scores["usod"])) if scores["usod"] else 0.0), (float(np.mean(scores["suim"])) if scores["suim"] else 0.0), p


def train(cfg, epochs, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Head(cfg["kind"], width=cfg.get("width", 256), depth=cfg.get("depth", 3), drop=cfg.get("drop", 0.1),
                 out=1 if SUBJ else 2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per = 30
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * steps_per, pct_start=0.1)
    best = (-1, None, None)
    for ep in range(epochs):
        model.train()
        for _ in range(steps_per):
            if SRC is None:
                ids = np.concatenate([np.random.choice(tr_usod, 16), np.random.choice(tr_suim, 16)])
            else:
                ids = np.random.choice(tr_usod if SRC == "usod" else tr_suim, 32)
            f = Ftr[ids].float(); g = Gtr[ids].float() / 255; c = Ctr[ids]
            if np.random.rand() < 0.5:
                f, g = f.flip(-1), g.flip(-1)
            loss = seg_loss(model(f), g, c, size_weight(g) if args.loss == "sizeweight" else None)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 3 == 2 or ep == epochs - 1:
            u, s, _ = val_score(model)
            sc = u if SRC == "usod" else s if SRC == "suim" else (u + s) / 2
            if sc > best[0]:
                best = (sc, {k: v.clone() for k, v in model.state_dict().items()}, (ep, u, s))
    model.load_state_dict(best[1])
    return model, best[0], best[2]


if stage == "A":
    grid = [dict(kind="linear", lr=lr, wd=wd) for lr, wd in itertools.product([3e-2, 1e-2, 3e-3], [1e-4, 1e-2])]
    epochs = 150
else:
    grid = [dict(kind="conv", lr=lr, wd=1e-2, width=w, depth=d) for lr, w, d in
            itertools.product([1e-3] if SRC else [1e-3, 3e-4], [128, 256], [2, 3])]
    epochs = 100

XRES = ROOT / "results" / "cross_dataset"
if SUBJ:
    res_dir = ROOT / "results" / ("subject_v3" if args.label == "subject2" else "subject_v2") / TAG / f"stage_{stage}{SD}"
    ck_name = f"v2_{TAG}_{stage}{SD}"
else:
    res_dir = XRES / args.tag / f"stage_{stage}" if SRC else RES / f"stage_{stage}"
    ck_name = f"cross_{args.tag}_{stage}" if SRC else f"stage_{stage}"
res_dir.mkdir(parents=True, exist_ok=True)
if args.seed:   # later seeds: no new search, same config as seed 0 (hyper-parameters were chosen once on val)
    cfg0 = json.load(open(res_dir.parent / f"stage_{stage}" / "config.json"))["cfg"]
    grid = [cfg0]
rows, best_all = [], (-1, None, None)
for cfg in grid:
    t0 = time.time()
    model, sc, info = train(cfg, epochs, seed=args.seed)
    rows.append(dict(**cfg, val_score=sc, best_epoch=info[0], val_iou_usod=info[1], val_iou_suim=info[2], secs=time.time() - t0))
    print(rows[-1], flush=True)
    if sc > best_all[0]:
        best_all = (sc, cfg, model)
with open(res_dir / "tuning.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

sc, cfg, model = best_all
print("BEST", cfg, sc)
# per-channel threshold tuned on val
_, _, pv = val_score(model)
thr = {}
groups = {"all": ["usod", "suim"] if not SRC else [SRC]} if SUBJ else {ds: [ds] for ds in ((SRC,) if SRC else ("usod", "suim"))}
for name, dss in groups.items():
    ids = [k for k, i in enumerate(idx["val"]) if recs[i]["ds"] in dss]
    best_t, best_s = 0.5, -1
    for t in np.arange(0.2, 0.81, 0.05):
        s = np.mean([mask_metrics(pv[k, 0 if SUBJ else CH[recs[idx["val"][k]]["ds"]]].numpy() > t, gts[idx["val"][k]] > 127)["iou"] for k in ids])
        if s > best_s: best_t, best_s = float(t), float(s)
    thr[name] = best_t
(OUT / "ckpt").mkdir(exist_ok=True)
torch.save(dict(cfg=cfg, state=model.state_dict(), thr=thr), OUT / "ckpt" / f"{ck_name}.pt")
json.dump(dict(cfg=cfg, val_score=sc, thr=thr, train=SRC, label=args.label, seed=args.seed), open(res_dir / "config.json", "w"), indent=1)

if SRC is None:
    ev_ids = np.array([i for i, r in enumerate(recs) if r["split"] in ("val", "test")])
    P = predict(model, ev_ids)
    np.save(pred_path(f"v2_{TAG}_{stage}{SD}" if SUBJ else f"stage_{stage}", "evalset"), P.numpy().astype(np.float16))
    print("saved preds", P.shape, "thr", thr)
else:
    # target = the other dataset: all USOD images (X2) or the 110 SUIM test images (X1)
    tgt = np.array([i for i, r in enumerate(recs) if (r["ds"] == "usod" if SRC == "suim" else r["ds"] == "suim" and r["split"] == "test")])
    assert not set(tgt) & set(idx["train"]) and not set(tgt) & set(idx["val"])
    assert not {recs[i]["ds"] for i in tgt} & {SRC}
    P = predict(model, tgt)[:, cid(SRC)]
    pn = f"v2_{TAG}_{stage}{SD}" if SUBJ else f"cross_{args.tag}_{stage}"
    np.save(OUT / "preds" / f"{pn}.npy", P.numpy().astype(np.float16))
    np.save(OUT / "preds" / (f"v2_{TAG}_index.npy" if SUBJ else f"cross_{args.tag}_index.npy"), tgt)
    print("saved cross preds", P.shape, "thr", thr)
