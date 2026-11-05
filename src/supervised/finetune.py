"""Stage C: LoRA fine-tuning of DINOv2 ViT-B/14 (qkv + proj of every block) together with the Stage-B head.

  python finetune.py [steps]
Augmentation: h-flip, random resized crop, brightness/contrast/saturation jitter, blue-green colour-cast jitter.
Model selection on the validation split only. Writes outputs/ckpt/stage_C.pt, outputs/preds/stage_C_evalset.npy.
"""
import sys, time, csv
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from common import *
from uw_aug import underwater_aug

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("steps", nargs="?", type=int, default=3000)
ap.add_argument("--train", choices=["usod", "suim"], default=None, help="cross-dataset mode: train/tune on this dataset only")
ap.add_argument("--tag", default=None)
ap.add_argument("--label", choices=["legacy", "subject", "subject2"], default="legacy")
ap.add_argument("--aug", choices=["none", "underwater"], default="none", help="underwater = haze / darkening / colour cast / contrast / blur (uw_aug.py)")
ap.add_argument("--neg", type=float, default=0.0, help="share of crops taken from a region without any subject pixels")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--loss", choices=["plain", "sizeweight"], default="plain")
args = ap.parse_args()
steps, SRC = args.steps, args.train
SUBJ = args.label != "legacy"
SD = f"_s{args.seed}" if args.seed else ""
TAG = args.tag or "pool"
cid = (lambda ds: 0) if SUBJ else (lambda ds: CH[ds])
dev = "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)
recs = build_records()
gts = np.load(OUT / "features" / (f"gt_{args.label}.npy" if SUBJ else "gt.npy"))
idx = {s: [i for i, r in enumerate(recs) if r["split"] == s and (SRC is None or r["ds"] == SRC)] for s in ("train", "val", "test")}
if SRC:   # target images: all USOD (X2) or the 110 SUIM test images (X1); never used for training or selection
    idx["tgt"] = [i for i, r in enumerate(recs) if (r["ds"] == "usod" if SRC == "suim" else r["ds"] == "suim" and r["split"] == "test")]
    assert not set(idx["tgt"]) & (set(idx["train"]) | set(idx["val"]))

print("loading images ...", flush=True)
imgs = {i: np.asarray(load_image(recs[i]["img"])) for i in idx["train"] + idx["val"] + (idx["tgt"] if SRC else idx["test"])}
tr_usod = [i for i in idx["train"] if recs[i]["ds"] == "usod"]
tr_suim = [i for i in idx["train"] if recs[i]["ds"] == "suim"]

ckb = f"v2_{TAG}_B" if SUBJ else (f"cross_{args.tag}_B" if SRC else "stage_B")   # head init: seed-matched B if it exists
if SD and (OUT / "ckpt" / f"{ckb}{SD}.pt").exists():
    ckb += SD
ck = torch.load(OUT / "ckpt" / f"{ckb}.pt")
head = Head(**{k: v for k, v in ck["cfg"].items() if k != "lr" and k != "wd"}, out=1 if SUBJ else 2).to(dev)
head.load_state_dict(ck["state"])

bb = load_backbone(dev)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["qkv", "proj"], bias="none")
bb = get_peft_model(bb, cfg)
bb.print_trainable_parameters()
base = bb.base_model.model


def forward(x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = base.forward_features(x)["x_norm_patchtokens"]
    f = t.float().transpose(1, 2).reshape(x.shape[0], -1, x.shape[2] // 14, x.shape[3] // 14)
    return head(f)


def make_batch(ids):
    xs, gs = [], []
    for i in ids:
        x = torch.from_numpy(imgs[i]).permute(2, 0, 1).float().div(255).to(dev)[None]
        g = torch.from_numpy(gts[i]).float().div(255).to(dev)[None, None]
        if np.random.rand() < 0.5:
            x, g = x.flip(-1), g.flip(-1)
        neg_case = args.neg > 0 and np.random.rand() < args.neg      # try to cut a window with no subject pixels (empty target)
        for _try in range(30 if neg_case else 1):
            s = np.random.uniform(0.3, 0.7) if neg_case else np.random.uniform(0.55, 1.0); ar = np.random.uniform(0.85, 1.15)
            ch, cw = min(1.0, s * ar), min(1.0, s / ar)
            y0, x0 = np.random.uniform(0, 1 - ch), np.random.uniform(0, 1 - cw)
            gb = g[..., int(y0 * TH):int((y0 + ch) * TH), int(x0 * TW):int((x0 + cw) * TW)]
            if not neg_case or gb.mean() < 0.002: break
        xb = x[..., int(y0 * H):int((y0 + ch) * H), int(x0 * W):int((x0 + cw) * W)]
        x = F.interpolate(xb, size=(H, W), mode="bilinear", align_corners=False)
        g = F.interpolate(gb, size=(TH, TW), mode="bilinear", align_corners=False)
        # colour jitter
        x = x * np.random.uniform(0.75, 1.25)
        m = x.mean(); x = (x - m) * np.random.uniform(0.75, 1.25) + m
        gray = x.mean(1, keepdim=True); x = (x - gray) * np.random.uniform(0.7, 1.3) + gray
        x = x * torch.tensor(np.random.uniform(0.9, 1.1, 3), dtype=torch.float32, device=dev).view(1, 3, 1, 1)
        x = x.clamp(0, 1)
        if args.aug == "underwater": x = underwater_aug(x)
        xs.append(x); gs.append(g[0, 0])
    return normalise(torch.cat(xs)), torch.stack(gs)


@torch.no_grad()
def predict(ids, bs=8):
    bb.eval(); head.eval()
    out = []
    for i in range(0, len(ids), bs):
        x = torch.stack([torch.from_numpy(imgs[j]).permute(2, 0, 1).float().div(255) for j in ids[i:i + bs]]).to(dev)
        out.append(torch.sigmoid(up(forward(normalise(x)))).cpu())
    return torch.cat(out)


def val_score():
    ids = idx["val"]; p = predict(ids)
    sc = {"usod": [], "suim": []}
    for k, i in enumerate(ids):
        ds = recs[i]["ds"]
        sc[ds].append(mask_metrics(p[k, cid(ds)].numpy() > 0.5, gts[i] > 127)["iou"])
    return (float(np.mean(sc["usod"])) if sc["usod"] else 0.0), (float(np.mean(sc["suim"])) if sc["suim"] else 0.0)


params = [dict(params=[p for p in bb.parameters() if p.requires_grad], lr=1e-4),
          dict(params=head.parameters(), lr=3e-4)]
opt = torch.optim.AdamW(params, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 3e-4], total_steps=steps, pct_start=0.05)
BS = 4
log, best = [], (-1, None)
t0 = time.time()
for step in range(1, steps + 1):
    bb.train(); head.train()
    ids = list(np.random.choice(tr_usod if SRC == "usod" else tr_suim, BS)) if SRC else list(np.random.choice(tr_usod, BS // 2)) + list(np.random.choice(tr_suim, BS // 2))
    x, g = make_batch(ids)
    c = torch.tensor([cid(recs[i]["ds"]) for i in ids], device=dev)
    loss = seg_loss(forward(x), g, c, size_weight(g) if args.loss == "sizeweight" else None)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for grp in params for p in grp["params"]], 1.0)
    opt.step(); sched.step()
    if step % 250 == 0 or step == steps:
        u, s = val_score()
        log.append(dict(step=step, loss=float(loss), val_iou_usod=u, val_iou_suim=s, secs=time.time() - t0))
        print(log[-1], flush=True)
        sel = u if SRC == "usod" else s if SRC == "suim" else (u + s) / 2
        if sel > best[0]:
            best = (sel, dict(lora={k: v.detach().cpu().clone() for k, v in bb.state_dict().items() if "lora" in k},
                                      head={k: v.cpu().clone() for k, v in head.state_dict().items()}, step=step))

res_dir = (ROOT / "results" / ("subject_v3" if args.label == "subject2" else "subject_v2") / TAG / f"stage_C{SD}") if SUBJ else (ROOT / "results" / "cross_dataset" / args.tag / "stage_C") if SRC else RES / "stage_C"
res_dir.mkdir(parents=True, exist_ok=True)
with open(res_dir / "training_log.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
sd = bb.state_dict(); sd.update(best[1]["lora"]); bb.load_state_dict(sd); head.load_state_dict(best[1]["head"])
print("best step", best[1]["step"], best[0])
ev_ids = idx["tgt"] if SRC else [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
vp_ids = idx["val"] if SRC else ev_ids
VP = predict(vp_ids)           # thresholds on source-val only
vmap = {i: k for k, i in enumerate(vp_ids)}
thr = {}
groups = {"all": ["usod", "suim"] if not SRC else [SRC]} if SUBJ else {ds: [ds] for ds in ((SRC,) if SRC else ("usod", "suim"))}
for gname, dss in groups.items():
    vids = [i for i in idx["val"] if recs[i]["ds"] in dss]
    ts = np.arange(0.2, 0.81, 0.05)
    sc = [np.mean([mask_metrics(VP[vmap[i], cid(recs[i]["ds"])].numpy() > t, gts[i] > 127)["iou"] for i in vids]) for t in ts]
    thr[gname] = float(ts[int(np.argmax(sc))])
name = f"v2_{TAG}_C{SD}" if SUBJ else f"cross_{args.tag}_C" if SRC else "stage_C"
OUT.joinpath("ckpt").mkdir(exist_ok=True)
torch.save(dict(lora=best[1]["lora"], head=best[1]["head"], thr=thr, head_cfg=ck["cfg"], step=best[1]["step"]), OUT / "ckpt" / f"{name}.pt")
json.dump(dict(steps=steps, best_step=best[1]["step"], val_score=best[0], thr=thr, lora="r16 qkv+proj", train=SRC, label=args.label, seed=args.seed, aug=args.aug, neg=args.neg), open(res_dir / "config.json", "w"), indent=1)
if SRC:
    P = predict(ev_ids)[:, cid(SRC)]
    np.save(OUT / "preds" / f"{name}.npy", P.numpy().astype(np.float16))
    np.save(OUT / "preds" / (f"v2_{TAG}_index.npy" if SUBJ else f"cross_{args.tag}_index.npy"), np.array(ev_ids))
else:
    P = VP
    np.save(pred_path(name if SUBJ else "stage_C", "evalset"), P.numpy().astype(np.float16))
print("saved", P.shape, thr)
