"""Shared pieces for the supervised stages (A-D): data splits, loading, model, losses, metrics.

Two output channels everywhere: ch0 = USOD salient object, ch1 = SUIM "not water".
"""
import os, sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "common"))
from gt_utils import load_gt  # noqa: E402

OUT = ROOT / "outputs"
RES = ROOT / "results" / "supervised"
USOD_DIR = ROOT / "data" / "USOD"
SUIM_TR = ROOT / "data" / "SUIM" / "train_val"
SUIM_TE = ROOT / "data" / "SUIM" / "TEST"

H, W = 462, 616            # network input (33 x 44 patches of 14 px)
GH, GW = H // 14, W // 14
TH, TW = H // 2, W // 2    # 231 x 308 = 7x the patch grid: training-target / stored-prediction resolution
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CH = {"usod": 0, "suim": 1}
SEED = 0


def build_records():
    """All images with a fixed seeded split. Record = dict(ds, split, stem, img, data, gt_mode)."""
    rng = np.random.RandomState(SEED)
    recs = []
    stems = sorted(p.stem for p in (USOD_DIR / "images").glob("*.jpg"))
    perm = rng.permutation(len(stems))
    split = {}
    for rank, i in enumerate(perm):
        split[stems[i]] = "test" if rank < 100 else "val" if rank < 140 else "train"   # 100 / 40 / 160
    for s in stems:
        recs.append(dict(ds="usod", split=split[s], stem=s, img=USOD_DIR / "images" / f"{s}.jpg",
                         data=USOD_DIR, gt_mode="usod"))
    stems = sorted(p.stem for p in (SUIM_TR / "images").glob("*.jpg"))
    perm = rng.permutation(len(stems))
    for rank, i in enumerate(perm):
        recs.append(dict(ds="suim", split="val" if rank < 150 else "train", stem=stems[i],
                         img=SUIM_TR / "images" / f"{stems[i]}.jpg", data=SUIM_TR, gt_mode="suim_nonwater"))
    for p in sorted((SUIM_TE / "images").glob("*.jpg")):
        recs.append(dict(ds="suim", split="test", stem=p.stem, img=p, data=SUIM_TE, gt_mode="suim_nonwater"))
    return recs


def subset(recs, ds=None, split=None):
    return [r for r in recs if (ds is None or r["ds"] == ds) and (split is None or r["split"] == split)]


def load_image(path, size=(W, H)):
    return Image.open(path).convert("RGB").resize(size, Image.BILINEAR)


def to_tensor(img):
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
    return x


def normalise(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def gt_soft(rec, mode=None, size=(TW, TH)):
    """Binary GT resized (area average) to the target resolution, float32 in [0,1]."""
    g = load_gt(rec["data"], rec["stem"], mode or rec["gt_mode"]).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(g).resize(size, Image.BOX), dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------- model
def load_backbone(device="cuda"):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", skip_validation=True)
    return m.eval().to(device)


@torch.no_grad()
def extract(backbone, x):
    """x: normalised [B,3,H,W] -> [B,768,GH,GW] final-layer patch tokens."""
    t = backbone.forward_features(x)["x_norm_patchtokens"]
    return t.transpose(1, 2).reshape(x.shape[0], -1, x.shape[2] // 14, x.shape[3] // 14)


class Head(nn.Module):
    """kind='linear': one 1x1 conv.  kind='conv': 1x1 reduce + residual dilated 3x3 blocks + 1x1 classifier."""

    def __init__(self, kind="linear", dim=768, width=256, depth=3, out=2, drop=0.1):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.norm = nn.BatchNorm2d(dim, affine=False)
            self.fc = nn.Conv2d(dim, out, 1)
        else:
            self.inp = nn.Sequential(nn.BatchNorm2d(dim, affine=False), nn.Dropout2d(drop),
                                     nn.Conv2d(dim, width, 1), nn.BatchNorm2d(width), nn.GELU())
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.BatchNorm2d(width), nn.GELU())
                for d in [1, 2, 4, 1][:depth]])
            self.fc = nn.Conv2d(width, out, 1)

    def forward(self, f):
        if self.kind == "linear":
            return self.fc(self.norm(f))
        x = self.inp(f)
        for b in self.blocks:
            x = x + b(x)
        return self.fc(x)


def up(logits, size=(TH, TW)):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def size_weight(target):
    """Per-image weight for the size-aware loss: small subjects count more, very large ones slightly less. [B,TH,TW] -> [B]."""
    area = target.flatten(1).mean(1).clamp(min=1e-3)
    return (0.15 / area).sqrt().clamp(0.5, 3.0)


def seg_loss(logits, target, ch, weight=None):
    """logits [B,2,gh,gw]; target [B,TH,TW] soft in [0,1]; ch [B] long (which channel has a label).
    weight: optional [B] per-image weights (mean-normalised) applied to both terms."""
    lg = up(logits.float())[torch.arange(len(ch)), ch]
    bce = F.binary_cross_entropy_with_logits(lg, target, reduction="none").flatten(1).mean(1)
    p = torch.sigmoid(lg)
    inter = (p * target).flatten(1).sum(1)
    dice = 1 - (2 * inter + 1) / (p.flatten(1).sum(1) + target.flatten(1).sum(1) + 1)
    if weight is None:
        return bce.mean() + dice.mean()
    w = weight / weight.mean()
    return (w * bce).mean() + (w * dice).mean()


# ----------------------------------------------------------------------------- metrics
def mask_metrics(pred, gt):
    """pred, gt: bool arrays, same shape. Returns IoU, F1, F-beta(0.3), MAE."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    iou = inter / union if union else 1.0
    prec = inter / pred.sum() if pred.sum() else (1.0 if not gt.sum() else 0.0)
    rec = inter / gt.sum() if gt.sum() else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fb = 1.3 * prec * rec / (0.3 * prec + rec) if (0.3 * prec + rec) else 0.0
    return dict(iou=float(iou), f1=float(f1), fbeta=float(fb), mae=float((pred != gt).mean()))


def prob_to_mask(prob, shape, thr=0.5):
    """prob: [h,w] float array -> bool mask at `shape` (bilinear upsample then threshold)."""
    t = torch.from_numpy(prob.astype(np.float32))[None, None]
    t = F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    return t > thr


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pred_path(stage, split):
    d = OUT / "preds"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}_{split}.npy"


def eval_records(recs):
    """Canonical ordering of the images whose predictions are stored (val + test)."""
    return [r for r in recs if r["split"] in ("val", "test")]
