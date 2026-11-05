"""Worked example for docs/README_supervised.md: one image through the pooled Stage-A linear probe, with the real numbers.

Image choice by rule: the test image of each dataset whose Stage-A IoU is closest to the median Stage-A IoU of that dataset's test set.
Writes results/doc_figure_sources/worked_example_<ds>.png (one image per file) and results/subject_v2/worked_example.md.
"""
import json
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

V2 = ROOT / "results" / "subject_v2"
FIG = ROOT / "results" / "doc_figure_sources"; FIG.mkdir(parents=True, exist_ok=True)
recs = build_records()
feats = np.load(OUT / "features" / "feats.npy", mmap_mode="r")
ck = torch.load(OUT / "ckpt" / "v2_pool_A.pt")
thr = ck["thr"]["all"]
head = Head(**{k: v for k, v in ck["cfg"].items() if k not in ("lr", "wd")}, out=1).eval(); head.load_state_dict(ck["state"])
md = ["# Worked example (pooled Stage-A linear probe)\n", f"threshold {thr:.2f} (chosen on validation)\n"]
for ds, csv in (("usod", "per_image_pool_usod.csv"), ("suim", "per_image_pool_suim.csv")):
    d = pd.read_csv(V2 / csv, dtype={"image": str})
    row = d.iloc[(d["model A"] - d["model A"].median()).abs().argsort().iloc[0]]
    i = int(row.rec); r = recs[i]
    f = torch.from_numpy(feats[i].astype(np.float32))[None]
    with torch.no_grad():
        z = head.norm(f)                                   # BatchNorm (running mean/var, no affine)
        logit = head.fc(z)[0, 0]                            # [33,44] = w . z + b per patch
        prob_hw = torch.sigmoid(up(head.fc(z)))[0, 0].numpy()
    gt = load_gt(r["data"], r["stem"], "suim_subject" if ds == "suim" else "usod")
    gtl = gt_soft(r, "suim_subject" if ds == "suim" else "usod") > 0.5
    pm = prob_to_mask(prob_hw, gt.shape, thr)
    inter, union = int((pm & gt).sum()), int((pm | gt).sum())
    pg = np.asarray(Image.fromarray(gtl.astype(np.uint8) * 255).resize((GW, GH), Image.BOX)) > 127
    lg = logit.numpy()
    py, px = np.unravel_index(np.argmax(np.where(pg, lg, -1e9)), lg.shape)     # most confident subject patch
    wy, wx = np.unravel_index(np.argmin(np.where(~pg, lg, 1e9)), lg.shape)      # most confident water patch
    w = head.fc.weight[0, :, 0, 0].detach().numpy(); b = float(head.fc.bias[0])
    def patch(y, x):
        zz = z[0, :, y, x].numpy()
        return dict(pos=(int(y), int(x)), feat_norm=float(np.linalg.norm(f[0, :, y, x].numpy())), logit=float(zz @ w + b), prob=float(1 / (1 + np.exp(-(zz @ w + b)))))
    ps, pw = patch(py, px), patch(wy, wx)
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.2))
    axs[0].imshow(Image.open(r["img"]).convert("RGB").resize(gt.shape[::-1])); axs[0].set_title(f"image ({r['stem']})", fontsize=9)
    im = axs[1].imshow(lg, cmap="coolwarm", vmin=-6, vmax=6); axs[1].set_title("patch scores w.x+b (33x44)", fontsize=9)
    axs[1].plot(px, py, "ko", ms=5); axs[1].plot(wx, wy, "ks", ms=5, mfc="none")
    axs[2].imshow(prob_hw, cmap="viridis", vmin=0, vmax=1); axs[2].set_title("probability, upsampled (231x308)", fontsize=9)
    axs[3].imshow(pm, cmap="gray"); axs[3].set_title(f"threshold {thr:.2f} -> mask", fontsize=9)
    axs[4].imshow(gt, cmap="gray"); axs[4].set_title("ground truth", fontsize=9)
    axs[3].text(.03, .04, f"IoU {inter/union:.2f}", transform=axs[3].transAxes, color="white", fontsize=10, bbox=dict(fc="black", alpha=.65, lw=0, pad=2))
    for a in axs: a.set_xticks([]); a.set_yticks([])
    plt.colorbar(im, ax=axs[1], fraction=0.046); plt.tight_layout(); plt.savefig(FIG / f"worked_example_{ds}.png", dpi=110); plt.close()
    md += [f"## {ds}: {r['stem']}", f"* Stage-A IoU {row['model A']:.3f} (median of the test set {d['model A'].median():.3f}); GT covers {gt.mean():.1%} of the image",
           f"* subject patch (row {ps['pos'][0]}, col {ps['pos'][1]}, black dot): ||feature|| = {ps['feat_norm']:.1f}, logit = {ps['logit']:.2f}, probability = {ps['prob']:.3f}",
           f"* water patch (row {pw['pos'][0]}, col {pw['pos'][1]}, hollow square): ||feature|| = {pw['feat_norm']:.1f}, logit = {pw['logit']:.2f}, probability = {pw['prob']:.3f}",
           f"* mask vs GT at native resolution ({gt.shape[1]}x{gt.shape[0]}): intersection {inter:,} px, union {union:,} px, IoU = {inter/union:.3f}",
           f"* patches above threshold: {(1/(1+np.exp(-lg)) > thr).sum()} of {lg.size}\n"]
(V2 / "worked_example.md").write_text("\n".join(md)); print("\n".join(md))
