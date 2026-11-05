"""Stage 0: cache frozen DINOv2 ViT-B/14 patch features and low-res targets for every image.

outputs/features/feats.npy  float16 [N,768,33,44]   (final-layer normalised patch tokens)
outputs/features/gt.npy     uint8   [N,231,308]     (binary GT area-resized, x255; the channel is CH[ds])
Row order = build_records() order.
"""
import argparse, time
import numpy as np
import torch
from tqdm import tqdm
from common import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=1, help="n>1: average of the last n normalised layers -> feats_l<n>.npy (no gt rewrite)")
    NL = ap.parse_args().layers
    recs = build_records()
    d = OUT / "features"
    d.mkdir(parents=True, exist_ok=True)
    N = len(recs)
    feats = np.lib.format.open_memmap(d / ("feats.npy" if NL == 1 else f"feats_l{NL}.npy"), "w+", np.float16, (N, 768, GH, GW))
    gts = np.zeros((N, TH, TW), np.uint8)
    bb = load_backbone()
    t0 = time.time()
    B = 8
    for i in tqdm(range(0, N, B)):
        batch = recs[i:i + B]
        x = torch.stack([to_tensor(load_image(r["img"])) for r in batch]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if NL == 1:
                f = extract(bb, normalise(x))
            else:   # mean of the last NL layers (each layer-normed), same 768-D
                with torch.no_grad():
                    fs = bb.get_intermediate_layers(normalise(x), n=NL, reshape=True, norm=True)
                f = torch.stack(fs).mean(0)
        feats[i:i + len(batch)] = f.float().cpu().numpy().astype(np.float16)
        if NL == 1:
            for j, r in enumerate(batch):
                gts[i + j] = (gt_soft(r) * 255).round().astype(np.uint8)
    feats.flush()
    if NL == 1:
        np.save(d / "gt.npy", gts)
    print(f"cached {N} images in {time.time()-t0:.0f}s")
