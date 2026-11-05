"""Underwater degradation augmentation for Stage-C training (Sec. 26).

Applied to an image tensor x [1,3,H,W] in [0,1]. Ranges were fixed a priori from the brightness / blue-green statistics of the tank frames
(metadata.csv: mean grey 30-157, blue-green ratio 0.9-1.8) and are never tuned on them. Components, each applied with its own probability:
haze (blend towards a water colour), darkening, colour cast (less red), contrast loss, blur.
"""
import numpy as np
import torch
import torch.nn.functional as F

WATER = np.array([[0.10, 0.22, 0.15], [0.05, 0.20, 0.38], [0.22, 0.30, 0.28], [0.12, 0.18, 0.20]], np.float32)   # green, blue, grey-green, dark teal


def _blur(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    w = torch.exp(-(t ** 2) / (2 * sigma ** 2)); w = w / w.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def underwater_aug(x, rng=np.random):
    dev = x.device
    if rng.rand() < 0.6:                                   # haze
        t = rng.uniform(0.05, 0.6); c = torch.tensor(WATER[rng.randint(len(WATER))], device=dev).view(1, 3, 1, 1)
        x = (1 - t) * x + t * c
    if rng.rand() < 0.5:                                   # darker scene
        x = x * rng.uniform(0.4, 1.0)
    if rng.rand() < 0.5:                                   # colour cast: red is absorbed first
        x = x * torch.tensor([rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.1), rng.uniform(0.8, 1.1)], device=dev, dtype=x.dtype).view(1, 3, 1, 1)
    if rng.rand() < 0.5:                                   # contrast loss
        m = x.mean(); x = (x - m) * rng.uniform(0.5, 1.0) + m
    if rng.rand() < 0.3:                                   # blur (turbidity / focus)
        x = _blur(x, rng.uniform(0.5, 2.0))
    return x.clamp(0, 1)


if __name__ == "__main__":     # preview: one image under 6 random degradations -> results/subject_v3/aug_preview.png
    import sys
    from pathlib import Path
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    root = Path(__file__).resolve().parents[2]
    img = Image.open(next((root / "data" / "SUIM" / "TEST" / "images").glob("*.jpg"))).convert("RGB").resize((616, 462))
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)[None].float() / 255
    rng = np.random.RandomState(1)
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.4))
    axs[0, 0].imshow(img); axs[0, 0].set_title("original")
    for a in axs.ravel()[1:7]:
        a.imshow(underwater_aug(x, rng)[0].permute(1, 2, 0).numpy())
    axs.ravel()[7].imshow(np.zeros((8, 8, 3))); 
    for a in axs.ravel(): a.set_xticks([]); a.set_yticks([])
    out = root / "results" / "subject_v3"; out.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out / "aug_preview.png", dpi=90); print("saved")
