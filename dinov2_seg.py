"""Training-free foreground segmentation with frozen DINOv2 patch features (PCA / k-means)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT = Path(__file__).parent
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PATCH = 14


def load_model(name, device):
    model = torch.hub.load("facebookresearch/dinov2", name, skip_validation=True)
    return model.eval().half().to(device)


@torch.no_grad()
def patch_features(model, img, size, device):
    w, h = size
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.BICUBIC))).permute(2, 0, 1)[None].float() / 255
    x = ((x - MEAN) / STD).half().to(device)
    feats = model.forward_features(x)["x_norm_patchtokens"][0].float().cpu().numpy()
    return feats.reshape(h // PATCH, w // PATCH, -1)


def border_share(mask):
    """Fraction of a boolean patch mask's pixels lying on the image border."""
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return (mask & border).sum() / max(mask.sum(), 1)


def pick_fg(mask):
    """Foreground = the side of the split that touches the image border least."""
    return mask if border_share(mask) <= border_share(~mask) else ~mask


def pca_masks(feats):
    gh, gw, c = feats.shape
    flat = feats.reshape(-1, c)
    comp = PCA(n_components=3).fit_transform(flat)
    mask = pick_fg(comp[:, 0].reshape(gh, gw) > 0)
    rgb = comp.reshape(gh, gw, 3)
    rgb = (rgb - rgb.min((0, 1))) / (np.ptp(rgb, axis=(0, 1)) + 1e-8)
    return mask, rgb


def kmeans_mask(feats):
    gh, gw, c = feats.shape
    labels = KMeans(n_clusters=2, n_init=4, random_state=0).fit_predict(feats.reshape(-1, c))
    return pick_fg(labels.reshape(gh, gw) == 1)


def upsample(mask, out_hw):
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def metrics(prob, gt):
    pred = prob > 0.5
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    iou = inter / union if union else 1.0
    prec = inter / max(pred.sum(), 1)
    rec = inter / max(gt.sum(), 1)
    f = 1.3 * prec * rec / (0.3 * prec + rec + 1e-8)
    mae = np.abs(prob - gt.astype(np.float32)).mean()
    return float(iou), float(f), float(mae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "USOD")
    ap.add_argument("--out", default=ROOT / "dinov2_results")
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=588)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    for sub in ("panels", "masks_pca", "masks_kmeans"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    paths = sorted((data / "images").glob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]

    rows = []
    for p in tqdm(paths):
        img = Image.open(p).convert("RGB")
        gt = np.asarray(Image.open(data / "masks" / f"{p.stem}.png").convert("L")) > 128
        feats = patch_features(model, img, (args.width, args.height), device)
        m_pca, rgb = pca_masks(feats)
        m_km = kmeans_mask(feats)

        probs = {"pca": upsample(m_pca, gt.shape), "kmeans": upsample(m_km, gt.shape)}
        row = {"image": p.name}
        for name, prob in probs.items():
            row.update(zip((f"{name}_iou", f"{name}_f", f"{name}_mae"), metrics(prob, gt)))
            Image.fromarray(((prob > 0.5) * 255).astype(np.uint8)).save(out / f"masks_{name}" / f"{p.stem}.png")
        rows.append(row)

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        panels = [(img, "image"), (rgb, "PCA RGB"),
                  (probs["pca"] > 0.5, f"PCA mask IoU={row['pca_iou']:.2f}"),
                  (probs["kmeans"] > 0.5, f"k-means IoU={row['kmeans_iou']:.2f}"), (gt, "GT")]
        for ax, (im, title) in zip(axs, panels):
            ax.imshow(im, cmap=None if im is img or im is rgb else "gray")
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "panels" / f"{p.stem}.png", dpi=70)
        plt.close(fig)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for name in ("pca", "kmeans"):
        print(name, {k: round(float(np.mean([r[f"{name}_{k}"] for r in rows])), 4) for k in ("iou", "f", "mae")})


if __name__ == "__main__":
    main()
