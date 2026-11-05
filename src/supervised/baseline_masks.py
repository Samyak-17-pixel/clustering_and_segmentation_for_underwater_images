"""Re-create the *masks* of the earlier training-free methods for chosen images (for side-by-side figures).

Earlier runs stored only per-image metrics (and oracle k-means masks), so:
  - k-means border rule : recomputed with the original functions (768-D, stride 3, k=2..6, silhouette k*, border rule)
  - dendrogram gap rule : unpacked from results/dendrogram/_cache/<tag>/<stem>_s3.npz (768|free|average|gap_border)
  - k-means oracle      : read from results/kmeans/<ds>/cluster_union_stride3_k6/masks_768/<stem>.png
Each mask is returned at the native GT resolution together with its IoU against the GT it was scored on.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "kmeans"))
sys.path.insert(0, str(ROOT / "src" / "common"))
import dinov2_cluster_union_stride as KM   # noqa: E402
from gt_utils import load_gt               # noqa: E402

SIZE, STRIDE = (588, 448), 3
_model = None


def _iou(pred, gt):
    u = (pred | gt).sum()
    return float((pred & gt).sum() / u) if u else 1.0


def kmeans_border(img_path, gt_shape):
    global _model
    if _model is None:
        _model = KM.load_model("dinov2_vitb14", "cuda")
    img = Image.open(img_path).convert("RGB")
    dense, _ = KM.dense_features(_model, img, SIZE, STRIDE, "cuda")
    grid = dense.shape[:2]
    X = dense.reshape(-1, dense.shape[-1])
    labels, sil = {}, {}
    for k in range(2, 7):
        lab = KM.cluster(X, k)
        labels[k] = lab.reshape(grid)
        sil[k] = float(silhouette_score(X, lab, sample_size=KM.SIL if len(X) > KM.SIL else None, random_state=0))
    ks = max(sil, key=sil.get)
    s = KM.border_rule_union(labels[ks], ks, round(KM.PATCH / STRIDE))
    prob = KM.upsample_maps(np.isin(labels[ks], s)[None], gt_shape, STRIDE, SIZE)[0]
    return prob > 0.5


def dendro_gap(ds_tag, stem, gt_shape):
    z = np.load(ROOT / "results" / "dendrogram" / "_cache" / ds_tag / f"{stem}_s{STRIDE}.npz")
    grid = tuple(z["grid"])
    m = np.unpackbits(z["768|free|average|gap_border"])[: grid[0] * grid[1]].reshape(grid)
    return KM.upsample_maps(m[None].astype(np.float32), gt_shape, STRIDE, SIZE)[0] > 0.5


def kmeans_oracle(ds_dir, stem):
    p = ROOT / "results" / "kmeans" / ds_dir / "cluster_union_stride3_k6" / "masks_768" / f"{stem}.png"
    return np.asarray(Image.open(p).convert("L")) > 127


def get_masks(items):
    """items: list of dict(ds='usod'|'suim', stem, img, data, gt_mode). Returns {stem: {name: (mask, iou)}}."""
    out = {}
    for it in items:
        gt = load_gt(it["data"], it["stem"], it["gt_mode"])
        tag = it["ds"]
        kdir = "usod" if it["ds"] == "usod" else ("suim_saliency" if it["gt_mode"] == "suim_saliency" else "suim")
        ms = {"k-means (border rule)": kmeans_border(it["img"], gt.shape),
              "dendrogram (gap rule)": dendro_gap(tag, it["stem"], gt.shape)}
        if it["gt_mode"] != "suim_subject":     # stored oracle masks exist only for the old GT definitions
            ms["k-means oracle (uses GT)"] = kmeans_oracle(kdir, it["stem"])
        out[it["stem"]] = {k: (m, _iou(m, gt)) for k, m in ms.items()}
    return out


if __name__ == "__main__":   # self-check against the stored per-image numbers
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from common import build_records
    recs = [r for r in build_records() if r["split"] == "test"]
    pick = [next(r for r in recs if r["ds"] == "usod"), next(r for r in recs if r["ds"] == "suim")]
    base = pd.read_csv(ROOT / "results/comparison/per_image.csv").set_index("image")
    for r, m in zip(pick, get_masks(pick).values()):
        row = base.loc[r["stem"]]
        print(r["stem"], {k: round(v[1], 3) for k, v in m.items()},
              "stored:", round(row["k-means, border rule (stride 3, k<=6)__iou"], 3), round(row["dendrogram, gap rule (best label-free)__iou"], 3),
              round(row["k-means ORACLE (uses GT)__iou"], 3))
