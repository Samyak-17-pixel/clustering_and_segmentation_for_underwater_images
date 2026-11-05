"""Shared pieces of the Gaussian-mixture study (Sec. 27): image sets, DINOv2 dense features (stride 3), PCA, GMM fitting, masks.

Pipeline per image (same as the k-means study): dense DINOv2 features (588 x 448 input, stride 3, 145 x 192 patches x 768) ->
PCA -> Gaussian mixture -> hard labels (argmax of the responsibilities) -> border rule picks the foreground components -> mask upsampled to the GT size.
"""
import sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/kmeans", "src/common", "src/supervised", "src/dendrogram"):
    sys.path.insert(0, str(ROOT / p))
import dinov2_cluster_union_stride as KM          # noqa: E402
from dinov2_seg import load_model, metrics        # noqa: E402
from dinov2_cluster_union import subsets          # noqa: E402
from dendro_cluster import silhouette_gpu         # noqa: E402
from gt_utils import load_gt                      # noqa: E402
from common import build_records                  # noqa: E402

SIZE, STRIDE = (588, 448), 3
RING = round(KM.PATCH / STRIDE)                   # border ring width in grid cells (5)
FIT_N = 3000                                      # patches a per-image GMM is fitted on
D_MAX = 64
KS = range(2, 7)
DEV = "cuda"
OUT = ROOT / "outputs" / "gmm"
RES = ROOT / "results" / "gmm"
USOD, SUIM_TR, SUIM_TE = ROOT / "data" / "USOD", ROOT / "data" / "SUIM" / "train_val", ROOT / "data" / "SUIM" / "TEST"


def image_sets():
    """name -> dict(data dir, stems). Splits from src/supervised/common.build_records (seed 0); SUIM dev / fit subsets are seeded random draws from the training split."""
    recs = build_records()
    usod = {s: [r["stem"] for r in recs if r["ds"] == "usod" and r["split"] == s] for s in ("train", "val", "test")}
    suim_train = sorted(r["stem"] for r in recs if r["ds"] == "suim" and r["split"] == "train")
    suim_test = sorted(r["stem"] for r in recs if r["ds"] == "suim" and r["split"] == "test")
    rng = np.random.RandomState(11)
    perm = rng.permutation(len(suim_train))
    dev = [suim_train[i] for i in perm[:160]]
    fit = [suim_train[i] for i in perm[160:460]]
    assert not set(dev + fit) & set(suim_test) and not set(usod["train"]) & set(usod["test"])
    return dict(usod_train=(USOD, usod["train"]), usod_test=(USOD, usod["test"]), usod_all=(USOD, sorted(usod["train"] + usod["val"] + usod["test"])),
                suim_dev=(SUIM_TR, dev), suim_fit=(SUIM_TR, fit), suim_test=(SUIM_TE, suim_test)), usod


def dense(model, path):
    """(Hd, Wd, 768) float32 dense features, as in the k-means study."""
    img = Image.open(path).convert("RGB")
    d, _ = KM.dense_features(model, img, SIZE, STRIDE, DEV)
    return d


@torch.no_grad()
def pca64(X, q=D_MAX):
    """Per-image PCA on the 768-D patches (GPU): returns Z [N, q] (components ordered by variance, so Z[:, :d] is the d-dim PCA)."""
    Xt = torch.from_numpy(X).to(DEV); mu = Xt.mean(0, keepdim=True); Xc = Xt - mu
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False, niter=3)
    return (Xc @ V).cpu().numpy().astype(np.float32)


def fit_gmm(Zs, K, cov, seed=0):
    """Full covariance is 30-50x slower than diagonal, so it uses one initialisation and fewer EM iterations."""
    full = cov == "full"
    return GaussianMixture(n_components=K, covariance_type=cov, reg_covar=1e-4, n_init=1 if full else 2, max_iter=60 if full else 100,
                           init_params="k-means++", random_state=seed).fit(Zs)


def sample_idx(N, n=FIT_N, seed=0):
    return np.arange(N) if N <= n else np.sort(np.random.RandomState(seed).choice(N, n, replace=False))


def labels_from_resp(resp, grid, smooth):
    """resp [N, K] -> hard label grid [Hd, Wd]; optional 3 x 3 box smoothing of the responsibilities first."""
    R = resp.reshape(*grid, -1)
    if smooth: R = uniform_filter(R, size=(3, 3, 1), mode="nearest")
    return R.argmax(-1)


def mask_from_labels(lab, K, gt_shape):
    fg = KM.border_rule_union(lab, K, RING)
    m = np.isin(lab, fg)
    return KM.upsample_maps(m[None].astype(np.float32), gt_shape, STRIDE, SIZE)[0] > 0.5, fg


def mask_from_prob(prob_grid, gt_shape):
    return KM.upsample_maps(prob_grid[None].astype(np.float32), gt_shape, STRIDE, SIZE)[0]


def iou(mask, gt):
    u = (mask | gt).sum()
    return float((mask & gt).sum() / u) if u else 1.0


def silhouette(Zs, lab, K):
    D = torch.cdist(torch.from_numpy(Zs).to(DEV), torch.from_numpy(Zs).to(DEV))
    return silhouette_gpu(D, torch.from_numpy(lab).long().to(DEV), K)


def per_image_gmm(Z, grid, cfg, gt_shape, return_all=False):
    """Per-image GMM with a configuration cfg = dict(cov, d, krule in {'bic','sil'}, smooth in {0,1}).
    Returns mask (bool, gt_shape), K, and (if return_all) the foreground probability grid and the labels."""
    d = cfg["d"]; Zd = Z[:, :d]; idx = sample_idx(len(Zd)); Zs = Zd[idx]
    fits, bic, sil = {}, {}, {}
    for K in KS:
        gm = fit_gmm(Zs, K, cfg["cov"]); fits[K] = gm; bic[K] = gm.bic(Zs)
        if cfg["krule"] == "sil":
            lab_s = gm.predict(Zs)
            sil[K] = silhouette(Zs, lab_s, K) if len(np.unique(lab_s)) > 1 else -1.0
    K = min(KS, key=bic.get) if cfg["krule"] == "bic" else max(KS, key=sil.get)
    resp = fits[K].predict_proba(Zd); lab = labels_from_resp(resp, grid, cfg["smooth"])
    mask, fg = mask_from_labels(lab, K, gt_shape)
    if not return_all: return mask, K
    R = resp.reshape(*grid, K)
    prob = R[..., list(fg)].sum(-1)
    return mask, K, mask_from_prob(prob, gt_shape), lab, fg


# ----------------------------------------------------------------------------- dataset-level GMM
import pickle


def ring_flags(grid):
    gh, gw = grid; y, x = np.divmod(np.arange(gh * gw), gw)
    return (y < RING) | (y >= gh - RING) | (x < RING) | (x >= gw - RING)


def load_global(src):
    return pickle.load(open(OUT / f"global_{src}.pkl", "rb"))


@torch.no_grad()
def global_prob(X, grid, model):
    """Foreground probability grid [Hd, Wd] of one image under a dataset-level GMM (global PCA -> responsibilities -> sum over foreground components)."""
    Z = ((torch.from_numpy(X).to(DEV) - torch.from_numpy(model["mean"]).to(DEV)) @ torch.from_numpy(model["V"]).to(DEV)).cpu().numpy()
    resp = model["gm"].predict_proba(Z)
    return resp[:, model["fg"]].sum(1).reshape(grid)
