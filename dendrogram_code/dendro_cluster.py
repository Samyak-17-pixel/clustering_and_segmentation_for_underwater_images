"""Hierarchical (dendrogram) clustering of DINOv2 patch features and label-free foreground selection rules.

A tree is built per image over its patches. Every rule turns the tree into a boolean foreground mask on the
patch grid without using the ground truth; the GT is only used for the oracle and for the metrics.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import leaves_list, linkage as sp_linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph

PATCH = 14
NSUB = 3000                    # trees without spatial constraint are built on at most this many patches
KS = (2, 3, 4, 5, 6)
LINKAGES = ("ward", "average", "complete", "single")
RULES = ("sil_border", "gap_border", "subtree", "score")

# A-priori constants of the tree-walking rules (never tuned on the ground truth).
MIN_NODE = 0.02                # nodes smaller than this fraction of the patches are not considered on their own
SUBTREE_FG, SUBTREE_WATER = 0.5, 1.5   # border share relative to the global border share
SCORE_W = dict(border=0.35, centre=0.25, compact=0.20, size=0.20)
SCORE_KEEP = 0.9               # also keep non-overlapping nodes scoring >= this fraction of the best node


def metric_for(linkage, feat):
    """Ward needs Euclidean; the other linkages use cosine on the 768-D features and Euclidean on PCA-RGB."""
    return "cosine" if (linkage != "ward" and feat == "768") else "euclidean"


def subsample_index(n, seed=0):
    if n <= NSUB:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, NSUB, replace=False))


def build_tree(X, grid, linkage, spatial, feat):
    """Linkage matrix (scipy format) of the patches in X. `spatial` merges only grid neighbours."""
    metric = metric_for(linkage, feat)
    if not spatial:
        return sp_linkage(X, method=linkage, metric=metric)
    n = len(X)
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_full_tree=True,
                                    linkage=linkage, metric=metric, connectivity=grid_to_graph(*grid))
    model.fit(X)
    children = model.children_.astype(int)
    count = np.ones(2 * n - 1)
    for i, (a, b) in enumerate(children):
        count[n + i] = count[a] + count[b]
    return np.column_stack([children, model.distances_, count[n:]])


def node_stats(Z, py, px, ring):
    """Per-node arrays over the 2n-1 nodes: size, ring size, coordinate sums, bbox, and the leaf-order range."""
    n = len(py)
    m = 2 * n - 1
    cnt, rng_ = np.zeros(m), np.zeros(m)
    sy, sx = np.zeros(m), np.zeros(m)
    y0, y1 = np.full(m, np.inf), np.full(m, -np.inf)
    x0, x1 = np.full(m, np.inf), np.full(m, -np.inf)
    lo, hi = np.zeros(m, int), np.zeros(m, int)
    order = leaves_list(Z)
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    cnt[:n], rng_[:n], sy[:n], sx[:n] = 1, ring, py, px
    y0[:n] = y1[:n] = py
    x0[:n] = x1[:n] = px
    lo[:n], hi[:n] = pos, pos + 1
    for i, (a, b) in enumerate(Z[:, :2].astype(int)):
        v = n + i
        cnt[v], rng_[v] = cnt[a] + cnt[b], rng_[a] + rng_[b]
        sy[v], sx[v] = sy[a] + sy[b], sx[a] + sx[b]
        y0[v], y1[v] = min(y0[a], y0[b]), max(y1[a], y1[b])
        x0[v], x1[v] = min(x0[a], x0[b]), max(x1[a], x1[b])
        lo[v], hi[v] = min(lo[a], lo[b]), max(hi[a], hi[b])
    return dict(n=n, cnt=cnt, ring=rng_, sy=sy, sx=sx, y0=y0, y1=y1, x0=x0, x1=x1, lo=lo, hi=hi, order=order)


def cut_labels(Z, st, ks=KS):
    """(n, len(ks)) cluster labels from undoing the last k-1 merges in creation order.

    Unlike scipy's cut_tree this stays correct when merge heights are not monotone, which happens for the
    connectivity-constrained trees from scikit-learn."""
    n = st["n"]
    nodes, out = [2 * n - 2], np.zeros((n, len(ks)), int)
    for j in range(max(ks) - 1):
        v = nodes.pop(nodes.index(n + len(Z) - 1 - j))
        nodes.extend(Z[v - n, :2].astype(int))
        k = j + 2
        if k in ks:
            for c, u in enumerate(nodes):
                out[st["order"][st["lo"][u]:st["hi"][u]], ks.index(k)] = c
    return out


def gap_k(Z, ks=KS):
    """k with the largest jump between the heights of the merges that separate k-1 from k and k from k+1 clusters
    (top merges taken in creation order)."""
    d = Z[::-1, 2]
    return max(ks, key=lambda k: d[k - 2] - d[k - 1])


def _fallback(labels2, ring):
    """Two-cluster cut: the cluster with the smaller border share is foreground."""
    share = [ring[labels2 == c].mean() if (labels2 == c).any() else 1.0 for c in (0, 1)]
    return labels2 == int(np.argmin(share))


def rule_subtree(st, Z, labels2, ring_leaf):
    """Top-down walk: nodes with (almost) no border patches are foreground, border-heavy nodes are water,
    mixed nodes are split. Tiny nodes are decided by the plain border-share test."""
    n, cnt, rng_ = st["n"], st["cnt"], st["ring"]
    ref = rng_[2 * n - 2] / cnt[2 * n - 2]
    fg_leaf = np.zeros(n, bool)
    stack = [2 * n - 2]
    while stack:
        v = stack.pop()
        share = rng_[v] / cnt[v]
        small = cnt[v] < MIN_NODE * n
        root = v == 2 * n - 2
        if not root and (v < n or small):
            decided = share < ref
        elif not root and share < SUBTREE_FG * ref:
            decided = True
        elif not root and share > SUBTREE_WATER * ref:
            decided = False
        else:
            stack.extend(Z[v - n, :2].astype(int))
            continue
        if decided:
            fg_leaf[st["order"][st["lo"][v]:st["hi"][v]]] = True
    if fg_leaf.all() or not fg_leaf.any():
        return _fallback(labels2, ring_leaf)
    return fg_leaf


def rule_score(st, grid_hw, labels2, ring_leaf):
    """Score every node (low border share, near the centre, compact bbox, plausible size), keep the best ones."""
    n, gh, gw = st["n"], *grid_hw
    cnt = st["cnt"]
    ref = st["ring"][2 * n - 2] / cnt[2 * n - 2]
    frac = cnt / n
    share = st["ring"] / cnt
    border = 1 - np.minimum(share / max(ref, 1e-9), 2) / 2
    cy, cx = st["sy"] / cnt, st["sx"] / cnt
    centre = 1 - np.hypot((cy - (gh - 1) / 2) / (gh / 2), (cx - (gw - 1) / 2) / (gw / 2)) / np.sqrt(2)
    bbox = (st["y1"] - st["y0"] + 1) * (st["x1"] - st["x0"] + 1)
    compact = np.minimum(frac * gh * gw / bbox, 1)
    size = np.interp(frac, [0.01, 0.05, 0.5, 0.9], [0, 1, 1, 0])
    score = (SCORE_W["border"] * border + SCORE_W["centre"] * centre
             + SCORE_W["compact"] * compact + SCORE_W["size"] * size)
    score[(frac < MIN_NODE) | (np.arange(len(cnt)) == 2 * n - 2)] = -np.inf
    fg_leaf = np.zeros(n, bool)
    taken = []
    for v in np.argsort(-score):
        if not np.isfinite(score[v]) or score[v] < SCORE_KEEP * score.max():
            break
        if any(st["lo"][v] < h and l < st["hi"][v] for l, h in taken):
            continue
        taken.append((st["lo"][v], st["hi"][v]))
        fg_leaf[st["order"][st["lo"][v]:st["hi"][v]]] = True
    if fg_leaf.all() or not fg_leaf.any():
        return _fallback(labels2, ring_leaf)
    return fg_leaf


def silhouette_gpu(D, lab, k):
    """Mean silhouette from a precomputed (n, n) distance matrix and integer labels in [0, k)."""
    Y = F.one_hot(lab, k).float()
    cnt = Y.sum(0)
    present = cnt > 0
    if int(present.sum()) < 2:
        return -1.0
    S = D @ Y
    own = cnt[lab]
    a = S.gather(1, lab[:, None])[:, 0] / (own - 1).clamp(min=1)
    other = S / cnt.clamp(min=1)
    other[:, ~present] = float("inf")
    other.scatter_(1, lab[:, None], float("inf"))
    b = other.min(1).values
    s = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s[own == 1] = 0
    return float(s.mean())


def make_sampling_grid(out_hw, size, grid_hw, stride, device):
    """Same bilinear sampling as dinov2_cluster_union_stride.upsample_maps, but built once per image."""
    W, H = size
    oh, ow = out_hw
    gh, gw = grid_hw

    def coords(n_out, length, n_grid):
        m = np.clip(((np.arange(n_out) + 0.5) * length / n_out - PATCH / 2) / stride, 0, n_grid - 1)
        return 2 * m / max(n_grid - 1, 1) - 1

    gy, gx = np.meshgrid(coords(oh, H, gh), coords(ow, W, gw), indexing="ij")
    return torch.from_numpy(np.stack([gx, gy], -1).astype(np.float32))[None].to(device)


def upsample_gpu(maps, sampling_grid):
    """(c, gh, gw) float tensor -> (c, oh, ow)."""
    return F.grid_sample(maps[None].float(), sampling_grid, mode="bilinear", padding_mode="border",
                         align_corners=True)[0]


def metrics_batch(prob, gt):
    """IoU / F(beta^2=0.3) / MAE for each row of prob (m, P) against the boolean gt (P,); same as dinov2_seg.metrics."""
    pred = prob > 0.5
    inter = (pred & gt).sum(1).float()
    union = (pred | gt).sum(1).float()
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    prec = inter / pred.sum(1).clamp(min=1)
    rec = inter / gt.sum().clamp(min=1)
    f = 1.3 * prec * rec / (0.3 * prec + rec + 1e-8)
    mae = (prob - gt.float()).abs().mean(1)
    return iou.cpu().numpy(), f.cpu().numpy(), mae.cpu().numpy()


def subset_matrix(k, device):
    """(2^k - 2, k) 0/1 matrix of all non-empty proper cluster subsets."""
    rows = [[(s >> c) & 1 for c in range(k)] for s in range(1, 2 ** k - 1)]
    return torch.tensor(rows, dtype=torch.float32, device=device)
