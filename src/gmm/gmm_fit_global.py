"""Dataset-level GMM (Sec. 27): fit ONE mixture on patches pooled from the source dataset's training images (no labels), then label its components
foreground / background with the border rule aggregated over those images.

  python gmm_fit_global.py usod   # 160 USOD training images
  python gmm_fit_global.py suim   # 300 random SUIM training images (suim_fit)
Steps: sample 300 patches per image (uniform, with a flag for the border ring) -> global PCA (64-D) -> GMM for K in {8, 16, 32} x covariance {diag, full} fitted on 90% of the images,
chosen by BIC on the held-out 10% of images (-2 * held-out log-likelihood + parameters * log n) -> component c is background if its patches are in the border ring more often than the ring's share of the image.
"""
import sys, time
import numpy as np, torch, pandas as pd
from gmm_common import *

SRC = sys.argv[1]
PER_IMG = 300


def main():
    sets, _ = image_sets(); data, stems = sets["usod_train"] if SRC == "usod" else sets["suim_fit"]
    OUT.mkdir(parents=True, exist_ok=True); cache = OUT / f"global_{SRC}_samples.npz"
    if cache.exists():
        z = np.load(cache); P, ring, img_id = z["P"].astype(np.float32), z["ring"], z["img"]
    else:
        model = load_model("dinov2_vitb14", DEV); P, ring, img_id = [], [], []
        for i, stem in enumerate(stems):
            X = dense(model, data / "images" / f"{stem}.jpg"); grid = X.shape[:2]; X = X.reshape(-1, X.shape[-1]); idx = sample_idx(len(X), PER_IMG, seed=i)
            P.append(X[idx].astype(np.float16)); ring.append(ring_flags(grid)[idx]); img_id.append(np.full(len(idx), i))
            if (i + 1) % 50 == 0: print("sampled", i + 1, flush=True)
        P, ring, img_id = np.concatenate(P), np.concatenate(ring), np.concatenate(img_id)
        np.savez_compressed(cache, P=P, ring=ring, img=img_id); P = P.astype(np.float32)
    held = np.isin(img_id, np.arange(0, len(stems), 10))               # every 10th image is held out
    Pt = torch.from_numpy(P).to(DEV); mean = Pt.mean(0, keepdim=True); _, _, V = torch.pca_lowrank(Pt[~held] - mean, q=D_MAX, center=False, niter=3)
    mean, V = mean.cpu().numpy(), V.cpu().numpy(); Z = ((P - mean) @ V).astype(np.float32)
    rows, best = [], None
    for K in (8, 16, 32):
        for cov in ("diag", "full"):
            t0 = time.time(); gm = GaussianMixture(n_components=K, covariance_type=cov, reg_covar=1e-4, n_init=1, max_iter=100, init_params="k-means++", random_state=0).fit(Z[~held])
            n = held.sum(); bic = -2 * gm.score(Z[held]) * n + gm._n_parameters() * np.log(n)
            rows.append(dict(K=K, cov=cov, heldout_bic=bic, train_ll=gm.score(Z[~held]), secs=time.time() - t0)); print(rows[-1], flush=True)
            if best is None or bic < best[0]: best = (bic, K, cov, gm)
    _, K, cov, gm = best
    lab = gm.predict(Z); rho = ring.mean()
    share = np.array([ring[lab == c].mean() if (lab == c).any() else 1.0 for c in range(K)]); wts = np.array([(lab == c).mean() for c in range(K)])
    fg = [c for c in range(K) if share[c] < rho]
    if not fg: fg = [int(share.argmin())]
    if len(fg) == K: fg = [int(share.argmin())]
    pickle.dump(dict(mean=mean, V=V, gm=gm, fg=fg, K=K, cov=cov, share=share, weight=wts, rho=float(rho)), open(OUT / f"global_{SRC}.pkl", "wb"))
    pd.DataFrame(rows).to_csv(RES / f"global_{SRC}_selection.csv", index=False)
    md = [f"# Dataset-level GMM fitted on {SRC} training images\n", f"{len(stems)} images, {PER_IMG} patches each ({len(P):,} patches), global PCA 64-D; chosen K = {K}, covariance = {cov} (held-out BIC). Border-ring share of the image: {rho:.3f}.\n",
          "| component | weight | share of its patches in the border ring | role |", "|---|---|---|---|"]
    for c in np.argsort(-wts): md.append(f"| {c} | {wts[c]:.3f} | {share[c]:.3f} | {'foreground' if c in fg else 'background'} |")
    md += ["", "| K | covariance | held-out BIC | train log-likelihood | fit time (s) |", "|---|---|---|---|---|"] + [f"| {r['K']} | {r['cov']} | {r['heldout_bic']:.0f} | {r['train_ll']:.2f} | {r['secs']:.0f} |" for r in rows]
    (RES / f"global_{SRC}.md").write_text("\n".join(md)); print("\n".join(md[:8]))


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True); main()
