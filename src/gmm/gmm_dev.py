"""Development ablation of the per-image GMM on the SOURCE dataset's training images only (Sec. 27).

  python gmm_dev.py usod    # 160 USOD training images, GT = salient object
  python gmm_dev.py suim    # 160 random SUIM training images, GT = not-water
Grid (32 settings): (covariance, PCA dim) in {spherical, diag} x {16, 32, 64} + {full} x {16, 32}  x  K rule {BIC, silhouette}  x  smoothing {off, on}.
The setting with the best mean IoU of the border-rule mask is written to results/gmm/config_<source>.json; nothing is scored on test images here.
"""
import csv, json, sys, time
import numpy as np
from gmm_common import *

SRC = sys.argv[1]
COMBOS = [("spherical", 16), ("spherical", 32), ("spherical", 64), ("diag", 16), ("diag", 32), ("diag", 64), ("full", 16), ("full", 32)]


def dev_variants(Z, grid, gt):
    out = {}
    for cov, d in COMBOS:
        Zd = Z[:, :d]; Zs = Zd[sample_idx(len(Zd))]; fits, bic, sil = {}, {}, {}
        for K in KS:
            gm = fit_gmm(Zs, K, cov); fits[K] = gm; bic[K] = gm.bic(Zs)
            lab_s = gm.predict(Zs); sil[K] = silhouette(Zs, lab_s, K) if len(np.unique(lab_s)) > 1 else -1.0
        resp = {K: fits[K].predict_proba(Zd) for K in KS}
        for krule in ("bic", "sil"):
            K = min(KS, key=bic.get) if krule == "bic" else max(KS, key=sil.get)
            for smooth in (0, 1):
                m, _ = mask_from_labels(labels_from_resp(resp[K], grid, smooth), K, gt.shape)
                out[(cov, d, krule, smooth)] = (iou(m, gt), K)
    return out


if __name__ == "__main__":
    sets, _ = image_sets()
    data, stems = sets["usod_train"] if SRC == "usod" else sets["suim_dev"]
    gt_mode = "usod" if SRC == "usod" else "suim_nonwater"
    RES.mkdir(parents=True, exist_ok=True); csv_path = RES / f"dev_ablation_{SRC}.csv"
    done = set()
    if csv_path.exists(): done = {r["image"] for r in csv.DictReader(open(csv_path))}
    else: csv.writer(open(csv_path, "w", newline="")).writerow(["image", "cov", "d", "krule", "smooth", "iou", "K"])
    model = load_model("dinov2_vitb14", DEV); t0 = time.time()
    for n, stem in enumerate(stems):
        if stem in done: continue
        X = dense(model, data / "images" / f"{stem}.jpg"); grid = X.shape[:2]; Z = pca64(X.reshape(-1, X.shape[-1]))
        res = dev_variants(Z, grid, load_gt(data, stem, gt_mode))
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for (cov, d, kr, sm), (i, K) in res.items(): w.writerow([stem, cov, d, kr, sm, f"{i:.4f}", K])
        if (n + 1) % 10 == 0: print(SRC, n + 1, len(stems), round(time.time() - t0), "s", flush=True)
    import pandas as pd
    df = pd.read_csv(csv_path, dtype={"image": str}); g = df.groupby(["cov", "d", "krule", "smooth"]).agg(iou=("iou", "mean"), meanK=("K", "mean"), n=("image", "nunique")).reset_index().sort_values("iou", ascending=False)
    best = g.iloc[0]; cfg = dict(cov=str(best["cov"]), d=int(best["d"]), krule=str(best["krule"]), smooth=int(best["smooth"]), dev_iou=float(best["iou"]), source=SRC)
    json.dump(cfg, open(RES / f"config_{SRC}.json", "w"), indent=1)
    md = [f"# GMM development ablation, source {SRC} (mean IoU of the border-rule mask on {int(best['n'])} training images)\n", "| covariance | PCA dim | K rule | smoothing | mean IoU | mean K |", "|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        mark = " **(chosen)**" if (r["cov"], r["d"], r["krule"], r["smooth"]) == (best["cov"], best["d"], best["krule"], best["smooth"]) else ""
        md.append(f"| {r['cov']} | {int(r['d'])} | {r['krule']} | {'on' if r['smooth'] else 'off'} | {r['iou']:.3f}{mark} | {r['meanK']:.2f} |")
    (RES / f"dev_ablation_{SRC}.md").write_text("\n".join(md)); print("\n".join(md[:12])); print("chosen", cfg)
