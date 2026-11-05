"""Score the GMM methods on the test images (Sec. 27): every test image is scored once with the settings / fits of BOTH source datasets.

  python gmm_eval.py usod    # all 300 USOD images (100 test + 200 others, as in the cross-dataset study X2); GT = salient object
  python gmm_eval.py suim    # 110 SUIM test images; GT = not-water and saliency
For each image: per-image GMM with config_usod and config_suim, dataset-level GMM fitted on USOD and on SUIM, GMM oracle (best component union by GT) for the
dataset's own config. 'own' = settings from the same dataset (in-domain), 'cross' = settings from the other dataset (X1 for SUIM test images, X2 for USOD).
Writes results/gmm/per_image_<set>.csv and outputs/gmm/eval_<set>.pkl (packed masks).
"""
import json, pickle, sys, time
import numpy as np, pandas as pd
from gmm_common import *

SET = sys.argv[1]
CFG = {s: json.load(open(RES / f"config_{s}.json")) for s in ("usod", "suim")}


def oracle(lab, K, gt):
    cm = KM.upsample_maps(np.stack([lab == c for c in range(K)]), gt.shape, STRIDE, SIZE)
    return max(metrics(cm[list(s)].sum(0), gt)[0] for s in subsets(K))


def main():
    sets, _ = image_sets(); data, stems = sets["usod_all"] if SET == "usod" else sets["suim_test"]
    gts = {"usod": ["usod"], "suim": ["suim_nonwater", "suim_saliency"]}[SET]
    own, cross = ("usod", "suim") if SET == "usod" else ("suim", "usod")
    glob = {s: load_global(s) for s in ("usod", "suim")}
    model = load_model("dinov2_vitb14", DEV); rows, masks = [], {}; t_start = time.time()
    for n, stem in enumerate(stems):
        X = dense(model, data / "images" / f"{stem}.jpg"); grid = X.shape[:2]; X2 = X.reshape(-1, X.shape[-1]); Z = pca64(X2)
        G = {g: load_gt(data, stem, g) for g in gts}; shape = next(iter(G.values())).shape
        row = dict(image=stem, h=shape[0], w=shape[1]); mk = {}
        for tag, src in (("own", own), ("cross", cross)):
            t = time.time(); m, K, prob, lab, fg = per_image_gmm(Z, grid, CFG[src], shape, return_all=True); dt = time.time() - t
            mk[f"pi_{tag}"] = m; row[f"K_{tag}"] = K; row[f"secs_pi_{tag}"] = dt
            if tag == "own":
                for g in gts: row[f"oracle_{g}"] = oracle(lab, K, G[g])
            t = time.time(); gp = mask_from_prob(global_prob(X2, grid, glob[src]), shape) > 0.5; row[f"secs_gl_{tag}"] = time.time() - t
            mk[f"gl_{tag}"] = gp
            for g in gts:
                row[f"iou_pi_{tag}_{g}"] = iou(m, G[g]); row[f"iou_gl_{tag}_{g}"] = iou(gp, G[g])
            row[f"fg_pi_{tag}"] = float(m.mean()); row[f"fg_gl_{tag}"] = float(gp.mean())
        rows.append(row); masks[stem] = {k: (np.packbits(v), v.shape) for k, v in mk.items()}
        if (n + 1) % 20 == 0:
            print(SET, n + 1, len(stems), round(time.time() - t_start), "s", flush=True)
            pd.DataFrame(rows).to_csv(RES / f"per_image_{SET}.csv", index=False)
    pd.DataFrame(rows).to_csv(RES / f"per_image_{SET}.csv", index=False)
    pickle.dump(masks, open(OUT / f"eval_{SET}.pkl", "wb")); print("eval done", SET)


if __name__ == "__main__":
    main()
