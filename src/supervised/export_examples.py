"""One PNG per example image: that image and every method's mask side by side (never many images in one file).

  python export_examples.py          # existing cross-dataset results -> results/cross_dataset/examples/
Selection by rule: 10 best / 10 worst images for the model, 5 where a label-free baseline beats it.
Folders: good/, bad/, baseline_wins/ ; IoU of the model is in the file name; index.csv lists every score.
"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
from baseline_masks import get_masks

BASELINES = ["k-means (border rule)", "dendrogram (gap rule)"]


def save_collage(path, img, gt, panels, header):
    """panels: list of (title, mask, iou|None). One row: image | ground truth | panels..."""
    n = 2 + len(panels)
    fig, axs = plt.subplots(1, n, figsize=(3.0 * n, 2.9))
    axs[0].imshow(img); axs[0].set_title("image", fontsize=9)
    axs[1].imshow(gt, cmap="gray"); axs[1].set_title("ground truth", fontsize=9)
    for ax, (t, m, iou) in zip(axs[2:], panels):
        ax.imshow(m, cmap="gray"); ax.set_title(t, fontsize=9)
        if iou is not None:
            ax.text(.03, .04, f"IoU {iou:.2f}", transform=ax.transAxes, color="white", fontsize=10, va="bottom",
                    bbox=dict(fc="black", alpha=.65, lw=0, pad=2))
    for ax in axs: ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    fig.suptitle(header, fontsize=9, x=0.01, ha="left")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def select(df, model, n_good=10, n_bad=10, n_base=5):
    d = df[df[BASELINES[1]].notna()].sort_values(model)
    sel = {"good": list(d.image.iloc[::-1][:n_good]), "bad": list(d.image[:n_bad])}
    gap = d.assign(g=d[BASELINES].max(1) - d[model]).sort_values("g", ascending=False)
    sel["baseline_wins"] = [s for s in gap.image if gap.set_index("image").g[s] > 0][:n_base]
    return sel


def export(setting, outdir, items, gtfun, panel_fn, df, model, extra_sel=None):
    """items: dict stem -> record; panel_fn(stem, masks) -> list of (title, mask, iou)."""
    sel = select(df, model)
    sel.update(extra_sel or {})
    rows = []
    for cat, stems in sel.items():
        (outdir / cat).mkdir(parents=True, exist_ok=True)
        masks = get_masks([items[s] for s in stems])
        for s in stems:
            it = items[s]; gt = gtfun(it)
            iou = float(df.set_index("image").loc[s, model])
            img = np.asarray(Image.open(it["img"]).convert("RGB").resize(gt.shape[::-1]))
            f = outdir / cat / f"{setting}_{s}_iou{iou:.2f}.png"
            save_collage(f, img, gt, panel_fn(s, masks[s], gt), f"{setting} | {s} | {cat}")
            rows.append(dict(category=cat, setting=setting, image=s, file=str(f.relative_to(outdir)), **{c: df.set_index("image").loc[s, c] for c in df.columns if c.startswith(("cross", "in-domain", "k-means", "dendro", "model"))}))
    pd.DataFrame(rows).to_csv(outdir / f"index_{setting}.csv", index=False)
    return rows


if __name__ == "__main__":
    XR = ROOT / "results" / "cross_dataset"
    recs = build_records()
    ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
    evpos = {i: k for k, i in enumerate(ev)}
    IND = np.load(pred_path("stage_C", "evalset")).astype(np.float32)
    IND_THR = json.load(open(RES / "stage_C" / "config.json"))["thr"]
    SET = {"X1_saliency": ("x1", "usod", "suim", "suim_saliency"), "X1_nonwater": ("x1", "usod", "suim", "suim_nonwater"),
           "X2_usod": ("x2", "suim", "usod", "usod")}
    for name, (tag, src, tgt, gtm) in SET.items():
        df = pd.read_csv(XR / f"per_image_{name}.csv", dtype={"image": str})
        df = df[df["dendrogram (gap rule)"].notna()]
        if "in-domain C" in df: df = df[df["in-domain C"].notna()]
        idx = list(np.load(OUT / "preds" / f"cross_{tag}_index.npy"))
        P = {s: np.load(OUT / "preds" / f"cross_{tag}_{s}.npy").astype(np.float32) for s in "ABC"}
        thr = {s: json.load(open(XR / tag / f"stage_{s}" / "config.json"))["thr"][src] for s in "ABC"}
        items = {}
        for i in idx:
            r = recs[i]; items[r["stem"]] = dict(r, gt_mode=gtm, rec=i)

        def panel_fn(stem, m, gt, df=df, idx=idx, P=P, thr=thr, items=items, tgt=tgt):
            i = items[stem]["rec"]; k = idx.index(i); row = df.set_index("image").loc[stem]
            out = [(n, *m[n]) for n in ("k-means (border rule)", "dendrogram (gap rule)", "k-means oracle (uses GT)")]
            if "in-domain C" in df:
                out.append(("in-domain C (trained on target)", prob_to_mask(IND[evpos[i]][CH[tgt]], gt.shape, IND_THR[tgt]), row["in-domain C"]))
            out += [(f"cross {s}", prob_to_mask(P[s][k], gt.shape, thr[s]), row[f"cross {s}"]) for s in "ABC"]
            return out
        rows = export(name, XR / "examples", items, lambda it, gtm=gtm: load_gt(it["data"], it["stem"], gtm), panel_fn, df, "cross C")
        print(name, len(rows), "collages")
