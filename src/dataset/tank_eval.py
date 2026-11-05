"""Score the models on the tank frames with the agreed ground truth (Sec. 26).

Ground truth: fish frames = masks (approved) or masks_draft (SAM drafts, marked provisional until the user has approved them) in data/dhwani_dataset/;
every other frame has NO subject (empty mask). Tank frames are test-only: this script never trains or tunes anything.

  python tank_eval.py pool2        # new unified model (tag pool2, label subject2) against the earlier methods
Outputs: results/subject_v3/tank/{report.md, summary.csv, per_image.csv, summary_bars.png, examples/}
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_on_gopro as R   # noqa: E402
from run_on_gopro import ROOT, BULK, OUT_HW, unpack, overlay, meta, pack, sup_predict, load_old, tta  # noqa: E402
from PIL import Image  # noqa: E402

D = ROOT / "data" / "dhwani_dataset"
RES = ROOT / "results" / "subject_v3" / "tank"
TAG = sys.argv[1] if len(sys.argv) > 1 else "pool2"
ANYFG = 0.01
OLD = ["kmeans", "dendro", "old_sal", "uni_A", "uni_B", "uni_C"]
NEW = ["new_A", "new_B", "new_C"]
NAMES = {"kmeans": "k-means (border rule)", "dendro": "dendrogram (gap rule)", "old_sal": "earlier C (Sec. 21, salient channel)",
         "uni_A": "unified A (Sec. 23)", "uni_B": "unified B (Sec. 23)", "uni_C": "unified C (Sec. 23)",
         "new_A": "new A (Sec. 26)", "new_B": "new B (Sec. 26)", "new_C": "new C (Sec. 26)"}
METHODS = OLD + NEW


def gt_masks(d):
    """-> list of bool [540,960] masks, and per-frame source ('approved' | 'draft' | 'none')."""
    out, src = [], []
    for f in d.file:
        stem = Path(f).stem
        for folder, tag in (("masks", "approved"), ("masks_draft", "draft")):
            p = D / folder / f"{stem}.png"
            if p.exists():
                m = np.asarray(Image.open(p).convert("L").resize(OUT_HW[::-1], Image.NEAREST)) > 127
                out.append(m); src.append(tag); break
        else:
            out.append(np.zeros(OUT_HW, bool)); src.append("none")
    return out, src


def new_masks(d):
    f = BULK / f"masks_{TAG}.npz"
    if f.exists(): return dict(np.load(f))
    uni = tta.load_models(TAG)
    thr = {s: json.load(open(ROOT / "results" / "subject_v3" / TAG / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
    models = (uni, thr, load_old())
    store = {k: [] for k in ("uni_A", "uni_B", "uni_C")}
    for i, p in enumerate(d.path):
        m = sup_predict(Image.open(p), models, extras=False)
        for k in store: store[k].append(m[k])
        if i % 50 == 0: print("new model", i, flush=True)
    z = {f"new_{k[-1]}": pack(np.stack(v)) for k, v in store.items()}
    np.savez_compressed(f, **z); return z


def metrics(pred, gt):
    i = (pred & gt).sum(); u = (pred | gt).sum()
    prec = i / pred.sum() if pred.sum() else 0.0; rec = i / gt.sum() if gt.sum() else 1.0
    fb = 1.3 * prec * rec / (0.3 * prec + rec) if (0.3 * prec + rec) else 0.0
    return dict(iou=i / u if u else 1.0, precision=prec, recall=rec, fbeta=fb)


def boot(x, n=2000):
    x = np.asarray(x, float); rng = np.random.RandomState(0)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1); return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = meta(); gts, gsrc = gt_masks(d)
    z = {**dict(np.load(BULK / "masks_sup.npz")), **dict(np.load(BULK / "masks_base.npz")), **new_masks(d)}
    RES.mkdir(parents=True, exist_ok=True)
    has = np.array([g.any() for g in gts]); d["subject"] = has; d["gt_source"] = gsrc
    rows = []
    for i in range(len(d)):
        row = dict(file=d.file[i], tag=d.tag[i], subject=bool(has[i]), gt_source=gsrc[i], gt_share=float(gts[i].mean()))
        for k in METHODS:
            p = unpack(z[k], i); row[f"fg_{k}"] = float(p.mean())
            if has[i]: row.update({f"{m}_{k}": v for m, v in metrics(p, gts[i]).items()})
        rows.append(row)
    per = pd.DataFrame(rows); per.to_csv(RES / "per_image.csv", index=False)
    sub, non = per[per.subject], per[~per.subject]
    summ = []
    for k in METHODS:
        lo, hi = boot(sub[f"iou_{k}"]); fa = non[f"fg_{k}"]
        summ.append(dict(method=k, n_subject=len(sub), iou=sub[f"iou_{k}"].mean(), iou_lo=lo, iou_hi=hi, recall=sub[f"recall_{k}"].mean(), precision=sub[f"precision_{k}"].mean(),
                         n_nosubject=len(non), false_alarm_rate=float((fa > ANYFG).mean()), severe_rate=float((fa > 0.2).mean()), mean_fp_share=float(fa.mean())))
    S = pd.DataFrame(summ); S.to_csv(RES / "summary.csv", index=False)
    prov = int((per.gt_source == "draft").sum())
    md = ["# Tank frames with ground truth (Sec. 26)\n",
          f"Subject = animal or object (fish here); everything else, including tank panels, floor and debris, is background. {len(sub)} frames contain a subject, {len(non)} frames have none (empty mask). "
          + (f"**{prov} of the subject masks are SAM drafts that the project owner has not yet approved: the numbers are provisional.**\n" if prov else "All subject masks are approved.\n"),
          "Tank frames are test-only: nothing was trained or tuned on them, and each model was scored once.\n",
          f"## Fish frames (n={len(sub)}): overlap with the ground-truth mask\n", "| method | mean IoU [95% CI] | recall (fish pixels found) | precision |", "|---|---|---|---|"]
    for _, r in S.iterrows(): md.append(f"| {NAMES[r.method]} | **{r.iou:.3f}** [{r.iou_lo:.3f}, {r.iou_hi:.3f}] | {r.recall:.3f} | {r.precision:.3f} |")
    md += ["", f"## Frames with no subject (n={len(non)}): false alarms (a false alarm = more than {ANYFG:.0%} of the frame predicted as subject)\n",
           "| method | frames with a false alarm | severe (>20% of the frame) | mean predicted share of the frame |", "|---|---|---|---|"]
    for _, r in S.iterrows(): md.append(f"| {NAMES[r.method]} | {r.false_alarm_rate:.0%} | {r.severe_rate:.0%} | {r.mean_fp_share:.1%} |")
    md += ["", "By scene tag (false-alarm rate / mean predicted share):\n", "| method | " + " | ".join(f"{t} (n={int((non.tag == t).sum())})" for t in ["structure", "floor", "waterline", "empty"]) + " |", "|---|---|---|---|---|"]
    for k in METHODS:
        md.append(f"| {NAMES[k]} | " + " | ".join(f"{(non[non.tag == t][f'fg_{k}'] > ANYFG).mean():.0%} / {non[non.tag == t][f'fg_{k}'].mean():.1%}" for t in ["structure", "floor", "waterline", "empty"]) + " |")
    # ---- figure
    COL = {"kmeans": "#9ca3af", "dendro": "#6b7280", "old_sal": "#d97706", "uni_A": "#bfdbfe", "uni_B": "#60a5fa", "uni_C": "#1d4ed8", "new_A": "#bbf7d0", "new_B": "#4ade80", "new_C": "#15803d"}
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.6))
    axs[0].barh(range(len(S)), S.iou[::-1], color=[COL[m] for m in S.method[::-1]], xerr=[S.iou[::-1] - S.iou_lo[::-1], S.iou_hi[::-1] - S.iou[::-1]], error_kw=dict(lw=1, capsize=2))
    axs[0].set_yticks(range(len(S))); axs[0].set_yticklabels([NAMES[m] for m in S.method[::-1]], fontsize=7); axs[0].set_title(f"IoU on the {len(sub)} fish frames", fontsize=10, loc="left"); axs[0].set_xlim(0, 1)
    axs[1].barh(range(len(S)), S.false_alarm_rate[::-1], color=[COL[m] for m in S.method[::-1]]); axs[1].set_yticks(range(len(S))); axs[1].set_yticklabels([]); axs[1].set_title(f"no-subject frames with a false alarm (n={len(non)})", fontsize=10, loc="left"); axs[1].set_xlim(0, 1)
    axs[2].barh(range(len(S)), S.mean_fp_share[::-1], color=[COL[m] for m in S.method[::-1]]); axs[2].set_yticks(range(len(S))); axs[2].set_yticklabels([]); axs[2].set_title("mean predicted share of a no-subject frame", fontsize=10, loc="left"); axs[2].set_xlim(0, 1)
    for a in axs:
        for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    plt.tight_layout(); plt.savefig(RES / "summary_bars.png", dpi=140); plt.close()
    # ---- one collage per example frame
    ex = RES / "examples"; picks = []
    f = sub.sort_values("iou_new_C")
    for why, idx in (("fish_best", f.index[-3:][::-1]), ("fish_worst", f.index[:3])):
        picks += [(why, i) for i in idx]
    for t in ["structure", "floor", "waterline", "empty"]:
        g = non[non.tag == t].sort_values("fg_new_C")
        picks += [("false_alarm_largest", i) for i in g.index[-2:][::-1]] + [("no_subject_median", g.index[len(g) // 2])]
    show = ["kmeans", "dendro", "old_sal", "uni_C", "new_A", "new_B", "new_C"]
    for why, i in picks:
        img = np.asarray(Image.open(d.path[i]).convert("RGB").resize(OUT_HW[::-1]))
        panels = [("image" + (" + ground truth outline" if has[i] else " (no subject)"), overlay(img, gts[i]) if has[i] else img, None)]
        for k in show:
            pm = unpack(z[k], i); im = overlay(img, pm)
            if has[i]:
                cs, _ = cv2.findContours(gts[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cs, -1, (0, 255, 255), 2)
            lab = f"IoU {per[f'iou_{k}'][i]:.2f}, " if has[i] else ""
            panels.append((NAMES[k], im, lab + f"predicted {pm.mean():.0%}"))
        fig, axs = plt.subplots(2, 4, figsize=(20, 6.4))
        for ax, (ttl, im, txt) in zip(axs.ravel(), panels):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=10)
            for s_ in ax.spines.values(): s_.set_visible(False)
            if txt: ax.text(.02, .04, txt, transform=ax.transAxes, color="white", fontsize=11, bbox=dict(fc="black", alpha=.65, lw=0, pad=2))
        fig.suptitle(f"{d.file[i]} | {d.tag[i]} | {why} | orange = predicted subject, cyan = ground truth outline", fontsize=10, x=0.01, ha="left")
        plt.tight_layout(); (ex / why).mkdir(parents=True, exist_ok=True)
        plt.savefig(ex / why / f"{Path(d.file[i]).stem}_new{per['fg_new_C'][i]*100:02.0f}pct.png", dpi=80); plt.close()
    md += ["", "## Examples (one PNG per frame)\n", "`examples/fish_best`, `fish_worst` (by new-C IoU), `false_alarm_largest` and `no_subject_median` for each no-subject tag. Panels: frame (with the ground-truth outline in cyan), k-means, dendrogram, earlier C, unified C (Sec. 23), new A, B, C.\n",
           "`summary_bars.png`, `per_image.csv` and `summary.csv` hold all numbers."]
    (RES / "report.md").write_text("\n".join(md)); print("\n".join(md))


if __name__ == "__main__":
    main()
