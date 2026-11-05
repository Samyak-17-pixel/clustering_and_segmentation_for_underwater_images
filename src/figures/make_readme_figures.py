"""Builds every image used in README.md, results.md and docs/*.md  ->  docs/figures/

Two kinds of figures:
  * diagrams and charts drawn here with matplotlib (concept diagrams, bar charts from the numbers file);
  * photo figures (original image + masks of each method) that were produced by earlier scripts; they are copied
    here, shrunk to at most 1600 px wide, so the documentation does not depend on the large local result folders.

Inputs (local, not in git): presentations/figures_r3/ (run presentations/build_figures_review3.py first) and
results/doc_figure_sources/ (written by analysis/analyze_results.py, supervised/pipeline_figure.py, supervised/worked_example.py,
dendrogram figures). Output: docs/figures/*.png (about 40 files, 12 MB).

    python src/figures/make_readme_figures.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
R3 = ROOT / "presentations" / "figures_r3"
OUT.mkdir(parents=True, exist_ok=True)

INK, BLUE, ORANGE, GREEN, GREY, RED, LIGHT = "#1f2937", "#1a73b5", "#e8832a", "#2e9e6b", "#9aa5b1", "#c0392b", "#eef3f8"
FROZEN, TRAIN, NEUT = "#cfe3f7", "#fde3c4", "#eeeeee"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def shrink(src, name, width=1600):
    im = Image.open(src).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(OUT / name, optimize=True)
    print("copied", name, im.size)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=NEUT, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))


# ----------------------------------------------------------------------------------------- concept diagrams
def fig_overview():
    fig, ax = canvas(15, 6.4)
    ax.text(7.5, 6.05, "Project overview: from a photo to a subject mask", ha="center", fontsize=13, fontweight="bold", color=INK)
    box(ax, 0.2, 2.6, 1.8, 1.2, "underwater\nimage", NEUT)
    arrow(ax, 2.05, 3.2, 2.6, 3.2)
    box(ax, 2.65, 2.4, 2.4, 1.6, "frozen DINOv2\n(768 numbers per\n14x14 patch)", FROZEN, bold=True)
    arrow(ax, 5.1, 3.5, 5.9, 4.6); arrow(ax, 5.1, 2.9, 5.9, 1.7)
    ax.text(6.0, 5.55, "PART 1: no labels (unsupervised)", fontsize=10.5, fontweight="bold", color=BLUE)
    box(ax, 5.9, 4.1, 2.2, 1.2, "group the patches\nk-means, dendrogram,\nGMM", LIGHT, 8.5)
    arrow(ax, 8.15, 4.7, 8.75, 4.7)
    box(ax, 8.8, 4.1, 2.3, 1.2, "guess which groups\nare the subject\n(border rule)", LIGHT, 8.5)
    ax.text(6.0, 2.05, "PART 2: with labels (supervised)", fontsize=10.5, fontweight="bold", color=ORANGE)
    box(ax, 5.9, 0.45, 2.2, 1.35, "learn from masks\nA linear, B conv head,\nC LoRA, D SAM", TRAIN, 8.5)
    arrow(ax, 8.15, 1.1, 8.75, 1.1)
    box(ax, 8.8, 0.45, 2.3, 1.35, "one subject rule\n(same label for\nall datasets)", TRAIN, 8.5)
    arrow(ax, 11.15, 4.7, 11.9, 3.5); arrow(ax, 11.15, 1.1, 11.9, 2.7)
    box(ax, 11.95, 2.4, 2.8, 1.6, "subject mask\n(white = subject)\nscored by IoU", NEUT, bold=True)
    ax.text(7.5, 0.05, "PART 3: the same models are tested on 302 new tank frames (Dhwani) that were never used for training.",
            ha="center", fontsize=9.5, color=INK)
    save(fig, "01_overview.png")


def fig_border_rule():
    rng = np.random.default_rng(3)
    H, W = 12, 16
    lab = np.zeros((H, W), int)                      # 0 water, 1 object, 2 object-touching-border example
    lab[3:9, 5:11] = 1
    lab[0:3, 12:16] = 2
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = np.array([[0.75, 0.85, 0.95], [0.95, 0.6, 0.2], [0.6, 0.8, 0.5]])
    axs[0].imshow(cols[lab]); axs[0].set_title("1. k-means puts patches into groups\n(blue = group A, orange = B, green = C)", fontsize=9.5)
    border = np.zeros((H, W), bool); border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    axs[1].imshow(cols[lab]); axs[1].set_title("2. which groups touch the image border?\n(red frame = border patches)", fontsize=9.5)
    axs[1].add_patch(Rectangle((-0.5, -0.5), W, H, fill=False, ec=RED, lw=4))
    fg = ~np.isin(lab, [0, 2])
    axs[2].imshow(np.where(fg, 1.0, 0.1), cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("3. border groups = water, the rest = subject\n(white = predicted subject)", fontsize=9.5)
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
    fig.suptitle("Border rule: no labels needed, but it fails when the subject touches the border", fontsize=11, fontweight="bold")
    save(fig, "03_border_rule.png")


def fig_dendrogram():
    rng = np.random.default_rng(1)
    pts = np.vstack([rng.normal(0, .35, (6, 2)), rng.normal([3, 0], .35, (6, 2)), rng.normal([1.5, 3.2], .35, (5, 2))])
    Z = linkage(pts, "average")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    axs[0].scatter(pts[:, 0], pts[:, 1], c=[BLUE] * 6 + [ORANGE] * 6 + [GREEN] * 5, s=45)
    axs[0].set_title("patches as points (two numbers each\ninstead of 768, for the picture)", fontsize=10); axs[0].set_xticks([]); axs[0].set_yticks([])
    h = Z[:, 2]; gaps = np.diff(h); cut = (h[np.argmax(gaps)] + h[np.argmax(gaps) + 1]) / 2
    dendrogram(Z, ax=axs[1], color_threshold=cut, above_threshold_color=GREY, no_labels=True)
    axs[1].axhline(cut, color=RED, ls="--", lw=1.6)
    axs[1].text(axs[1].get_xlim()[1] * 0.98, cut + 0.08, "cut here: the largest jump between merge heights", color=RED, fontsize=9, ha="right", bbox=dict(fc="white", ec="none", pad=1))
    axs[1].set_ylabel("merge height (distance)")
    axs[1].set_title("dendrogram: merge the two closest groups, again and again", fontsize=10)
    save(fig, "04_dendrogram_gap_rule.png")


def fig_splits():
    fig, ax = canvas(13, 4.2)
    ax.text(6.5, 3.9, "Data splits (fixed seed, made once). Test images are never used to choose anything.", ha="center", fontsize=11, fontweight="bold")
    def bar(y, label, parts):
        ax.text(0.1, y + 0.3, label, fontsize=10, fontweight="bold", va="center")
        x = 2.3; total = sum(p[1] for p in parts)
        for name, n, c in parts:
            w = 10.2 * n / total
            ax.add_patch(Rectangle((x, y), w, 0.6, fc=c, ec=INK)); ax.text(x + w / 2, y + 0.3, f"{name}\n{n}", ha="center", va="center", fontsize=9)
            x += w
    bar(2.4, "USOD (300)", [("train", 160, FROZEN), ("val", 40, TRAIN), ("test", 100, "#cdeccf")])
    bar(1.3, "SUIM (1,635)", [("train", 1375, FROZEN), ("val", 150, TRAIN), ("test", 110, "#cdeccf")])
    bar(0.2, "Dhwani (302)", [("test only: 19 fish frames (draft masks) + 283 with no subject", 302, "#e6d5f2")])
    ax.text(2.3, 3.3, "train = the model learns from it    val = we pick settings    test = final score, used once", fontsize=9, color=INK)
    save(fig, "08_data_splits.png")


def fig_lora():
    fig, ax = canvas(12, 3.6)
    ax.text(6, 3.3, "LoRA: leave the big weight matrix alone, add a small trainable correction next to it", ha="center", fontsize=11, fontweight="bold")
    box(ax, 0.3, 1.2, 2.6, 1.4, "x (patch\nfeatures)", NEUT); arrow(ax, 2.95, 1.9, 3.6, 2.4); arrow(ax, 2.95, 1.9, 3.6, 1.1)
    box(ax, 3.65, 2.0, 3.3, 0.9, "frozen weight W (768 x 768)", FROZEN, bold=True)
    box(ax, 3.65, 0.5, 1.4, 0.9, "A  (768 x 16)\ntrained", TRAIN, 8.5); arrow(ax, 5.1, 0.95, 5.35, 0.95)
    box(ax, 5.4, 0.5, 1.55, 0.9, "B  (16 x 768)\ntrained", TRAIN, 8.5)
    arrow(ax, 7.0, 2.45, 7.7, 1.95); arrow(ax, 7.0, 0.95, 7.7, 1.55)
    box(ax, 7.75, 1.3, 1.1, 1.0, "add", NEUT); arrow(ax, 8.9, 1.8, 9.5, 1.8)
    box(ax, 9.55, 1.3, 2.2, 1.0, "output\nW x + B(A x)", NEUT)
    ax.text(6, 0.05, "rank 16 in all 12 blocks (attention + patch embedding): 906,432 trainable weights, about 1% of DINOv2", ha="center", fontsize=9)
    save(fig, "09_lora_concept.png")


def fig_subject_rule():
    fig, ax = canvas(13, 4.6)
    ax.text(6.5, 4.3, "The one subject rule (written down so that every dataset uses the same definition)", ha="center", fontsize=11.5, fontweight="bold")
    box(ax, 0.2, 1.8, 2.2, 1.0, "look at the\nscene's classes", NEUT)
    arrow(ax, 2.45, 2.3, 3.2, 3.4); arrow(ax, 2.45, 2.3, 3.2, 2.3); arrow(ax, 2.45, 2.3, 3.2, 1.1)
    box(ax, 3.25, 3.0, 4.6, 0.9, "1  animals, divers, robots or wrecks present?", LIGHT, 9)
    box(ax, 3.25, 1.85, 4.6, 0.9, "2  only reef / plants + water?", LIGHT, 9)
    box(ax, 3.25, 0.7, 4.6, 0.9, "3  only water / sand / tank walls?", LIGHT, 9)
    for y in (3.45, 2.3, 1.15): arrow(ax, 7.9, y, 8.6, y)
    box(ax, 8.65, 3.0, 4.1, 0.9, "subject = those animals / objects", "#cdeccf", 9)
    box(ax, 8.65, 1.85, 4.1, 0.9, "subject = the reef / plants", "#cdeccf", 9)
    box(ax, 8.65, 0.7, 4.1, 0.9, "no subject: the correct mask is EMPTY", "#f6d5d5", 9)
    ax.text(6.5, 0.15, "(Section 26 later tightened rule 2: reef and plants count as background. USOD already follows the rule: its salient object is the subject.)",
            ha="center", fontsize=8.5)
    save(fig, "10_subject_rule.png")


def fig_dataflow():
    fig, ax = canvas(15.4, 6.2)
    ax.text(7.5, 5.9, "Which script produces what (all paths relative to the repository root)", ha="center", fontsize=12, fontweight="bold")
    cols = [("data/ (not in git)", 0.2, ["USOD images + masks", "SUIM images + masks", "videos/ -> Dhwani frames"], NEUT),
            ("src/kmeans, dendrogram, gmm", 3.4, ["kmeans/*.py", "dendrogram/*.py", "gmm/*.py"], LIGHT),
            ("src/supervised", 6.6, ["features.py", "labels.py", "train_head.py  finetune.py", "sam_refine.py  tta.py", "compare*.py  false_alarm.py"], TRAIN),
            ("src/dataset", 9.8, ["build_gopro_dataset.py", "sam_draft.py", "run_on_gopro.py  tank_eval.py", "make_videos.py"], "#e6d5f2"),
            ("outputs/, results/ (not in git)", 12.6, ["features, checkpoints,", "predictions (outputs/)", "tables, collages, videos", "(results/)"], NEUT)]
    for title, x, items, c in cols:
        ax.text(x + 1.1, 5.35, title, ha="center", fontsize=9.5, fontweight="bold")
        for i, t in enumerate(items): box(ax, x, 4.3 - i * 0.95, 2.4, 0.75, t, c, 8.2)
    for x0, x1 in ((2.65, 3.35), (5.85, 6.55), (9.05, 9.75), (12.25, 12.55)): arrow(ax, x0, 3.2, x1, 3.2)
    ax.text(7.5, 0.25, "docs/figures/ and results.md are built from results/ by src/figures/make_readme_figures.py", ha="center", fontsize=9)
    save(fig, "21_data_flow.png")


def fig_timeline():
    fig, ax = canvas(14, 3.8)
    ax.text(7, 3.5, "Project timeline and status", ha="center", fontsize=12, fontweight="bold")
    items = [("Review 1", "DINOv2 features,\nPCA mask, k-means k=2\nIoU 0.42 / 0.49", BLUE),
             ("Review 2", "k-means 2-6, oracle,\ndendrogram\nIoU 0.52 -> 0.65", BLUE),
             ("After R2", "Gaussian mixtures\n(per-image,\ndataset-level) 0.54 / 0.59", BLUE),
             ("Supervised", "A linear 0.80,\nB conv 0.82,\nC LoRA 0.87, D SAM 0.87", ORANGE),
             ("Subject rule", "cross-dataset 0.36 -> 0.78\nunified model 0.84", ORANGE),
             ("Tank (Dhwani)", "302 new frames,\nfish IoU up to 0.81,\nfalse alarms open", GREEN)]
    ax.plot([0.7, 13.3], [2.2, 2.2], color=GREY, lw=3, zorder=0)
    for i, (t, d, c) in enumerate(items):
        x = 1.2 + i * 2.35
        ax.scatter([x], [2.2], s=260, color=c, zorder=3, ec=INK)
        ax.text(x, 2.75, t, ha="center", fontsize=10, fontweight="bold"); ax.text(x, 1.15, d, ha="center", va="center", fontsize=8.5)
    ax.text(0.7, 0.2, "blue = no labels (unsupervised)    orange = with labels (supervised)    green = test on new footage", fontsize=9)
    save(fig, "22_roadmap_timeline.png")


def fig_repro_order():
    fig, ax = canvas(15, 3.6)
    ax.text(7.5, 3.3, "Run order to regenerate everything (details and commands: docs/REPRODUCE.md)", ha="center", fontsize=12, fontweight="bold")
    steps = [("0 setup\nrequirements,\ndatasets", NEUT), ("1 k-means\n(E1-E4, SUIM)", LIGHT), ("2 dendrogram\n+ summaries", LIGHT), ("3 GMM\ndev, fit, eval", LIGHT),
             ("4 features\n+ labels", TRAIN), ("5 heads A, B\nLoRA C, SAM D", TRAIN), ("6 compare,\ncross-dataset,\nseeds, alarms", TRAIN), ("7 Dhwani:\nbuild, SAM drafts,\ntank eval", "#e6d5f2"), ("8 figures\n+ decks", NEUT)]
    for i, (t, c) in enumerate(steps):
        x = 0.15 + i * 1.65
        box(ax, x, 1.1, 1.45, 1.4, t, c, 8)
        if i < len(steps) - 1: arrow(ax, x + 1.47, 1.8, x + 1.63, 1.8)
    ax.text(7.5, 0.4, "steps 1-3 are independent of 4-6; step 7 needs the trained models of step 5; every step writes only to outputs/ and results/", ha="center", fontsize=9)
    save(fig, "24_repro_order.png")


def fig_data_layout():
    fig, ax = canvas(12, 4.6)
    ax.text(6, 4.3, "Where the data lives (folders are git-ignored; you supply them)", ha="center", fontsize=12, fontweight="bold")
    rows = [("data/USOD/", "images/ (300 .jpg 640x480)   masks/ (300 .png, salient object)", FROZEN),
            ("data/SUIM/", "train_val/ (1,525 images + class masks)   TEST/ (110 images, masks/ per class + Saliency)", FROZEN),
            ("data/dhwani_dataset/", "images/ (302 frames)   metadata.csv (tags)   masks_draft/ (19 SAM drafts)   REVIEW.md", "#e6d5f2"),
            ("videos/", "raw GoPro clips (only needed to rebuild the Dhwani frames and the videos)", NEUT),
            ("outputs/  results/", "created by the scripts (feature caches, checkpoints, predictions, tables, collages)", TRAIN)]
    for i, (a, b, c) in enumerate(rows):
        y = 3.4 - i * 0.78
        box(ax, 0.2, y, 2.9, 0.62, a, c, 9.5, True); ax.text(3.3, y + 0.31, b, va="center", fontsize=8.8)
    save(fig, "23_data_layout.png")


# ----------------------------------------------------------------------------------------- charts from numbers.json
def load_numbers():
    p = R3 / "numbers.json"
    if not p.exists():
        raise SystemExit("presentations/figures_r3/numbers.json missing: run presentations/build_figures_review3.py first")
    return json.load(open(p))


def bars(ax, groups, series, colors, ylabel="mean IoU", ymax=1.0, labels=True, width=None):
    n = len(series); width = width or 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        xs = np.arange(len(groups)) + (i - (n - 1) / 2) * width
        b = ax.bar(xs, vals, width * 0.92, label=name, color=colors[i], ec="white")
        if labels:
            for x, v in zip(xs, vals):
                if v == v: ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(np.arange(len(groups))); ax.set_xticklabels(groups, fontsize=9); ax.set_ylim(0, ymax); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e5e7eb"); ax.set_axisbelow(True)


def fig_unsup_bars(N):
    g = N["gmm"]
    sets = [("In-domain USOD (100 test)", "In-domain USOD (test 100)"), ("SUIM not-water", "In-domain SUIM, not-water (test)"),
            ("SUIM saliency", "In-domain SUIM, saliency (test)"), ("SUIM settings -> USOD", "X2: SUIM settings -> USOD test 100")]
    meth = [("k-means (border rule)", "k-means (border rule)"), ("dendrogram (gap rule)", "dendrogram (gap rule)"),
            ("GMM per-image", "GMM per-image"), ("GMM dataset-level", "GMM dataset-level")]
    series = {m[0]: [g[s[1]][m[1]] for s in sets] for m in meth}
    fig, ax = plt.subplots(figsize=(12, 4.4))
    bars(ax, [s[0] for s in sets], series, [GREY, BLUE, ORANGE, GREEN])
    orc = [g[s[1]]["k-means oracle (uses GT)"] for s in sets]
    for i, v in enumerate(orc): ax.hlines(v, i - 0.42, i + 0.42, color=RED, lw=2, ls="--")
    ax.plot([], [], color=RED, ls="--", label="oracle (uses the answer: ceiling)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title("Unsupervised methods: mean IoU on test images", y=1.12, fontsize=11, fontweight="bold")
    save(fig, "13_bars_unsupervised.png")


def fig_bic(N):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, key, ttl in zip(axs, ("dev_usod", "dev_suim"), ("USOD training images", "SUIM training images")):
        d = N[key]; v = [d["bic"], d["sil"]]
        ax.bar(["BIC rule\n(picks %.1f)" % d["meanK_bic"], "silhouette rule\n(picks %.1f)" % d["meanK_sil"]], v, color=[GREY, GREEN], width=0.55)
        for i, x in enumerate(v): ax.text(i, x + 0.01, f"{x:.2f}", ha="center")
        if "kmeans" in d: ax.axhline(d["kmeans"], color=RED, ls="--"); ax.text(-0.45, d["kmeans"] + 0.015, "k-means %.2f" % d["kmeans"], color=RED, ha="left", fontsize=8.5)
        ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 0.8)
    axs[0].set_ylabel("mean IoU")
    fig.suptitle("Choosing the number of Gaussians: BIC over-segments, silhouette works", fontsize=11, fontweight="bold")
    save(fig, "06_bic_vs_silhouette.png")


def fig_sup_bars(N):
    s = N["sup"]
    order = [("k-means\n(border)", "k-means, border rule (stride 3, k<=6)"), ("dendrogram\n(best no-label)", "dendrogram, gap rule (best label-free)"),
             ("A linear\nprobe", "A: linear probe (frozen)"), ("B conv\nhead", "B: conv head (frozen)"), ("C + LoRA", "C: + LoRA fine-tune"), ("D + SAM", "D: + SAM refinement")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    cols = [GREY, BLUE, ORANGE, ORANGE, ORANGE, "#d9b38c"]
    for ax, ds, ttl in zip(axs, ("usod", "suim"), ("USOD test (100 images, salient object)", "SUIM test (92 images, everything that is not water)")):
        v = [s[ds][k] for _, k in order]
        ax.bar([o[0] for o in order], v, color=cols, width=0.7)
        for i, x in enumerate(v): ax.text(i, x + 0.01, f"{x:.2f}", ha="center", fontsize=9)
        ax.axhline(s[ds]["k-means ORACLE (uses GT)"], color=RED, ls="--", lw=1.3); ax.text(5.4, s[ds]["k-means ORACLE (uses GT)"] + 0.01, "oracle", color=RED, ha="right", fontsize=8.5)
        ax.set_title(ttl, fontsize=10); ax.set_ylim(0, 1.0); ax.tick_params(axis="x", labelsize=8)
    axs[0].set_ylabel("mean IoU")
    fig.suptitle("Supervised stages A to D versus the best label-free methods", fontsize=11, fontweight="bold")
    save(fig, "14_bars_supervised.png")


def fig_cross(N):
    c, u = N["cross"], N["unified"]
    groups = ["USOD -> SUIM\n(saliency masks)", "SUIM -> USOD\n(two different labels)", "SUIM -> USOD\n(one subject rule)", "USOD -> SUIM\n(one subject rule)"]
    series = {"A": [c["X1_saliency"]["cross A"], c["X2_usod"]["cross A"], u["X2"]["model A"], u["X1"]["model A"]],
              "B": [c["X1_saliency"]["cross B"], c["X2_usod"]["cross B"], u["X2"]["model B"], u["X1"]["model B"]],
              "C": [c["X1_saliency"]["cross C"], c["X2_usod"]["cross C"], u["X2"]["model C"], u["X1"]["model C"]]}
    fig, ax = plt.subplots(figsize=(11, 4))
    bars(ax, groups, series, [BLUE, "#e8a05f", ORANGE])
    best = [c["X1_saliency"]["dendrogram (gap rule)"], c["X2_usod"]["dendrogram (gap rule)"], u["X2"]["dendrogram (gap rule)"], u["X1"]["dendrogram (gap rule)"]]
    for i, v in enumerate(best): ax.hlines(v, i - 0.4, i + 0.4, color=GREY, lw=2.5)
    ax.plot([], [], color=GREY, lw=2.5, label="dendrogram (best no-label)")
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    ax.set_title("Train on one dataset, test on the other: the label definition was the problem", y=1.13, fontsize=11, fontweight="bold")
    save(fig, "15_cross_dataset.png")


def fig_tank(N):
    t = N["tank"]
    names = [("k-means", "kmeans"), ("dendrogram", "dendro"), ("earlier C\n(salient)", "old_sal"), ("unified C", "uni_C"), ("strict-rule C", "new_C")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
    for ax, key, ttl, col in ((axs[0], "iou", "Fish IoU on 19 fish frames (higher is better)", GREEN), (axs[1], "false_alarm_rate", "Frames with a false alarm, of 283 empty frames (lower is better)", RED)):
        v = [t[k][key] for _, k in names]
        ax.bar([n for n, _ in names], v, color=col, width=0.65)
        for i, x in enumerate(v): ax.text(i, x + 0.015, f"{x:.2f}" if key == "iou" else f"{100 * x:.0f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1); ax.set_title(ttl, fontsize=9.5); ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Tank frames (Dhwani): numbers are provisional until the 19 draft fish masks are reviewed", fontsize=10.5, fontweight="bold")
    save(fig, "16_tank_results.png")


def fig_size(N):
    rows = N["sup_size"]
    sizes = ["small (<10%)", "medium (10-30%)", "large (>30%)"]
    keys = [("dendrogram", "dendrogram, gap rule (best label-free)"), ("A", "A: linear probe (frozen)"), ("B", "B: conv head (frozen)"), ("C", "C: + LoRA fine-tune")]
    series = {n: [next(r["iou"] for r in rows if r["ds"] == "usod" and r["size"] == s and r["method"] == k) for s in sizes] for n, k in keys}
    fig, ax = plt.subplots(figsize=(9, 3.9))
    bars(ax, ["small object\n(under 10% of image)", "medium\n(10-30%)", "large\n(over 30%)"], series, [BLUE, "#9ec5e6", "#e8a05f", ORANGE])
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title("USOD test: the gain from supervision is largest for small objects", y=1.13, fontsize=11, fontweight="bold")
    save(fig, "17_iou_by_object_size.png")


# ----------------------------------------------------------------------------------------- copied photo figures
COPIES = {  # new name: source
    "02_pca_rgb_stride14_vs_3.png": ROOT / "results/doc_figure_sources/stride3_usod_test5.png",
    "05_gmm_concept.png": R3 / "gmm_concept.png",
    "07_supervised_pipeline.png": ROOT / "results/doc_figure_sources/supervised_pipeline.png",
    "11_compare_unsupervised.png": R3 / "cmp_unsup.png",
    "12_compare_supervised.png": R3 / "cmp_sup.png",
    "18_dhwani_frames.png": R3 / "dh_frames.png",
    "19_dhwani_unsupervised.png": R3 / "dh_unsup.png",
    "20_dhwani_supervised.png": R3 / "dh_sup.png",
    "m1_review1_pca_kmeans2.png": R3 / "pair_review1.png",
    "m2_kmeans.png": R3 / "pair_kmeans.png",
    "m3_dendrogram.png": R3 / "pair_dendro.png",
    "m4_gmm_per_image.png": R3 / "pair_gmm_pi.png",
    "m5_gmm_dataset_level.png": R3 / "pair_gmm_gl.png",
    "m6_stageA_linear_probe.png": R3 / "pair_stageA.png",
    "m7_stageB_conv_head.png": R3 / "pair_stageB.png",
    "m8_stageC_lora.png": R3 / "pair_stageC.png",
    "m9_stageD_sam.png": R3 / "pair_stageD.png",
    "m10_unified_model.png": R3 / "pair_unified.png",
    "w1_worked_example_usod.png": ROOT / "results/doc_figure_sources/worked_example_usod.png",
    "w2_worked_example_suim.png": ROOT / "results/doc_figure_sources/worked_example_suim.png",
    "a1_dendrogram_vs_kmeans_example.png": ROOT / "results/doc_figure_sources/dendro_compare_usod_test153.png",
    "a2_oracle_vs_border_rule.png": ROOT / "results/doc_figure_sources/analysis/oracle_vs_border.png",
    "a3_kmeans_method_comparison.png": ROOT / "results/doc_figure_sources/analysis/method_comparison.png",
    "a4_failure_breakdown.png": ROOT / "results/doc_figure_sources/analysis/failure_breakdown.png",
}


def copy_photos():
    for new, src in COPIES.items():
        if new == "07_supervised_pipeline.png" and src.parent == OUT:
            tmp = OUT / "_pipe.png"; shutil.copy(src, tmp); shrink(tmp, new); tmp.unlink(); continue
        if not src.exists():
            print("MISSING source (skipped):", src); continue
        shrink(src, new)


if __name__ == "__main__":
    N = load_numbers()
    fig_overview(); fig_border_rule(); fig_dendrogram(); fig_splits(); fig_lora(); fig_subject_rule()
    fig_dataflow(); fig_timeline(); fig_repro_order(); fig_data_layout()
    fig_unsup_bars(N); fig_bic(N); fig_sup_bars(N); fig_cross(N); fig_tank(N); fig_size(N)
    copy_photos()
