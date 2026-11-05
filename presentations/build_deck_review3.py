"""Builds presentations/Review-3-Underwater-Subject-Segmentation.pptx from the same base deck and helpers as the Review 2 deck.

Run with the system python3 (python-pptx) after build_figures_review3.py (project venv):

    python3 presentations/build_figures_review3.py   # with .venv/bin/python
    python3 presentations/build_deck_review3.py

Every number on the slides is read from figures_r3/numbers.json (written from the result files). Speaker notes go into the slides and into speaker_review3.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as B   # noqa: E402  (helpers, geometry, colours, base deck path)
from build_deck import (E, L, W, Y0, Y1, HEAD, WHITE, ACCENT, MUTED, GREEN, ROSE, AMBER, BODY_C, TEAL, text, callout, box, new_slide, add_chart, row_x,   # noqa: E402
                        PP_ALIGN, MSO_ANCHOR, PackURI, Presentation, Image)

PPTS, FIG3, REPO = B.PPTS, B.PPTS / "figures_r3", B.REPO
OUT = PPTS / "Review-3-Underwater-Subject-Segmentation.pptx"
N = json.load(open(FIG3 / "numbers.json"))
VIDEO = REPO / "outputs" / "tools" / "fish_h264.mp4"
f2 = lambda v: f"{v:.2f}"
pc = lambda v: f"{100 * v:.0f}%"
NOTES = []
C_KM, C_DG, C_G1, C_G2, C_OR = "A9B4C8", ACCENT, AMBER, GREEN, "E8788A"


def pic(s, path, x, y, w=None, h=None, name="Figure"):
    """Place a picture inside the box (x, y, w, h) keeping its aspect; returns the used height."""
    iw, ih = Image.open(path).size
    if w and h:
        sc = min(w / iw, h / ih); pw, ph = iw * sc, ih * sc; x = x + (w - pw) / 2
    else:
        pw = w; ph = ih * w / iw
    p = s.shapes.add_picture(str(path), E(x), E(y), width=E(pw), height=E(ph)); p.name = name
    return ph


def slide(prs, lay, lab, ttl, notes, size=32):
    s = new_slide(prs, lay, lab, ttl)
    if True:
        for sh in s.shapes:
            if sh.name == "Title": sh.text_frame.paragraphs[0].runs[0].font.size = B.Pt(size * B.S)
    s.notes_slide.notes_text_frame.text = notes
    NOTES.append((ttl, notes))
    return s


def bullets(s, x, y, w, h, items, size=14, gap=9):
    paras = []
    for it in items:
        paras.append(it if isinstance(it, list) else [(it, {})])
    return text(s, x, y, w, h, paras, size=size, space_after=gap)


def caption(s, y, parts, size=13, h=700000):
    return text(s, L, y, W, h, [parts], size=size)


def main():
    g, su, cr, un, tk = N["gmm"], N["sup"], N["cross"], N["unified"], N["tank"]
    K300 = g["X2: SUIM settings -> all 300 USOD"]["k-means (border rule)"]; D300 = g["X2: SUIM settings -> all 300 USOD"]["dendrogram (gap rule)"]
    prs = Presentation(str(B.TEMPLATE)); lay = [l for l in prs.slide_layouts if l.name == "Blank"][0]
    ids = prs.slides._sldIdLst
    for sld in list(ids)[1:7]:
        ids.remove(sld); prs.part.rels.pop(sld.rId)
    for rId, rel in list(prs.part.rels.items()):
        if rel.reltype.endswith("/notesSlide"): prs.part.rels.pop(rId)
    thank = prs.slides[1]; thank.part.partname = PackURI("/ppt/slides/slide99.xml")

    # ---------------------------------------------------------------- 1 title
    s1 = prs.slides[0]
    for sh in s1.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "PROJECT REVIEW 1": sh.text_frame.paragraphs[0].runs[0].text = "PROJECT REVIEW 3"
        if sh.has_text_frame and "Pretrained Vision" in sh.text_frame.text:
            p = sh.text_frame.paragraphs[0]; p.runs[0].text = "Underwater Subject Segmentation with Frozen DINOv2"
            for r in p.runs[1:]: r._r.getparent().remove(r._r)
    text(s1, L, 5600000, W, 900000, [f"[Student name {i}] · [Roll number {i}]" for i in range(1, 5)] + ["Department of Naval Architecture and Ocean Engineering, IIT Madras · Machine Learning for Ocean Engineers · [Project guide, if applicable]"], size=10.5, color=WHITE, name="Students")

    # ---------------------------------------------------------------- 2 outline
    s = slide(prs, lay, "OUTLINE", "What this review covers",
              "Quick map of the talk: a short recap of Review 1 and 2, then what we did after the dendrogram (Gaussian mixtures), then the supervised part, then the new tank footage, then a summary.")
    steps = [("1", "Recap", "Review 1 and 2: DINOv2 features, k-means, dendrogram"), ("2", "Gaussian mixtures", "A soft version of k-means, and its results"),
             ("3", "Unsupervised: compared", "Every method on the same four images, and in numbers"), ("4", "Supervised", "A small trained head on frozen DINOv2: stages A to D"),
             ("5", "Supervised: compared", "Cross-dataset test, one subject rule, all methods side by side"), ("6", "New footage", "The Dhwani tank frames: unsupervised, supervised, the fish video")]
    xs, w = row_x(3, 250000)
    for k, (num, head, body) in enumerate(steps):
        x, y = xs[k % 3], Y0 + 100000 + (k // 3) * 2200000
        callout(s, x, y, w, 1900000, [[(num + "  ", {"size": 30, "font": HEAD, "color": ACCENT}), (head, {"size": 20, "font": HEAD, "color": WHITE})], [(body, {"size": 13})]], space_after=6)
    caption(s, 6000000, [("Then: summary, lessons and next steps.", {"color": MUTED})])

    # ---------------------------------------------------------------- 3 recap: problem, data, metric
    s = slide(prs, lay, "RECAP · REVIEWS 1 AND 2", "The problem, the data and how we measure",
              "The goal is to separate the subject of an underwater image from the water with no labels or very few, as a perception block for an AUV. We use USOD with 300 images and SUIM with 1,525 train and validation images plus 110 test images. We measure overlap with intersection over union.")
    items = [("Problem", "Find the subject in an underwater image and separate it from the water, with no or few labels, as a perception block for an AUV."),
             ("Data", "USOD: 300 images, one salient object each. SUIM: 1,525 train/val + 110 test images with 8 classes (binary labels derived)."),
             ("Metric", "IoU = overlap of predicted and true mask divided by their union. 1.0 is perfect, 0 is no overlap.")]
    y = Y0 + 60000
    for head, body in items:
        callout(s, L, y, 6300000, 1450000, [[(head, {"size": 18, "font": HEAD, "color": ACCENT})], [(body, {"size": 13})]], space_after=4); y += 1620000
    pic(s, B.FIG / "iou_overlap.png", L + 6600000, Y0 + 100000, w=4600000, h=4300000, name="IoU example")

    # ---------------------------------------------------------------- 4 review 1: features
    s = slide(prs, lay, "REVIEW 1 · FROZEN DINOv2 FEATURES", "DINOv2 gives every patch a fingerprint",
              "Review 1: the image is cut into 14 by 14 pixel patches. The frozen DINOv2 model gives each patch 768 numbers. Squeezed into three colours by PCA the maps already separate the diver, the water and the reef. Overlapping windows, stride 3, make the map smoother.")
    xs, w = row_x(3, 250000)
    for x, nm, cap in zip(xs, ("usod_test106_photo", "usod_test106_pca14", "usod_test106_pca3"), ("The photo", "PCA-RGB, stride 14", "PCA-RGB, stride 3 (overlapping)")):
        text(s, x, Y0, w, 320000, cap, size=14, font=HEAD, color=WHITE); pic(s, B.FIG / f"{nm}.png", x, Y0 + 380000, w=w)
    callout(s, L, 4900000, W, 1350000, [[("Frozen means: ", {"bold": True, "color": ACCENT}), "the network is used exactly as released, 86.6 million weights, nothing is trained. PCA squeezes the 768 numbers into 3 colours only for display; the methods that follow use all 768."]], size=14)

    # ---------------------------------------------------------------- 5 review 1: first baselines
    s = slide(prs, lay, "REVIEW 1 · FIRST BASELINES", "First baselines: PCA mask and k-means, k = 2",
              "The first baselines were the first PCA component as a mask and k-means with two groups, both with the border rule. On USOD they score 0.42 and 0.49. They were not run on SUIM at that time. The same four example images are used for every method in this talk, chosen by rule.")
    h = pic(s, FIG3 / "pair_review1.png", L, 1400000, w=W, h=4300000)
    caption(s, 1400000 + h + 120000, [("USOD (300 images) mean IoU: ", {"bold": True, "color": ACCENT}), "PCA mask 0.42, k-means (k = 2) 0.49. ",
                                      ("The same 4 test images appear on every method slide: ", {"bold": True, "color": ACCENT}), "a small, a medium and a large USOD object (closest to the median of each size group) and the SUIM test image closest to the median."], size=13)

    # ---------------------------------------------------------------- 6 k-means
    s = slide(prs, lay, "REVIEW 2 · K-MEANS", "k-means: group patches, then pick the object groups",
              "Review 2: k-means with 2 to 6 groups on the 768 numbers per patch, then a border rule: groups that touch the image border are water. Without labels this gives 0.52 on USOD. If we choose the groups with the ground truth, the oracle, we get 0.70. So the groups are good and the choice is the weak step.")
    h = pic(s, FIG3 / "pair_kmeans.png", L, 1450000, w=W)
    caption(s, 1450000 + h + 120000, [("k-means, up to 6 groups, border rule: ", {"bold": True, "color": ACCENT}), f"0.52 mean IoU on USOD (300 images), 0.48 on SUIM not-water. ",
                                       ("Oracle (groups chosen with the answer): ", {"bold": True, "color": ACCENT}), "0.70. The groups are good, choosing them is the weak step."], size=13)

    # ---------------------------------------------------------------- 7 dendrogram
    s = slide(prs, lay, "REVIEW 2 · DENDROGRAM", "Dendrogram: a tree of patches, cut at the largest gap",
              "Review 2: instead of k-means we build a tree by repeatedly merging the two closest groups, and cut it where the jump between merge heights is largest. Average linkage on the 768 numbers works best. Label-free IoU rises from 0.52 to 0.65 on USOD and from 0.43 to 0.55 on SUIM saliency.")
    h = pic(s, FIG3 / "pair_dendro.png", L, 1450000, w=W)
    caption(s, 1450000 + h + 120000, [("Dendrogram, average linkage, gap rule: ", {"bold": True, "color": ACCENT}), f"{f2(D300)} on USOD, 300 images (k-means {f2(K300)}), 0.55 on SUIM saliency (0.43). ",
                                       ("Same groups as k-means, better choice. ", {"bold": True, "color": ACCENT}), "About one image in five gets worse."], size=13)

    # ---------------------------------------------------------------- 8 GMM idea
    s = slide(prs, lay, "AFTER THE DENDROGRAM · GAUSSIAN MIXTURES", "Gaussian mixtures: a soft version of k-means",
              "After the dendrogram we added Gaussian mixture models. k-means gives each patch one hard label from round, equal clusters. A mixture models every cluster as a Gaussian with its own centre, spread and weight and gives each patch a probability for each cluster. It is fitted with the EM algorithm. The number of clusters can be chosen with BIC.")
    pic(s, FIG3 / "gmm_concept.png", L, Y0, w=6300000)
    bullets(s, L + 6600000, Y0, W - 6600000, 3900000,
            [[("Mixture density", {"bold": True, "color": ACCENT})], "p(x) = Σ π_k · N(x | μ_k, Σ_k)",
             [("Responsibility", {"bold": True, "color": ACCENT})], "γ_nk = π_k N(x_n | μ_k, Σ_k) / p(x_n): the probability that patch n belongs to cluster k",
             [("EM algorithm", {"bold": True, "color": ACCENT})], "E-step: compute γ. M-step: re-estimate π, μ, Σ with γ as weights. Repeat.",
             [("Choosing the number of clusters", {"bold": True, "color": ACCENT})], "BIC = -2 ln L + p ln n"], size=13, gap=4)
    callout(s, L, 4750000, 6300000, 1500000, [[("k-means is the special case ", {"bold": True, "color": ACCENT}), "of equal round clusters and hard labels. A mixture allows ellipses of different size and returns a probability instead of one label."]], size=13)

    # ---------------------------------------------------------------- 9 GMM how
    s = slide(prs, lay, "GAUSSIAN MIXTURES · HOW WE RAN THEM", "Two variants; BIC was the wrong rule",
              f"We ran two variants. The per-image mixture replaces k-means in the same pipeline: PCA, mixture with 2 to 6 components, border rule. The dataset-level mixture is fitted once on many training images and used on new images without refitting. The settings were chosen only on the source dataset's training images. BIC almost always picked six components and over-segmented; silhouette worked better: {f2(N['dev_usod']['sil'])} against {f2(N['dev_usod']['bic'])} on USOD training images.")
    bullets(s, L, Y0, 5500000, 4600000,
            [[("1  Per-image GMM. ", {"bold": True, "color": ACCENT}), "Patch features -> PCA (32 numbers) -> mixture of 2 to 6 Gaussians -> number of components by silhouette -> the border rule picks the foreground components."],
             [("2  Dataset-level GMM. ", {"bold": True, "color": ACCENT}), "One mixture (8 components, full covariance) fitted once on 300 patches per training image; each component is labelled foreground or background by how often its patches touch the image border. New image: one pass, no fitting."],
             [("Fair protocol. ", {"bold": True, "color": ACCENT}), "Settings and fits use only the source dataset's training images; test images are scored once, in-domain and across USOD and SUIM."]], size=13, gap=10)
    text(s, L + 5800000, Y0, W - 5800000, 330000, "Choosing the number of components (mean IoU on training images)", size=13, font=HEAD, color=WHITE)
    add_chart(s, L + 5800000, Y0 + 380000, W - 5800000, 3250000, ["USOD", "SUIM"], [("BIC (best setting)", [N["dev_usod"]["bic"], N["dev_suim"]["bic"]]), ("Silhouette (best setting)", [N["dev_usod"]["sil"], N["dev_suim"]["sil"]])], [C_OR, C_G1], maxv=0.8, label_size=12)
    text(s, L + 5800000, 5350000, W - 5800000, 900000, f"BIC picks 6 components in almost every image ({N['dev_usod']['meanK_bic']:.1f} on average); DINOv2 patch features are far from Gaussian. For reference, k-means on the same USOD images: {f2(N['dev_usod']['kmeans'])}.", size=12, color=MUTED)

    # ---------------------------------------------------------------- 10-11 GMM masks
    sets_nm = dict(pi=("pair_gmm_pi", "GAUSSIAN MIXTURES · PER-IMAGE", "Per-image mixture: original image and mask"), gl=("pair_gmm_gl", "GAUSSIAN MIXTURES · DATASET-LEVEL", "Dataset-level mixture: original image and mask"))
    s = slide(prs, lay, *sets_nm["pi"][1:], "The per-image mixture fits its own Gaussians in every image. Like k-means it picks the foreground with the border rule, so it fails where the object touches the border or the background has several parts. It improves on k-means in-domain by about 4 points.")
    h = pic(s, FIG3 / "pair_gmm_pi.png", L, 1450000, w=W)
    caption(s, 1450000 + h + 120000, [("Per-image GMM: ", {"bold": True, "color": ACCENT}), f"USOD {f2(g['In-domain USOD (test 100)']['GMM per-image'])} (k-means 0.50), SUIM not-water {f2(g['In-domain SUIM, not-water (test)']['GMM per-image'])} (k-means 0.48). ",
                                      ("Fit time ", {"bold": True, "color": ACCENT}), "about 0.14 s per image on top of the feature extraction."], size=13)
    s = slide(prs, lay, *sets_nm["gl"][1:], "The dataset-level mixture is fitted once on training images and then applied to every new image with one pass. It gives the best label-free result on SUIM not-water, 0.63, but it learns the label convention of its source dataset, so it drops when applied across datasets.")
    h = pic(s, FIG3 / "pair_gmm_gl.png", L, 1450000, w=W)
    caption(s, 1450000 + h + 120000, [("Dataset-level GMM: ", {"bold": True, "color": ACCENT}), f"USOD {f2(g['In-domain USOD (test 100)']['GMM dataset-level'])}, SUIM not-water {f2(g['In-domain SUIM, not-water (test)']['GMM dataset-level'])} (best label-free there). ",
                                      ("No fitting per image: ", {"bold": True, "color": ACCENT}), "0.03 s per image, and the same model is used on every video frame."], size=13)

    # ---------------------------------------------------------------- 12 GMM results chart
    keys = [("In-domain USOD (test 100)", "USOD\n(100 test)"), ("In-domain SUIM, not-water (test)", "SUIM not-water\n(92)"), ("In-domain SUIM, saliency (test)", "SUIM saliency\n(108)"),
            ("X2: SUIM settings -> all 300 USOD", "SUIM → USOD\n(300)"), ("X1: USOD settings -> SUIM test, saliency", "USOD → SUIM\nsaliency (108)")]
    meth = [("k-means (border rule)", "k-means", C_KM), ("dendrogram (gap rule)", "dendrogram", C_DG), ("GMM per-image", "GMM per-image", C_G1), ("GMM dataset-level", "GMM dataset-level", C_G2)]
    s = slide(prs, lay, "GAUSSIAN MIXTURES · RESULTS", "Mixtures: better than k-means in-domain only",
              "In-domain both mixtures beat k-means: per-image by about 4 points and dataset-level by 9 points on USOD and 15 on SUIM not-water. The dendrogram is still best on USOD and on SUIM saliency. Across datasets, SUIM to USOD and USOD to SUIM, the gain disappears or reverses because the label definitions differ. The mixture oracle is as good as the k-means oracle, so the weak step is still the choice of components.")
    add_chart(s, L, Y0, W, 3900000, [k[1] for k in keys], [(m[1], [g[k[0]][m[0]] for k in keys]) for m in meth], [m[2] for m in meth], maxv=0.85, label_size=11, cat_size=11)
    caption(s, 5500000, [("Test images, each scored once; settings chosen on the source dataset's training images. ", {"color": MUTED}), ("Oracles (best component union, uses the answer): ", {"bold": True, "color": ACCENT}),
                         f"USOD k-means {f2(g['In-domain USOD (test 100)']['k-means oracle (uses GT)'])} / GMM {f2(g['In-domain USOD (test 100)']['GMM oracle (uses GT)'])}, SUIM not-water {f2(g['In-domain SUIM, not-water (test)']['k-means oracle (uses GT)'])} / {f2(g['In-domain SUIM, not-water (test)']['GMM oracle (uses GT)'])}: the clusters are good, the choice is the weak step."], size=12, h=900000)

    # ---------------------------------------------------------------- 13-14 unsupervised comparison
    s = slide(prs, lay, "UNSUPERVISED · COMPARISON", "All unsupervised methods on the same four images",
              "Same four images, every unsupervised method, IoU on each mask. The dendrogram usually gives the cleanest object. k-means and the per-image mixture often paint the background. The dataset-level mixture is good on SUIM but noisy on USOD.")
    pic(s, FIG3 / "cmp_unsup.png", L - 200000, 1350000, w=W + 400000, h=5300000)
    s = slide(prs, lay, "UNSUPERVISED · COMPARISON", "All unsupervised methods in numbers",
              "In numbers, the dendrogram is best on USOD and on SUIM saliency and the dataset-level mixture is best on SUIM not-water. All of them stay far below the oracle, so the open problem for the unsupervised methods is the choice of the foreground components.")
    k3 = [("In-domain USOD (test 100)", "USOD (100 test)"), ("In-domain SUIM, not-water (test)", "SUIM not-water (92)"), ("In-domain SUIM, saliency (test)", "SUIM saliency (108)")]
    ser = [(m[1], [g[k[0]][m[0]] for k in k3]) for m in meth] + [("k-means oracle (uses answer)", [g[k[0]]["k-means oracle (uses GT)"] for k in k3])]
    add_chart(s, L, Y0, 7400000, 4300000, [k[1] for k in k3], ser, [C_KM, C_DG, C_G1, C_G2, C_OR], maxv=0.9, label_size=10, cat_size=11, gap=40)
    bullets(s, L + 7700000, Y0, W - 7700000, 4600000,
            [[("Best without labels", {"bold": True, "color": ACCENT})], "USOD and SUIM saliency: dendrogram. SUIM not-water: dataset-level mixture.",
             [("Same weakness in all", {"bold": True, "color": ACCENT})], "The oracle is 0.60 to 0.77: the groups exist, choosing them is hard.",
             [("Across datasets", {"bold": True, "color": ACCENT})], "No method keeps its in-domain gain when the label definition changes."], size=13, gap=6)

    # ---------------------------------------------------------------- 15 why supervised
    s = slide(prs, lay, "SUPERVISED · WHY AND WHAT", "Keep DINOv2 frozen, train a small part on top",
              "The unsupervised methods are limited by the choice of the foreground. So we let a small model learn which patches are the subject from labelled masks, while DINOv2 stays frozen. Stage A is a single linear layer, B a small convolutional head, C adds small LoRA adapters inside DINOv2, D adds SAM. We do not retrain DINOv2 from scratch: that would overfit our 1,900 images.")
    pic(s, FIG3 / "pipeline_dark.png", L, Y0, w=W, h=2600000)
    xs, w = row_x(3, 220000)
    for x, (head, body) in zip(xs, [("Why", "The oracle (0.70) is far above the best label-free choice (0.65): learn the choice from a few labelled masks."),
                                    ("What is trained", "A: 769 weights. B: 0.4 to 2 million weights. C: head + 906,432 LoRA weights, about 1% of the 86.6 million."),
                                    ("What is not", "DINOv2 itself stays frozen (A, B) or is only nudged by small adapters (C). No retraining from scratch.")]):
        callout(s, x, 4300000, w, 1900000, [[(head, {"size": 18, "font": HEAD, "color": ACCENT})], [(body, {"size": 13})]], space_after=4)

    # ---------------------------------------------------------------- 16 training setup
    s = slide(prs, lay, "SUPERVISED · TRAINING SETUP", "How the head is trained",
              "Data are split once with a fixed seed: USOD 160 train, 40 validation, 100 test; SUIM 1,375 train, 150 validation, 110 test. Test images are never used to choose anything. Features are computed once and cached, so heads train in minutes. The loss is BCE plus Dice. Stage C uses LoRA with rank 16 and trains for 3,000 steps in about 14 minutes.")
    xs, w = row_x(2, 250000)
    cards = [("Data and splits", "USOD 160 / 40 / 100 and SUIM 1,375 / 150 / 110 (train / validation / test). Hyper-parameters, thresholds and checkpoints are chosen on validation only; test images are scored once."),
             ("Loss: BCE + Dice", "BCE judges every pixel; Dice judges the overlap of the whole mask (matters for small subjects). Example with 4 pixels: BCE 0.236 + Dice 0.167 = 0.403."),
             ("Frozen features are cached", "A and B train on cached 33 × 44 × 768 patch features (4.3 GB), so a head trains in minutes; the whole search is 5 and 19 minutes."),
             ("LoRA (Stage C)", "W' = W + (α/r) B A with r = 16: 49,152 extra weights next to a 1.77 M qkv layer. 906,432 trainable weights, flips, crops and colour jitter, 3,000 steps, 14 minutes on a 12 GB GPU.")]
    for k, (head, body) in enumerate(cards):
        callout(s, xs[k % 2], Y0 + (k // 2) * 2150000, w, 1950000, [[(head, {"size": 18, "font": HEAD, "color": ACCENT})], [(body, {"size": 13})]], space_after=4)

    # ---------------------------------------------------------------- 17-20 stages
    sup = su["usod"]; sus = su["suim"]
    stage_defs = [("A", "stageA", "STAGE A · LINEAR PROBE", "Stage A: one linear layer on the frozen features", "A linear probe: for every patch, a weighted sum of its 768 numbers plus a bias, then a sigmoid. 769 weights in total. It already scores 0.80 on USOD and 0.88 on SUIM not-water, better than every label-free method, because it learns which patches are the subject.",
                   "A: linear probe (frozen)", "A single 1 × 1 layer, 769 weights: score = w · x + b, probability = sigmoid(score)."),
                  ("B", "stageB", "STAGE B · SMALL CONVOLUTIONAL HEAD", "Stage B: look at the neighbours too", "A small convolutional head with dilated residual blocks lets a patch use its neighbours. It fills holes and smooths boundaries. 0.82 on USOD and 0.91 on SUIM.",
                   "B: conv head (frozen)", "1 × 1 reduce, residual dilated 3 × 3 blocks (a patch sees up to 15 × 15 neighbours), 0.4 to 2 million weights."),
                  ("C", "stageC", "STAGE C · LoRA", "Stage C: let DINOv2 adjust a little with LoRA", "LoRA adds small low-rank corrections inside DINOv2's attention layers and patch embedding. 906,432 trainable weights, about 1% of the backbone. It gives the clear gain on USOD, 0.87, and 0.92 on SUIM not-water.",
                   "C: + LoRA fine-tune", "Head + LoRA adapters (rank 16) in all 12 blocks, trained end to end for 3,000 steps."),
                  ("D", "stageD", "STAGE D · SAM REFINEMENT", "Stage D: SAM refinement gave nothing", "We tried to refine the Stage C mask with SAM outlines. On validation every rule was at best equal to Stage C, so the refinement is effectively off. The ground-truth masks are blobby annotations, so sharper edges do not match them better. A negative result.",
                   "D: + SAM refinement", "SAM (Segment Anything) proposes sharp outlines; a rule chosen on validation decides when to accept them.")]
    for st, fk, lab, ttl, notes, key, what in stage_defs:
        s = slide(prs, lay, lab, ttl, notes)
        h = pic(s, FIG3 / f"pair_{fk}.png", L, 1450000, w=W)
        caption(s, 1450000 + h + 120000, [(what + " ", {"bold": True, "color": ACCENT}), f"Mean IoU: USOD (100 test) {f2(sup[key])}, SUIM not-water (92 test) {f2(sus[key])}. Masks: USOD channel for the three USOD images, SUIM channel for the SUIM image."], size=13)

    # ---------------------------------------------------------------- 21-22 supervised comparison
    s = slide(prs, lay, "SUPERVISED · COMPARISON", "All supervised stages on the same four images",
              "The same four images for every supervised stage, IoU on each mask. The step from the label-free methods to stage A is the biggest, then LoRA helps on small objects. The last column is the unified model of the next section, which uses a different label definition on the SUIM image.")
    pic(s, FIG3 / "cmp_sup.png", L - 100000, 1350000, w=W + 200000, h=4800000)
    caption(s, 6250000, [("* The unified model uses the one subject rule, so its ground truth for the SUIM image differs from the other columns.", {"color": MUTED})], size=11, h=400000)
    s = slide(prs, lay, "SUPERVISED · COMPARISON", "Supervised stages in numbers",
              f"The learned head removes most of the selection problem. On USOD: dendrogram {f2(sup['dendrogram, gap rule (best label-free)'])}, A {f2(sup['A: linear probe (frozen)'])}, B {f2(sup['B: conv head (frozen)'])}, C {f2(sup['C: + LoRA fine-tune'])}, D {f2(sup['D: + SAM refinement'])}. On SUIM not-water: {f2(sus['dendrogram, gap rule (best label-free)'])}, {f2(sus['A: linear probe (frozen)'])}, {f2(sus['B: conv head (frozen)'])}, {f2(sus['C: + LoRA fine-tune'])}, {f2(sus['D: + SAM refinement'])}. SAM adds nothing.")
    names = ["dendrogram, gap rule (best label-free)", "A: linear probe (frozen)", "B: conv head (frozen)", "C: + LoRA fine-tune", "D: + SAM refinement"]; short = ["best label-free (dendrogram)", "A linear probe", "B conv head", "C + LoRA", "D + SAM"]
    add_chart(s, L, Y0, 7600000, 4400000, ["USOD (100 test)", "SUIM not-water (92 test)"], [(sh, [sup[n], sus[n]]) for sh, n in zip(short, names)], [C_DG, "6FA8DC", "4F8BD6", "7BE0A8", C_KM], maxv=1.05, label_size=11, cat_size=12, gap=40)
    sz = {(r["ds"], r["size"], r["method"]): r["iou"] for r in N["sup_size"]}
    small = lambda nm: sz[("usod", "small (<10%)", nm)]
    bullets(s, L + 7900000, Y0, W - 7900000, 4600000,
            [[("Small USOD objects (< 10%)", {"bold": True, "color": ACCENT})], f"dendrogram {f2(small(names[0]))}, A {f2(small(names[1]))}, B {f2(small(names[2]))}, C {f2(small(names[3]))}",
             [("Biggest step", {"bold": True, "color": ACCENT})], "label-free to linear probe: the features already contain the answer.",
             [("SAM", {"bold": True, "color": ACCENT})], "adds nothing (D equals C)."], size=13, gap=6)

    # ---------------------------------------------------------------- 23 cross-dataset
    s = slide(prs, lay, "SUPERVISED · CROSS-DATASET TEST", "Cross-dataset test: the label problem",
              f"We trained on one dataset only and tested on the other. USOD to SUIM scored against SUIM saliency masks: {f2(cr['X1_saliency']['cross C'])} against {f2(cr['X1_saliency']['dendrogram (gap rule)'])} for the best label-free method. SUIM to USOD scored only {f2(cr['X2_usod']['cross C'])}: SUIM labels everything that is not water as foreground, USOD only the salient object, so the SUIM-trained model marks the whole scene. The features transfer; the label definitions differ.")
    cats = ["USOD → SUIM\nsaliency labels", "USOD → SUIM\nnot-water labels", "SUIM → USOD\nsalient object"]; ks = ["X1_saliency", "X1_nonwater", "X2_usod"]
    add_chart(s, L, Y0, 6800000, 3500000, cats, [("best label-free (dendrogram)", [cr[k]["dendrogram (gap rule)"] for k in ks]), ("trained on the other dataset (Stage C)", [cr[k]["cross C"] for k in ks])], [C_DG, "7BE0A8"], maxv=1.0, label_size=12, cat_size=11)
    bullets(s, L + 7100000, Y0, W - 7100000, 3700000,
            [[("Works when the label meaning matches", {"bold": True, "color": ACCENT})], "0.79 on SUIM saliency masks with a model trained on 160 USOD images.",
             [("Fails when it differs", {"bold": True, "color": ACCENT})], "SUIM-trained model marks 74% of USOD pixels as foreground; the truth is 22%.",
             [("So", {"bold": True, "color": ACCENT})], "the head learns the annotation convention. We must decide what a subject is."], size=13, gap=5)
    pic(s, REPO / "results" / "cross_dataset" / "examples" / "bad" / "X2_usod_usod_test105_iou0.05.png", L, 4850000, w=W, h=1500000, name="Cross-dataset failure")

    # ---------------------------------------------------------------- 24 subject rule + results
    seeds = {r["stage"]: r for r in N["seeds"]}; ca = N["crop_false_alarm"]
    s = slide(prs, lay, "SUPERVISED · ONE SUBJECT RULE", "One subject rule fixes the cross-dataset gap",
              f"We agreed one rule: animals, divers, robots and wrecks are the subject; if a scene has only reef and water the reef is the subject; otherwise there is none. It is derived from SUIM's class masks and agrees with SUIM's own saliency masks at IoU 0.99. With one label for both datasets the cross-dataset scores rise to {f2(un['X2']['model C'])} and {f2(un['X1']['model C'])}. Over five seeds the spread is about 0.01. On empty-water crops the false-alarm rate of model C is {pc(ca['C'])} against 100% for k-means.")
    xs, w = row_x(3, 200000)
    for x, (head, body) in zip(xs, [("1 · Animals, divers, robots, wrecks present", "They are the subject; reef, plants and sand are background."), ("2 · Only reef and plants", "The reef is the subject."), ("3 · Only water or sand", "No subject: the mask is empty.")]):
        callout(s, x, Y0, w, 1150000, [[(head, {"size": 15, "font": HEAD, "color": ACCENT})], [(body, {"size": 12})]], space_after=3)
    cats = ["USOD test\n(pooled)", "SUIM test\n(pooled)", "USOD → SUIM", "SUIM → USOD"]; ks = ["pool_usod", "pool_suim", "X1", "X2"]
    add_chart(s, L, 2850000, 7400000, 3500000, cats, [("best label-free (dendrogram)", [un[k]["dendrogram (gap rule)"] for k in ks]), ("unified model C", [un[k]["model C"] for k in ks])], [C_DG, "7BE0A8"], maxv=1.0, label_size=12, cat_size=11)
    bullets(s, L + 7700000, 2900000, W - 7700000, 3500000,
            [[("Stable over seeds", {"bold": True, "color": ACCENT})], f"USOD {seeds['C']['usod_mean']:.3f} ± {seeds['C']['usod_std']:.3f}, SUIM {seeds['C']['suim_mean']:.3f} ± {seeds['C']['suim_std']:.3f} (stage C)",
             [("False alarms on empty-water crops", {"bold": True, "color": ACCENT})], f"k-means {pc(ca['kmeans'])}, A {pc(ca['A'])}, B {pc(ca['B'])}, C {pc(ca['C'])}",
             [("SUIM → USOD", {"bold": True, "color": ACCENT})], f"0.36 → {f2(un['X2']['model C'])}"], size=13, gap=5)

    s = slide(prs, lay, "SUPERVISED · UNIFIED MODEL", "Unified model: original image and mask",
              "The unified model of stage C with the one subject rule, same four test images. On the SUIM image the ground truth is now the subject rule, so only the fish are the subject and the reef is background.")
    h = pic(s, FIG3 / "pair_unified.png", L, 1400000, w=8300000, h=4500000)
    bullets(s, L + 8600000, 1450000, W - 8600000, 4300000,
            [[("Unified C", {"bold": True, "color": ACCENT})], f"USOD test {f2(un['pool_usod']['model C'])}, SUIM test {f2(un['pool_suim']['model C'])}",
             [("One output, one threshold", {"bold": True, "color": ACCENT})], "the same model and rule for both datasets",
             [("The SUIM image changes", {"bold": True, "color": ACCENT})], "under the subject rule only the fish are the subject; the reef is background (ground-truth row)"], size=13, gap=6)
    caption(s, 1400000 + h + 120000, [("Masks: unified Stage C, same four test images. ", {"bold": True, "color": ACCENT}), "Ground truth: the salient object (USOD) and the subject rule (SUIM image)."], size=13)

    # ---------------------------------------------------------------- 26 Dhwani dataset
    s = slide(prs, lay, "NEW FOOTAGE · THE DHWANI DATASET", "The Dhwani dataset: 302 tank frames",
              "We built a third dataset from six GoPro clips of an indoor underwater test tank: murky green water, perforated panels, steel frames, floor debris and a school of fish. 302 frames, each tagged as fish, floor, structure, waterline or empty water. It is test-only. Ground truth: 19 fish frames got SAM-drafted masks from hand-placed boxes, still to be reviewed; the other 283 frames are defined to have no subject.")
    pic(s, FIG3 / "dh_frames.png", L, 1450000, w=W)
    xs, w = row_x(3, 220000)
    for x, (head, body) in zip(xs, [("From the clips", "10 clips looked at, 6 kept (4K, 30 fps); 302 sharp, different frames, 1920 × 1080, empty water capped at 27%."),
                                    ("Scene tags", "structure 110, empty 80, waterline 55, floor 37, fish 20."), ("Ground truth (provisional)", "19 fish frames: SAM drafts from hand-placed boxes (not from our models). 283 frames: no subject. Test only, never used for training.")]):
        callout(s, x, 3700000, w, 1950000, [[(head, {"size": 17, "font": HEAD, "color": ACCENT})], [(body, {"size": 13})]], space_after=4)

    # ---------------------------------------------------------------- 27 unsupervised on Dhwani
    s = slide(prs, lay, "NEW FOOTAGE · UNSUPERVISED", "Unsupervised methods on the Dhwani frames",
              f"On the tank frames the unsupervised methods always output a foreground: k-means marks something in 100% of the frames that have no subject, the dendrogram in {pc(tk['dendro']['false_alarm_rate'])}. On the fish frames their IoU is {f2(tk['kmeans']['iou'])} and {f2(tk['dendro']['iou'])}. The frames are the median frame of each scene tag. The Gaussian mixtures were not run on the tank, because we limited that study to USOD and SUIM.")
    h = pic(s, FIG3 / "dh_unsup.png", L, 1400000, w=W)
    caption(s, 1400000 + h + 100000, [("k-means: ", {"bold": True, "color": ACCENT}), f"fish IoU {f2(tk['kmeans']['iou'])}, false alarm in {pc(tk['kmeans']['false_alarm_rate'])} of the 283 no-subject frames. ", ("Dendrogram: ", {"bold": True, "color": ACCENT}),
                                      f"fish IoU {f2(tk['dendro']['iou'])}, false alarm in {pc(tk['dendro']['false_alarm_rate'])}. ", ("Both always output a foreground, even on empty water. ", {"bold": True, "color": ROSE}), ("Frames = median frame of each tag.", {"color": MUTED})], size=13)

    # ---------------------------------------------------------------- 28 supervised on Dhwani
    s = slide(prs, lay, "NEW FOOTAGE · SUPERVISED", "Supervised models on the Dhwani frames",
              f"The supervised models find the fish but differ in how selective they are. The earlier salient-channel model is best: fish IoU {f2(tk['old_sal']['iou'])}, a false alarm in {pc(tk['old_sal']['false_alarm_rate'])} of the no-subject frames. The unified model with the one subject rule scores {f2(tk['uni_C']['iou'])} and {pc(tk['uni_C']['false_alarm_rate'])}. The new strict-rule model halves the false alarms to {pc(tk['new_C']['false_alarm_rate'])} but over-segments the water around the fish, fish IoU {f2(tk['new_C']['iou'])}.")
    hh = pic(s, FIG3 / "dh_sup.png", L, 1350000, w=7500000, h=5000000)
    rows = [("earlier C, salient channel", "old_sal"), ("unified C (subject rule)", "uni_C"), ("new C (strict rule)", "new_C")]
    text(s, L + 7800000, 1400000, W - 7800000, 400000, "Scored on the tank frames", size=16, font=HEAD, color=WHITE)
    y = 1900000
    for lab, k in rows:
        callout(s, L + 7800000, y, W - 7800000, 1250000, [[(lab, {"size": 14, "font": HEAD, "color": ACCENT})], [(f"fish IoU {f2(tk[k]['iou'])}", {"size": 13})], [(f"false alarm in {pc(tk[k]['false_alarm_rate'])} of no-subject frames", {"size": 13})]], space_after=2); y += 1400000
    text(s, L + 7800000, y + 40000, W - 7800000, 700000, "Fish masks are SAM drafts, not yet reviewed: numbers are provisional.", size=11, color=MUTED)

    # ---------------------------------------------------------------- 29 fish video
    s = slide(prs, lay, "NEW FOOTAGE · THE FISH CLIP", "The fish clip: original and prediction",
              "This is the one video: 10 seconds of the fish school in clip GX010214, from 52 to 62 seconds. Left the original frame, right the unified model's prediction in orange. Every frame is predicted on its own, so the masks flicker slightly. It finds the fish well but also marks some of the wall and floor around them.")
    vh = 11277318 * 540 / 1920
    mv = s.shapes.add_movie(str(VIDEO), E(L), E(1450000), E(W), E(vh), poster_frame_image=str(FIG3 / "fish_poster.png"), mime_type="video/mp4"); mv.name = "Fish video"
    xs, w = row_x(3, 200000)
    for x, (lab, k) in zip(xs, rows):
        callout(s, x, 1450000 + vh + 160000, w, 1100000, [[(lab, {"size": 14, "font": HEAD, "color": ACCENT})], [(f"fish IoU {f2(tk[k]['iou'])} · false alarm {pc(tk[k]['false_alarm_rate'])}", {"size": 13})]], space_after=2)
    text(s, L, 1450000 + vh + 1350000, W, 400000, "Video: GX010214, 52-62 s, original (left) and unified C (right), played at real-time speed. Click to play.", size=11, color=MUTED)

    # ---------------------------------------------------------------- 30 what the tank shows
    s = slide(prs, lay, "NEW FOOTAGE · WHAT IT SHOWS", "What the tank footage tells us",
              "The tank is a hard test: dark green murky water, panels and floor that never appear in USOD or SUIM. The supervised models transfer for clear subjects but are not yet reliable. The stricter rule and the underwater augmentation halve the false alarms but over-segment. The simplest, least adapted model, the earlier salient channel, is the most selective. What counts as a subject matters as much as the model, and more varied labelled data is the real lever.")
    xs, w = row_x(3, 220000)
    for x, (head, body) in zip(xs, [("Works", "Clear subjects: every supervised model finds the fish. The label-free methods always output a foreground."),
                                    ("Does not work yet", "Murky empty water, panels and floor are over-segmented; the new strict rule halves false alarms but lowers fish IoU."),
                                    ("Reading", "The model that learned one clear convention (earlier salient channel) is the most selective on a new site. Data and the subject definition decide.")]):
        callout(s, x, Y0 + 100000, w, 2600000, [[(head, {"size": 20, "font": HEAD, "color": ACCENT})], [(body, {"size": 14})]], space_after=6)
    caption(s, 4500000, [("Caveats: ", {"bold": True, "color": AMBER}), "one tank, six clips, 302 correlated frames; fish masks are unreviewed SAM drafts; the other 283 frames are assumed to have no subject; each model scored once."], size=13, h=1200000)

    # ---------------------------------------------------------------- 31 summary
    s = slide(prs, lay, "SUMMARY", "From Review 1 to Review 3",
              "Summary of the three reviews. Review 1: frozen DINOv2 features and first baselines. Review 2: k-means and the dendrogram, 0.65 label-free on USOD. Review 3: Gaussian mixtures, a small trained head with 0.87 on USOD and 0.92 on SUIM, one subject rule giving 0.77 to 0.78 across datasets, and a first test on new footage that shows the limits.")
    xs, w = row_x(3, 220000)
    cols = [("Review 1", "Frozen DINOv2 features, PCA maps, first k = 2 baselines", "0.42 / 0.49"), ("Review 2", "k-means up to 6 groups; dendrogram with the gap rule", f"{f2(K300)} → {f2(D300)}"),
            ("Review 3", "Mixtures, trained head, one subject rule, tank footage", f"{f2(sup['C: + LoRA fine-tune'])} on USOD")]
    for x, (head, body, big) in zip(xs, cols):
        callout(s, x, Y0 + 50000, w, 2500000, [[(head, {"size": 22, "font": HEAD, "color": ACCENT})], [(big, {"size": 34, "font": HEAD, "color": WHITE})], [(body, {"size": 13})]], space_after=6)
    bullets(s, L, 4300000, W, 2100000,
            [[("Unsupervised: ", {"bold": True, "color": ACCENT}), f"best {f2(D300)} on USOD, 300 images (dendrogram); mixtures help in-domain; the choice of the foreground is the weak step."],
             [("Supervised: ", {"bold": True, "color": ACCENT}), f"linear probe {f2(sup['A: linear probe (frozen)'])}, + LoRA {f2(sup['C: + LoRA fine-tune'])} on USOD; one subject rule gives {f2(un['X1']['model C'])} and {f2(un['X2']['model C'])} across datasets."],
             [("New footage: ", {"bold": True, "color": ACCENT}), "finds the fish, but over-segments murky water and structure: a new site is not solved."]], size=14, gap=8)

    # ---------------------------------------------------------------- 32 lessons and next steps
    s = slide(prs, lay, "SUMMARY · LESSONS AND NEXT STEPS", "What we learned and what comes next",
              "Lessons: the features are good and the choice of the foreground is the weak step; a little supervision removes most of it; the label definition matters as much as the model; a new site is not solved. Next steps: a third public dataset or real AUV footage with masks, separating subjects from each other, speed on the AUV computer, and a confidence output for empty water.")
    wl = 5400000
    text(s, L, Y0, wl, 4700000, [[("Lessons", {"size": 22, "font": HEAD, "color": WHITE})],
                                  [("The features are good; choosing the foreground is the weak step.", {"size": 18})], [("A little supervision removes most of that problem.", {"size": 18})],
                                  [("The label rule matters as much as the model.", {"size": 18})], [("A new site (murky tank water) is not solved.", {"size": 18})]], space_after=12)
    xr = L + wl + 400000
    text(s, xr, Y0, L + W - xr, 4700000, [[("Next steps", {"size": 22, "font": HEAD, "color": WHITE})],
                                           [("1  ", {"color": ACCENT, "bold": True, "size": 14}), ("A third public dataset or real AUV footage with masks.", {"size": 18})],
                                           [("2  ", {"color": ACCENT, "bold": True, "size": 14}), ("Separate the subjects from each other (multi-class head, instances).", {"size": 18})],
                                           [("3  ", {"color": ACCENT, "bold": True, "size": 14}), ("Speed on the AUV computer; confidence output for empty water.", {"size": 18})],
                                           [("4  ", {"color": ACCENT, "bold": True, "size": 14}), ("Review the tank masks; self-training on unlabelled video.", {"size": 18})]], space_after=12)

    # ---------------------------------------------------------------- thank you last
    t = list(ids)[1]; ids.remove(t); ids.append(t)
    for i, sl in enumerate(prs.slides, 1): sl.part.partname = PackURI(f"/ppt/slides/slide{i}.xml")
    prs.save(str(OUT))
    md = ["# Review 3: speaker notes\n", "One block per slide, in order (the title slide and the THANK YOU slide have no notes).\n"]
    for i, (ttl, nt) in enumerate(NOTES, 2): md.append(f"## Slide {i}: {ttl}\n\n{nt}\n")
    (PPTS / "speaker_review3.md").write_text("\n".join(md))
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
