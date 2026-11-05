"""Figures and numbers for the Review 3 deck -> presentations/figures_r3/ (run with the project venv; the deck itself is built by build_deck_review3.py).

Four example images, chosen by rule and used for EVERY method: from the 100 USOD test images the one closest to the median object size of each group
(small < 10%, medium 10-30%, large > 30% of the image) and the SUIM test image closest to the median not-water share. Each method gets a figure with the
originals on top and the method's masks below (IoU printed on every mask). Also: comparison grids, Dhwani tank frame grids, numbers.json (every number on the slides).
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "supervised"))
from common import build_records, load_gt, prob_to_mask, pred_path, OUT  # noqa: E402
import baseline_masks as BM  # noqa: E402

R = ROOT / "results"; FIG = ROOT / "presentations" / "figures_r3"; FIG.mkdir(exist_ok=True)
BG = (13, 11, 36); WHITE = (255, 255, 255); CYAN = (111, 211, 242)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
recs = build_records(); rec_idx = {(r["ds"], r["stem"], r["split"]): i for i, r in enumerate(recs)}
ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]; evpos = {i: k for k, i in enumerate(ev)}


# ------------------------------------------------------------------ example images
def pick_examples():
    ut = [r for r in recs if r["ds"] == "usod" and r["split"] == "test"]
    area = {r["stem"]: float(load_gt(r["data"], r["stem"], "usod").mean()) for r in ut}
    out = []
    for name, lo, hi in (("small object", 0, 0.1), ("medium object", 0.1, 0.3), ("large object", 0.3, 1.01)):
        grp = {s: a for s, a in area.items() if lo <= a < hi}; med = np.median(list(grp.values()))
        out.append(dict(ds="usod", stem=min(grp, key=lambda s: abs(grp[s] - med)), label=f"USOD, {name}"))
    km = pd.read_csv(R / "kmeans" / "suim" / "cluster_union_stride3_k6" / "metrics_union.csv"); km["image"] = km["image"].str.replace(".jpg", "", regex=False)
    ok = set(km.image); st = [r for r in recs if r["ds"] == "suim" and r["split"] == "test" and r["stem"] in ok]
    area = {r["stem"]: float(load_gt(r["data"], r["stem"], "suim_nonwater").mean()) for r in st}; med = np.median(list(area.values()))
    out.append(dict(ds="suim", stem=min(area, key=lambda s: abs(area[s] - med)), label="SUIM, median image"))
    for e in out:
        e["rec"] = next(r for r in recs if r["ds"] == e["ds"] and r["stem"] == e["stem"] and (e["ds"] == "usod" or r["split"] == "test"))
        e["i"] = recs.index(e["rec"])
    return out


EX = pick_examples(); print("examples:", [(e["stem"], e["label"]) for e in EX])
TW, TH = 480, 360


def img_tile(path_or_arr, w=TW, h=TH, mode="RGB"):
    im = Image.open(path_or_arr).convert("RGB") if not isinstance(path_or_arr, np.ndarray) else Image.fromarray(path_or_arr)
    return im.resize((w, h), Image.BILINEAR)


def mask_tile(m, w=TW, h=TH, badge=None):
    a = (np.asarray(m) > 0.5).astype(np.uint8) * 255
    im = Image.fromarray(a).convert("RGB").resize((w, h), Image.NEAREST)
    if badge is not None: stamp(im, badge)
    return im


def stamp(im, text, size="S"):
    d = ImageDraw.Draw(im); f = FONT_S if size == "S" else FONT
    bb = d.textbbox((0, 0), text, font=f); pad = 5
    d.rectangle([6, im.height - 12 - (bb[3] - bb[1]) - 2 * pad, 6 + bb[2] - bb[0] + 2 * pad, im.height - 6], fill=(0, 0, 0))
    d.text((6 + pad, im.height - 12 - (bb[3] - bb[1]) - pad - bb[1] + 4), text, fill=WHITE, font=f)


def grid(rows, headers=None, tile_w=TW, tile_h=TH, label_w=210, gap=10, header_h=44):
    """rows: list of (row label, [PIL tiles]); headers: column titles."""
    ncol = max(len(t) for _, t in rows); W = label_w + ncol * (tile_w + gap); H = (header_h if headers else 0) + len(rows) * (tile_h + gap)
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)
    if headers:
        for j, h in enumerate(headers): d.text((label_w + j * (tile_w + gap) + 4, 8), h, fill=CYAN, font=FONT_S)
    for i, (lab, tiles) in enumerate(rows):
        y = (header_h if headers else 0) + i * (tile_h + gap)
        for k, line in enumerate(lab.split("\n")): d.text((6, y + tile_h // 2 - 14 + 26 * (k - (lab.count("\n")) / 2)), line, fill=WHITE if k == 0 else CYAN, font=FONT if k == 0 else FONT_S)
        for j, t in enumerate(tiles): canvas.paste(t, (label_w + j * (tile_w + gap), y))
    return canvas


def iou(m, g):
    u = (m | g).sum(); return float((m & g).sum() / u) if u else 1.0


# ------------------------------------------------------------------ masks of every method for the 4 examples
GT_NW = {e["i"]: load_gt(e["rec"]["data"], e["stem"], "usod" if e["ds"] == "usod" else "suim_nonwater") for e in EX}
GT_SUBJ = {e["i"]: load_gt(e["rec"]["data"], e["stem"], "usod" if e["ds"] == "usod" else "suim_subject") for e in EX}
items = [dict(e["rec"], gt_mode="usod" if e["ds"] == "usod" else "suim_nonwater") for e in EX]
bm = BM.get_masks(items)
M = {}   # method -> list of (mask, iou, gt_used)
M["kmeans"] = [(bm[e["stem"]]["k-means (border rule)"][0], bm[e["stem"]]["k-means (border rule)"][1]) for e in EX]
M["dendro"] = [(bm[e["stem"]]["dendrogram (gap rule)"][0], bm[e["stem"]]["dendrogram (gap rule)"][1]) for e in EX]
unp = lambda t: np.unpackbits(t[0])[: int(np.prod(t[1]))].reshape(t[1]).astype(bool)
for key, ds, gtcol in (("usod", "usod", "usod"), ("suim", "suim", "suim_nonwater")):
    pk = pickle.load(open(OUT / "gmm" / f"eval_{key}.pkl", "rb")); per = pd.read_csv(R / "gmm" / f"per_image_{key}.csv", dtype={"image": str}).set_index("image")
    for e in EX:
        if e["ds"] != ds: continue
        for name, sk, ck in (("gmm_pi", "pi_own", "iou_pi_own"), ("gmm_gl", "gl_own", "iou_gl_own")):
            M.setdefault(name, {})[e["i"]] = (unp(pk[e["stem"]][sk]), float(per.loc[e["stem"], f"{ck}_{gtcol}"]))
for name in ("gmm_pi", "gmm_gl"): M[name] = [M[name][e["i"]] for e in EX]
sup_thr = {s: json.load(open(R / "supervised" / f"stage_{s}" / "config.json")).get("thr", {"usod": 0.5, "suim": 0.5}) for s in "ABC"}; sup_thr["D"] = {"usod": 0.5, "suim": 0.5}
for s in "ABCD":
    P = np.load(pred_path(f"stage_{s}", "evalset")).astype(np.float32); out = []
    for e in EX:
        ch = 0 if e["ds"] == "usod" else 1; g = GT_NW[e["i"]]
        m = prob_to_mask(P[evpos[e["i"]], ch], g.shape, sup_thr[s][e["ds"]]); out.append((m, iou(m, g)))
    M[f"sup_{s}"] = out
uthr = {s: json.load(open(R / "subject_v2" / "pool" / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
for s in "ABC":
    P = np.load(pred_path(f"v2_pool_{s}", "evalset")).astype(np.float32)[:, 0]; out = []
    for e in EX:
        g = GT_SUBJ[e["i"]]; m = prob_to_mask(P[evpos[e["i"]]], g.shape, uthr[s]); out.append((m, iou(m, g)))
    M[f"uni_{s}"] = out


def originals(): return [img_tile(e["rec"]["img"]) for e in EX]
def pair_fig(name, label, key):
    rows = [("image", originals()), (label, [mask_tile(m, badge=f"IoU {v:.2f}") for m, v in M[key]])]
    grid(rows, headers=[e["label"] for e in EX]).save(FIG / f"pair_{name}.png")


for name, label, key in (("kmeans", "k-means\nborder rule", "kmeans"), ("dendro", "dendrogram\ngap rule", "dendro"), ("gmm_pi", "GMM\nper-image", "gmm_pi"), ("gmm_gl", "GMM\ndataset-level", "gmm_gl"),
                         ("stageA", "Stage A\nlinear probe", "sup_A"), ("stageB", "Stage B\nconv head", "sup_B"), ("stageC", "Stage C\n+ LoRA", "sup_C"), ("stageD", "Stage D\n+ SAM", "sup_D"), ("unified", "unified C\none subject rule", "uni_C")):
    if name != "unified": pair_fig(name, label, key)
grid([("image", originals()), ("ground truth\nsubject rule", [mask_tile(GT_SUBJ[e["i"]]) for e in EX]), ("unified C", [mask_tile(m, badge=f"IoU {v:.2f}") for m, v in M["uni_C"]])], headers=[e["label"] for e in EX]).save(FIG / "pair_unified.png")
# ground-truth row for the unified label differs for SUIM: show it in its own figure
grid([("image", originals()), ("ground truth\nsalient / not-water", [mask_tile(GT_NW[e["i"]]) for e in EX]), ("ground truth\nsubject rule (Sec. 23)", [mask_tile(GT_SUBJ[e["i"]]) for e in EX])], headers=[e["label"] for e in EX]).save(FIG / "gt_rows.png")
# comparison grids (4 rows = the four examples)
small = dict(tile_w=300, tile_h=225, label_w=170, header_h=40)
rows = [(e["label"].replace(", ", "\n"), [img_tile(e["rec"]["img"], 300, 225), mask_tile(GT_NW[e["i"]], 300, 225)] + [mask_tile(M[k][j][0], 300, 225, f"{M[k][j][1]:.2f}") for k in ("kmeans", "dendro", "gmm_pi", "gmm_gl")]) for j, e in enumerate(EX)]
grid(rows, headers=["image", "ground truth", "k-means", "dendrogram", "GMM per-image", "GMM dataset-level"], **small).save(FIG / "cmp_unsup.png")
rows = [(e["label"].replace(", ", "\n"), [img_tile(e["rec"]["img"], 270, 203), mask_tile(GT_NW[e["i"]], 270, 203)] + [mask_tile(M[k][j][0], 270, 203, f"{M[k][j][1]:.2f}") for k in ("sup_A", "sup_B", "sup_C", "sup_D")] + [mask_tile(M["uni_C"][j][0], 270, 203, f"{M['uni_C'][j][1]:.2f}")]) for j, e in enumerate(EX)]
grid(rows, headers=["image", "ground truth", "A linear", "B conv head", "C + LoRA", "D + SAM", "unified C*"], tile_w=270, tile_h=203, label_w=150, header_h=40).save(FIG / "cmp_sup.png")

# ------------------------------------------------------------------ Dhwani frames: the median frame of each scene tag (by unified-C foreground share)
meta = pd.read_csv(ROOT / "data" / "dhwani_dataset" / "metadata.csv"); per = pd.read_csv(R / "dhwani_dataset_results" / "per_image.csv")
BULK = OUT / "gopro"; Z = {**dict(np.load(BULK / "masks_sup.npz")), **dict(np.load(BULK / "masks_base.npz")), **dict(np.load(BULK / "masks_pool2.npz"))}
HW = (540, 960); un = lambda k, i: np.unpackbits(Z[k][i])[: HW[0] * HW[1]].reshape(HW).astype(bool)
fr = []
for tag in ("fish", "structure", "floor", "empty"):
    g = per[per.tag == tag].sort_values("fg_uni_C"); i = int(g.index[len(g) // 2]); fr.append(dict(tag=tag, i=i, file=per.file[i]))
print("tank frames:", [(f["tag"], f["file"]) for f in fr])
tank = pd.read_csv(R / "subject_v3" / "tank" / "per_image.csv").set_index("file")
tw, th = 480, 270
def dh_orig(f): return Image.open(ROOT / "data" / "dhwani_dataset" / "images" / f["file"]).convert("RGB").resize((tw, th))
def dh_mask(key, f):
    m = un(key, f["i"]); t = mask_tile(m, tw, th)
    lab = f"predicted {m.mean():.0%}" if f["tag"] != "fish" else f"IoU {tank.loc[f['file'], 'iou_' + key]:.2f}, predicted {m.mean():.0%}"
    stamp(t, lab); return t
hd = [f"{f['tag']} frame" for f in fr]
D = dict(tile_w=tw, tile_h=th, label_w=200, header_h=44)
grid([("frame", [dh_orig(f) for f in fr]), ("k-means", [dh_mask("kmeans", f) for f in fr]), ("dendrogram", [dh_mask("dendro", f) for f in fr])], headers=hd, **D).save(FIG / "dh_unsup.png")
grid([("frame", [dh_orig(f) for f in fr]), ("earlier C\nsalient channel", [dh_mask("old_sal", f) for f in fr]), ("unified C\n(subject rule)", [dh_mask("uni_C", f) for f in fr]), ("new C\n(strict rule)", [dh_mask("new_C", f) for f in fr])], headers=hd, **D).save(FIG / "dh_sup.png")
grid([("frames", [dh_orig(f) for f in fr])], headers=hd, **D).save(FIG / "dh_frames.png")
import cv2
c = cv2.VideoCapture(str(ROOT / "outputs" / "tools" / "fish_h264.mp4")); c.set(1, 45); ok, f0 = c.read(); cv2.imwrite(str(FIG / "fish_poster.png"), f0)

# ------------------------------------------------------------------ Review 1 baselines (USOD only) and the k-means vs GMM concept figure
e1 = pd.read_csv(R / "kmeans" / "usod" / "metrics.csv"); e1["stem"] = e1["image"].str.replace(".jpg", "", regex=False); e1 = e1.set_index("stem")
def not_run():
    t = Image.new("RGB", (TW, TH), (28, 26, 58)); d = ImageDraw.Draw(t); d.text((40, TH // 2 - 14), "not run in Review 1", fill=CYAN, font=FONT); return t
def e1_tile(kind, e):
    if e["ds"] != "usod": return not_run()
    m = np.asarray(Image.open(R / "kmeans" / "usod" / f"masks_{kind}" / f"{e['stem']}.png").convert("L")) > 127
    return mask_tile(m, badge=f"IoU {e1.loc[e['stem'], ('pca_iou' if kind == 'pca' else 'kmeans_iou')]:.2f}")
grid([("image", originals()), ("PCA mask\nfirst component", [e1_tile("pca", e) for e in EX]), ("k-means\nk = 2", [e1_tile("kmeans", e) for e in EX])], headers=[e["label"] for e in EX]).save(FIG / "pair_review1.png")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
rng = np.random.RandomState(3)
A = rng.multivariate_normal([0, 0], [[3.2, 2.6], [2.6, 2.4]], 400); B_ = rng.multivariate_normal([3.2, -2.2], [[0.5, 0], [0, 0.5]], 200); X = np.vstack([A, B_])
km = KMeans(2, n_init=5, random_state=0).fit(X); gm = GaussianMixture(2, covariance_type="full", random_state=0).fit(X)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#0D0B24")
for ax, lab, ttl in ((axs[0], km.labels_, "k-means: hard labels, round clusters"), (axs[1], gm.predict(X), "GMM: ellipses with their own size and weight")):
    ax.set_facecolor("#0D0B24"); ax.scatter(X[:, 0], X[:, 1], c=np.where(lab == 0, "#6FD3F2", "#F2C46F"), s=9); ax.set_title(ttl, color="white", fontsize=12); ax.axis("off")
from matplotlib.patches import Ellipse
for k in range(2):
    w, v = np.linalg.eigh(gm.covariances_[k]); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    for sc in (1, 2): axs[1].add_patch(Ellipse(gm.means_[k], 2 * sc * np.sqrt(w[1]), 2 * sc * np.sqrt(w[0]), angle=ang, fill=False, ec="white", lw=1.4))
    axs[0].scatter(*km.cluster_centers_[k], marker="x", c="white", s=120, lw=3)
plt.tight_layout(); plt.savefig(FIG / "gmm_concept.png", dpi=130, facecolor="#0D0B24"); plt.close()


# ------------------------------------------------------------------ dark pipeline figure
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(15, 3.6), facecolor="#0D0B24"); ax.set_xlim(0, 15); ax.set_ylim(0, 3.6); ax.axis("off")
def bx(x, y, w, h, t, c, fs=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=c, ec="white", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color="#1f2937")
def ar(x0, x1, y=1.8): ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color="white", lw=1.6))
bx(0.2, 1.2, 1.9, 1.2, "image\n462 x 616", "#eeeeee"); ar(2.15, 2.6); bx(2.65, 0.8, 3.3, 2.0, "DINOv2 ViT-B/14\n86.6 M weights\nFROZEN in A, B\n+ 0.9 M LoRA adapters\nin C (trained)", "#cfe3f7"); ar(6.0, 6.5)
bx(6.55, 1.2, 2.0, 1.2, "patch tokens\n33 x 44 x 768", "#eeeeee"); ar(8.6, 9.1); bx(9.1, 0.6, 3.35, 2.4, "HEAD (trained)\nA: one 1x1 layer, 769 weights\nB: 1x1 reduce + dilated\nresidual 3x3 blocks\n(0.4 M to 2 M weights)", "#fde3c4"); ar(12.5, 12.95)
bx(13.0, 1.2, 1.9, 1.2, "7x up, sigmoid,\nthreshold", "#eeeeee")
ax.text(7.5, 0.25, "Loss = BCE + Dice against the 231 x 308 ground-truth mask. A/B: features cached once, only the head trains. C: head + LoRA trained end to end.", ha="center", fontsize=11, color="#E6EAF2")
plt.savefig(FIG / "pipeline_dark.png", dpi=130, facecolor="#0D0B24", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------------ numbers.json
N = {"examples": [(e["stem"], e["label"]) for e in EX], "tank_frames": [(f["tag"], f["file"]) for f in fr]}
g = pd.read_csv(R / "gmm" / "summary.csv"); N["gmm"] = {s: {m: float(x) for m, x in zip(d.method, d.iou)} for s, d in g.groupby("setting")}
for src in ("usod", "suim"):
    d = pd.read_csv(R / "gmm" / f"dev_ablation_{src}.csv", dtype={"image": str}); a = d.groupby(["cov", "d", "krule", "smooth"]).iou.mean().reset_index()
    N[f"dev_{src}"] = {"bic": float(a[a.krule == "bic"].iou.max()), "sil": float(a[a.krule == "sil"].iou.max()), "meanK_bic": float(d[d.krule == "bic"].K.mean()), "meanK_sil": float(d[d.krule == "sil"].K.mean())}
kmd = pd.read_csv(R / "kmeans" / "usod" / "cluster_union_stride3_k6" / "metrics_union.csv"); kmd = kmd[kmd.feat.astype(str) == "768"]; kmd["stem"] = kmd["image"].str.replace(".jpg", "", regex=False)
dev_stems = set(pd.read_csv(R / "gmm" / "dev_ablation_usod.csv", dtype={"image": str}).image); N["dev_usod"]["kmeans"] = float(kmd[kmd.stem.isin(dev_stems)].border_iou.mean())
c = pd.read_csv(R / "comparison" / "comparison.csv"); N["sup"] = {}
for _, r in c.iterrows(): N["sup"].setdefault("usod" if r.dataset.startswith("USOD") else "suim", {})[r.method] = float(r.iou)
cs = pd.read_csv(R / "comparison" / "iou_by_object_size.csv"); N["sup_size"] = [dict(ds=r.ds, size=r["size"], method=r.method, iou=float(r.iou)) for _, r in cs.iterrows()]
x = pd.read_csv(R / "cross_dataset" / "summary.csv"); N["cross"] = {s: {m: float(i) for m, i in zip(d.method, d.iou)} for s, d in x.groupby("setting")}
v = pd.read_csv(R / "subject_v2" / "summary.csv"); N["unified"] = {s: {m: float(i) for m, i in zip(d.method, d.iou)} for s, d in v.groupby("setting")}
N["seeds"] = pd.read_csv(R / "subject_v2" / "seeds_summary.csv").to_dict("records")
t = pd.read_csv(R / "subject_v3" / "tank" / "summary.csv"); N["tank"] = {r.method: {k: float(r[k]) for k in ("iou", "recall", "precision", "false_alarm_rate", "severe_rate", "mean_fp_share")} for _, r in t.iterrows()}
fa = pd.read_csv(R / "subject_v2" / "false_alarm_per_crop.csv"); tt = fa[fa.split == "test"]; N["crop_false_alarm"] = {c: float((tt[f"fg_{c}"] > 0.01).mean()) for c in ("kmeans", "A", "B", "C")}
json.dump(N, open(FIG / "numbers.json", "w"), indent=1); print("figures:", sorted(p.name for p in FIG.glob("*.png")))
