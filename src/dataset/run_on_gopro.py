"""Run the current models and the earlier methods on data/dhwani_dataset (302 frames, no ground truth) and report what can be said without masks.

  python run_on_gopro.py sanity   # same code on USOD test images: unified C vs stored IoU, baselines vs stored IoU
  python run_on_gopro.py sup      # supervised models (unified A/B/C, +TTA, squashed input, earlier two-channel C)
  python run_on_gopro.py base     # k-means border rule and dendrogram gap rule (stride 3), resumable
  python run_on_gopro.py report   # tables, summary_bars.png, one collage per example frame (results/gopro_uw/)
  python run_on_gopro.py videos   # two short demo videos (original | unified C overlay)

Nothing is trained or tuned on this data. Masks are stored packed at 540 x 960 in outputs/gopro/.
"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/supervised", "src/common", "src/kmeans", "src/dendrogram"):
    sys.path.insert(0, str(ROOT / p))
from common import Head, OUT, extract, load_backbone, mask_metrics, normalise, prob_to_mask, to_tensor, build_records, load_gt  # noqa: E402
import tta  # noqa: E402
import baseline_masks as BM  # noqa: E402
import dinov2_cluster_union_stride as KM  # noqa: E402
from dendro_cluster import build_tree  # noqa: E402
from dinov2_dendro import FeatureContext, evaluate_tree  # noqa: E402

DATA = ROOT / "data" / "dhwani_dataset"
RES = ROOT / "results" / "dhwani_dataset_results"
BULK = OUT / "gopro"
OUT_HW = (540, 960)
BASE_HW = (462, 812)           # supervised input, aspect preserved (33 x 58 patches)
SQUASH_HW = (462, 616)         # training input size (4:3)
METHODS = ["kmeans", "dendro", "old_sal", "old_nw", "uni_A", "uni_B", "uni_C", "uni_C_tta", "uni_C_squash"]
NAMES = {"kmeans": "k-means (border rule)", "dendro": "dendrogram (gap rule)", "old_sal": "earlier C: salient channel",
         "old_nw": "earlier C: not-water channel", "uni_A": "unified A (linear probe)", "uni_B": "unified B (conv head)",
         "uni_C": "unified C (+LoRA)", "uni_C_tta": "unified C + TTA", "uni_C_squash": "unified C, 4:3 input"}
SHOW = ["kmeans", "dendro", "old_sal", "old_nw", "uni_A", "uni_B", "uni_C"]
ANY = 0.01                     # "a subject is predicted" = more than 1% of the frame
DEV = "cuda"
SIZE, STRIDE = (588, 448), 3
VIEWS = [(1.0, False), (1.0, True), (0.75, False), (1.25, False)]


# ----------------------------------------------------------------------------- supervised models
def load_old():
    ck = torch.load(OUT / "ckpt" / "stage_C.pt")
    bb = get_peft_model(load_backbone(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["qkv", "proj"], bias="none")).eval()
    sd = bb.state_dict(); sd.update(ck["lora"]); bb.load_state_dict(sd)
    head = Head(**{k: v for k, v in ck["head_cfg"].items() if k not in ("lr", "wd")}, out=2).to(DEV).eval()
    head.load_state_dict(ck["head"])
    return bb.base_model.model, head, ck["thr"]


def load_all():
    uni = tta.load_models("pool")
    thr = {s: json.load(open(ROOT / "results" / "subject_v2" / "pool" / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
    return uni, thr, load_old()


@torch.no_grad()
def _tokens(base, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = base.forward_features(x)["x_norm_patchtokens"]
    return t.float().transpose(1, 2).reshape(1, -1, x.shape[2] // 14, x.shape[3] // 14)


@torch.no_grad()
def sup_predict(img, models, out_hw=OUT_HW, base_hw=BASE_HW, extras=True):
    """-> {method: bool mask (out_hw)} for the supervised methods."""
    (frozen, ubase, heads, hc), uthr, (obase, ohead, othr) = models
    img = img.convert("RGB")
    P = lambda lg: torch.sigmoid(F.interpolate(lg.float(), size=out_hw, mode="bilinear", align_corners=False))[0]
    x = normalise(to_tensor(img.resize(base_hw[::-1], Image.BILINEAR))[None].to(DEV))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = extract(frozen, x).float()
    out = {"uni_A": P(heads["A"](f))[0] > uthr["A"], "uni_B": P(heads["B"](f))[0] > uthr["B"], "uni_C": P(hc(_tokens(ubase, x)))[0] > uthr["C"]}
    o = P(ohead(_tokens(obase, x)))
    out["old_sal"], out["old_nw"] = o[0] > othr["usod"], o[1] > othr["suim"]
    if extras:
        x0 = to_tensor(img.resize(base_hw[::-1], Image.BILINEAR))[None].to(DEV)
        acc = 0
        for sc, flip in VIEWS:
            hs, ws = round(base_hw[0] * sc / 14) * 14, round(base_hw[1] * sc / 14) * 14
            xv = x0 if (hs, ws) == tuple(base_hw) else F.interpolate(x0, size=(hs, ws), mode="bilinear", align_corners=False, antialias=sc < 1)
            if flip: xv = xv.flip(-1)
            p = P(hc(_tokens(ubase, normalise(xv))))[0]
            acc = acc + (p.flip(-1) if flip else p)
        out["uni_C_tta"] = acc / len(VIEWS) > uthr["C"]
        xs = normalise(to_tensor(img.resize(SQUASH_HW[::-1], Image.BILINEAR))[None].to(DEV))
        out["uni_C_squash"] = P(hc(_tokens(ubase, xs)))[0] > uthr["C"]
    return {k: v.cpu().numpy() for k, v in out.items()}


# ----------------------------------------------------------------------------- label-free baselines
_km = {}


def dendro_gap(img, out_hw):
    """Best Sec. 19 variant (768-D, stride 3, free, average linkage, gap rule) computed per image, same calls as dinov2_dendro.main."""
    if "m" not in _km: _km["m"] = KM.load_model("dinov2_vitb14", DEV)
    dense, _ = KM.dense_features(_km["m"], img, SIZE, STRIDE, DEV)
    grid = dense.shape[:2]
    X = dense.reshape(-1, dense.shape[-1])
    ctx = FeatureContext(X, grid, DEV)
    Z = build_tree(X[ctx.sub] if ctx.subsampled else X, grid, "average", False, "768")
    m = evaluate_tree(Z, ctx, False, round(KM.PATCH / STRIDE))["masks"]["gap_border"]
    return KM.upsample_maps(m[None].astype(np.float32), out_hw, STRIDE, SIZE)[0] > 0.5


def kmeans_border(path, out_hw):
    return BM.kmeans_border(path, out_hw)


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


# ----------------------------------------------------------------------------- commands
def meta():
    d = pd.read_csv(DATA / "metadata.csv"); d["path"] = [DATA / "images" / f for f in d.file]
    return d


def pack(masks):  # [N, H, W] bool -> [N, bytes]
    return np.stack([np.packbits(m.reshape(-1)) for m in masks])


def unpack(arr, i):
    return np.unpackbits(arr[i])[: OUT_HW[0] * OUT_HW[1]].reshape(OUT_HW).astype(bool)


def cmd_sanity():
    recs = build_records(); ev = [r for r in recs if r["ds"] == "usod" and r["split"] == "test"]
    ref = pd.read_csv(ROOT / "results" / "subject_v2" / "per_image_pool_usod.csv", dtype={"image": str}).set_index("image")
    models = load_all(); rows = []
    for r in ev[:20]:
        gt = load_gt(r["data"], r["stem"], "usod"); img = Image.open(r["img"])
        m = sup_predict(img, models, out_hw=gt.shape, base_hw=SQUASH_HW, extras=False)
        row = dict(image=r["stem"], stored_C=ref.loc[r["stem"], "model C"], new_C=mask_metrics(m["uni_C"], gt)["iou"],
                   stored_A=ref.loc[r["stem"], "model A"], new_A=mask_metrics(m["uni_A"], gt)["iou"])
        rows.append(row)
    d = pd.DataFrame(rows); print(d.round(3).to_string())
    print("max |new-stored| C: %.4f  A: %.4f" % ((d.new_C - d.stored_C).abs().max(), (d.new_A - d.stored_A).abs().max()))
    rows = []
    for r in ev[:5]:
        gt = load_gt(r["data"], r["stem"], "usod"); img = Image.open(r["img"])
        rows.append(dict(image=r["stem"], stored_dendro=ref.loc[r["stem"], "dendrogram (gap rule)"],
                         new_dendro=_iou(dendro_gap(img, gt.shape), gt), stored_kmeans=ref.loc[r["stem"], "k-means (border rule)"], new_kmeans=_iou(kmeans_border(r["img"], gt.shape), gt)))
    print(pd.DataFrame(rows).round(3).to_string())


def cmd_sup():
    d = meta(); models = load_all(); store = {k: [] for k in ["uni_A", "uni_B", "uni_C", "uni_C_tta", "uni_C_squash", "old_sal", "old_nw"]}
    t0 = time.time()
    for i, p in enumerate(d.path):
        m = sup_predict(Image.open(p), models)
        for k in store: store[k].append(m[k])
        if i % 30 == 0: print(i, round(time.time() - t0), "s", flush=True)
    BULK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BULK / "masks_sup.npz", **{k: pack(np.stack(v)) for k, v in store.items()})
    print("saved supervised masks", round(time.time() - t0), "s")


def cmd_base():
    d = meta(); f = BULK / "masks_base.npz"; BULK.mkdir(parents=True, exist_ok=True)
    km, dg = [], []
    if f.exists():
        z = np.load(f); km, dg = list(z["kmeans"]), list(z["dendro"])
    t0 = time.time()
    for i in range(len(km), len(d)):
        img = Image.open(d.path[i]).convert("RGB")
        km.append(np.packbits(kmeans_border(d.path[i], OUT_HW).reshape(-1))); dg.append(np.packbits(dendro_gap(img, OUT_HW).reshape(-1)))
        if (i + 1) % 20 == 0:
            np.savez_compressed(f, kmeans=np.stack(km), dendro=np.stack(dg)); print(i + 1, round(time.time() - t0), "s", flush=True)
    np.savez_compressed(f, kmeans=np.stack(km), dendro=np.stack(dg)); print("saved baseline masks", round(time.time() - t0), "s")


def overlay(img, mask):
    o = img.copy().astype(np.float32); col = np.array([230, 159, 0], np.float32)
    o[mask] = 0.5 * o[mask] + 0.5 * col
    o = o.astype(np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cs, -1, (255, 255, 255), 2)
    return o


def cmd_report():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = meta(); z = {**dict(np.load(BULK / "masks_sup.npz")), **dict(np.load(BULK / "masks_base.npz"))}
    RES.mkdir(parents=True, exist_ok=True)
    fg = {k: np.array([unpack(z[k], i).mean() for i in range(len(d))]) for k in METHODS}
    per = d[["file", "clip", "time_s", "tag"]].copy()
    for k in METHODS: per[f"fg_{k}"] = fg[k]
    for a, b, name in (("uni_C", "old_sal", "iou_unifiedC_vs_earlierC"), ("uni_C", "dendro", "iou_unifiedC_vs_dendro"), ("uni_C", "kmeans", "iou_unifiedC_vs_kmeans"),
                       ("uni_C", "uni_C_squash", "iou_unifiedC_vs_43input"), ("uni_C", "uni_C_tta", "iou_unifiedC_vs_TTA")):
        per[name] = [_iou(unpack(z[a], i), unpack(z[b], i)) for i in range(len(d))]
    per.to_csv(RES / "per_image.csv", index=False)
    tags = ["fish", "floor", "structure", "waterline", "empty"]
    rows = []
    for t in tags:
        sel = per.tag == t
        for k in METHODS:
            v = per.loc[sel, f"fg_{k}"]
            rows.append(dict(tag=t, method=k, n=int(sel.sum()), any_subject=float((v > ANY).mean()), mean_fg=float(v.mean()), median_fg=float(v.median())))
    S = pd.DataFrame(rows); S.to_csv(RES / "summary.csv", index=False)
    # ---------------- report
    def tbl(col, fmt):
        out = ["| method | " + " | ".join(f"{t} (n={int((per.tag == t).sum())})" for t in tags) + " |", "|---|" + "---|" * len(tags)]
        for k in METHODS:
            out.append(f"| {NAMES[k]} | " + " | ".join(fmt(S[(S.tag == t) & (S.method == k)][col].iloc[0]) for t in tags) + " |")
        return out
    md = ["# Third dataset (dhwani_dataset): what the models and earlier methods do on 302 unseen frames\n",
          "No ground truth exists for these frames, so **there is no IoU**. Nothing was trained or tuned on them. Supervised models run at 462 x 812 (aspect preserved) with the thresholds chosen earlier on validation data; k-means and dendrogram are the Sec. 10 / 19 label-free methods at stride 3. "
          "Tags are visual judgement (see `data/dhwani_dataset/README.md`). A frame counts as 'subject predicted' if more than 1% of its pixels are foreground.\n",
          "## Share of frames with a predicted subject (higher on `fish`, lower on `empty` is better)\n"] + tbl("any_subject", lambda v: f"{v:.0%}") + \
         ["", "## Mean predicted foreground share of the frame\n"] + tbl("mean_fg", lambda v: f"{v:.1%}")
    e = S[S.tag == "empty"].set_index("method")
    md += ["", "## False alarms on the 80 empty-water frames\n", "| method | frames with a false alarm | mean foreground share |", "|---|---|---|"]
    for k in METHODS: md.append(f"| {NAMES[k]} | {e.loc[k, 'any_subject']:.0%} | {e.loc[k, 'mean_fg']:.1%} |")
    md += ["", "For comparison, the crop test on SUIM empty crops (`results/subject_v2/false_alarm.md`) gave 19% / 19% / 10% for unified A / B / C and 100% for k-means.\n",
           "## Agreement and stability (mean IoU between two methods' masks over all 302 frames)\n", "| pair | mean IoU | median |", "|---|---|---|"]
    for name, lab in (("iou_unifiedC_vs_earlierC", "unified C vs earlier C (salient channel)"), ("iou_unifiedC_vs_dendro", "unified C vs dendrogram"), ("iou_unifiedC_vs_kmeans", "unified C vs k-means"),
                      ("iou_unifiedC_vs_43input", "unified C: aspect-preserved vs 4:3 squashed input"), ("iou_unifiedC_vs_TTA", "unified C vs unified C + TTA")):
        md.append(f"| {lab} | {per[name].mean():.3f} | {per[name].median():.3f} |")
    # ---------------- figure
    COL = {"kmeans": "#9ca3af", "dendro": "#6b7280", "old_sal": "#d97706", "old_nw": "#fbbf24", "uni_A": "#bfdbfe", "uni_B": "#60a5fa", "uni_C": "#1d4ed8"}
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.8))
    for ax, col, ttl in ((axs[0], "any_subject", "frames with a predicted subject"), (axs[1], "mean_fg", "mean predicted foreground share")):
        w = 0.11
        for j, k in enumerate(SHOW):
            ax.bar(np.arange(len(tags)) + (j - 3) * w, [S[(S.tag == t) & (S.method == k)][col].iloc[0] for t in tags], w, color=COL[k], label=NAMES[k])
        ax.set_xticks(range(len(tags))); ax.set_xticklabels([f"{t}\n(n={int((per.tag == t).sum())})" for t in tags]); ax.set_title(ttl, fontsize=10, loc="left")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    axs[0].set_ylim(0, 1.05); axs[0].legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout(); plt.savefig(RES / "summary_bars.png", dpi=140); plt.close()
    # ---------------- per-frame collages (rule-based selection)
    ex = RES / "examples"; picks = []
    for t in tags:
        g = per[per.tag == t].sort_values("fg_uni_C")
        med = g.iloc[(g.fg_uni_C - g.fg_uni_C.median()).abs().argsort().iloc[0]]
        chosen = [("median", med.name), ("largest", g.index[-1]), ("largest", g.index[-2]), ("smallest", g.index[0]), ("smallest", g.index[1])]
        seen = set()
        for why, i in chosen:
            if i not in seen: picks.append((t, why, i)); seen.add(i)
    dis = per[(per.fg_uni_C > ANY) | (per.fg_old_sal > ANY)].sort_values("iou_unifiedC_vs_earlierC").head(5)
    picks += [("disagreement", "unified C vs earlier C", i) for i in dis.index]
    done = set()
    for folder, why, i in picks:
        key = (folder, i)
        if key in done: continue
        done.add(key)
        img = np.asarray(Image.open(d.path[i]).convert("RGB").resize(OUT_HW[::-1]))
        panels = [("image", img, None)] + [(NAMES[k], overlay(img, unpack(z[k], i)), fg[k][i]) for k in SHOW]
        fig, axs = plt.subplots(2, 4, figsize=(20, 6.4))
        for ax, (ttl, im, f) in zip(axs.ravel(), panels):
            ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=10)
            for s_ in ax.spines.values(): s_.set_visible(False)
            if f is not None: ax.text(.02, .04, f"foreground {f:.0%}", transform=ax.transAxes, color="white", fontsize=11, bbox=dict(fc="black", alpha=.65, lw=0, pad=2))
        fig.suptitle(f"{d.file[i]} | tag {d.tag[i]} | {why} | orange = predicted subject", fontsize=10, x=0.01, ha="left")
        plt.tight_layout(); (ex / folder).mkdir(parents=True, exist_ok=True)
        plt.savefig(ex / folder / f"{Path(d.file[i]).stem}_C{fg['uni_C'][i]*100:02.0f}pct.png", dpi=80); plt.close()
    md += ["", "## Per-frame examples\n", "One PNG per frame under `examples/<tag>/` (median, two largest and two smallest unified-C foreground share per tag; for `empty` the largest are the worst false alarms) and `examples/disagreement/` (5 frames where unified C and earlier C differ most). "
           "Each file: frame, k-means, dendrogram, earlier C (salient and not-water channels), unified A, B, C; orange overlay = predicted subject, foreground share printed.\n",
           "`summary_bars.png`: bars of the two tables above. `per_image.csv`: all numbers per frame."]
    (RES / "report.md").write_text("\n".join(md)); print("\n".join(md))


def cmd_videos():
    models = load_all(); (RES / "videos").mkdir(parents=True, exist_ok=True)
    for name, clip, t0, t1 in (("fish_GX010214_52-62s", "GX010214", 52, 62), ("panels_GX010214_120-150s", "GX010214", 120, 150)):
        cap = cv2.VideoCapture(str(ROOT / "videos" / "underwater_selected" / f"{clip}.MP4")); fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000); n = int((t1 - t0) * fps)
        vw = cv2.VideoWriter(str(RES / "videos" / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / 2, (1920, 540)); k = 0
        for i in range(n):
            ok = cap.grab()
            if not ok: break
            if i % 2: continue
            ok, fr = cap.retrieve()
            fr = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA)
            m = sup_predict(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), models, out_hw=(540, 960), extras=False)["uni_C"]
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            cv2.putText(rgb, "unified C", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            vw.write(cv2.cvtColor(np.hstack([cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), overlay(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), m)]), cv2.COLOR_RGB2BGR)); k += 1
        vw.release(); print(name, k, "frames", flush=True)


if __name__ == "__main__":
    {"sanity": cmd_sanity, "sup": cmd_sup, "base": cmd_base, "report": cmd_report, "videos": cmd_videos}[sys.argv[1]]()
