"""Videos of every implemented method on the tank footage (results/dhwani_dataset_results/videos/).

Six 30 s segments (one per underwater clip), 5 frames/s, 12 methods, all frozen (nothing trained or tuned), per-frame prediction without temporal smoothing:
  1 k-means  2 dendrogram  3-5 earlier Stage A / B / C (salient channel)  6 earlier Stage C not-water channel  7-9 unified A/B/C (Sec. 23)  10-12 new A/B/C (Sec. 26)

  python make_videos.py frames   # decode the segments from the 4K clips -> outputs/dhwani_videos/frames_<clip>.npy
  python make_videos.py sup      # supervised masks (fast)
  python make_videos.py base     # k-means + dendrogram masks (about 5 s per frame, resumable)
  python make_videos.py check    # consistency check against the masks computed earlier from the extracted JPEG frames
  python make_videos.py render   # per-method videos, grid videos, foreground_share.csv, README
"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_on_gopro as R   # noqa: E402
from run_on_gopro import ROOT, OUT, overlay, tta, load_old, sup_predict, dendro_gap, kmeans_border, Head, extract, load_backbone, normalise, to_tensor  # noqa: E402

CACHE = OUT / "dhwani_videos"
RES = ROOT / "results" / "dhwani_dataset_results" / "videos"
SRC = ROOT / "videos" / "underwater_selected"
SEGMENTS = [("GX010214", 95, 125), ("GX010216", 100, 130), ("GX010217", 88, 118), ("GX010218", 90, 120), ("GX010220", 105, 135), ("GX010221", 80, 110)]
FPS = 5
NF = 150
H, W = 540, 960
METHODS = ["kmeans", "dendro", "old_A", "old_B", "old_C", "old_nw", "uni_A", "uni_B", "uni_C", "new_A", "new_B", "new_C"]
NAMES = {"kmeans": "k-means (border rule)", "dendro": "dendrogram (gap rule)", "old_A": "earlier Stage A (salient)", "old_B": "earlier Stage B (salient)",
         "old_C": "earlier Stage C (salient)", "old_nw": "earlier Stage C (not-water)", "uni_A": "unified A (Sec. 23)", "uni_B": "unified B (Sec. 23)",
         "uni_C": "unified C (Sec. 23)", "new_A": "new A (Sec. 26)", "new_B": "new B (Sec. 26)", "new_C": "new C (Sec. 26)"}


def pack(m):  # [N,H,W] bool
    return np.stack([np.packbits(x.reshape(-1)) for x in m])


def unpack(a, i):
    return np.unpackbits(a[i])[: H * W].reshape(H, W).astype(bool)


def cmd_frames():
    CACHE.mkdir(parents=True, exist_ok=True)
    for clip, t0, t1 in SEGMENTS:
        cap = cv2.VideoCapture(str(SRC / f"{clip}.MP4")); fps = cap.get(cv2.CAP_PROP_FPS)
        targets = [int(round((t0 + k / FPS) * fps)) for k in range(NF)]
        cap.set(cv2.CAP_PROP_POS_FRAMES, targets[0]); fi = targets[0]; out = []
        for tgt in targets:
            while fi < tgt:
                if not cap.grab(): break
                fi += 1
            ok, fr = cap.read(); fi += 1
            if not ok: print("decode failed", clip, tgt); break
            out.append(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        np.save(CACHE / f"frames_{clip}.npy", np.stack(out)); print(clip, len(out), "frames", flush=True)


def _old_ab():
    frozen = load_backbone()
    heads = {}
    for s in "AB":
        ck = torch.load(OUT / "ckpt" / f"stage_{s}.pt")
        h = Head(**{k: v for k, v in ck["cfg"].items() if k not in ("lr", "wd")}).cuda().eval(); h.load_state_dict(ck["state"]); heads[s] = (h, ck["thr"]["usod"])
    return frozen, heads


@torch.no_grad()
def old_ab(img, frozen, heads):
    x = normalise(to_tensor(img.resize((812, 462), Image.BILINEAR))[None].cuda())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = extract(frozen, x).float()
    return {f"old_{s}": (torch.sigmoid(F.interpolate(h(f).float(), size=(H, W), mode="bilinear", align_corners=False))[0, 0] > thr).cpu().numpy() for s, (h, thr) in heads.items()}


def cmd_sup():
    uni_old, uni_new = tta.load_models("pool"), tta.load_models("pool2")
    thr = lambda tag, rd: {s: json.load(open(ROOT / "results" / rd / tag / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}
    old = load_old(); m_old = (uni_old, thr("pool", "subject_v2"), old); m_new = (uni_new, thr("pool2", "subject_v3"), old)
    frozen, ab = _old_ab(); t0 = time.time()
    for clip, _, _ in SEGMENTS:
        fr = np.load(CACHE / f"frames_{clip}.npy"); store = {k: [] for k in METHODS if k not in ("kmeans", "dendro")}
        for i in range(len(fr)):
            img = Image.fromarray(fr[i])
            a = sup_predict(img, m_old, out_hw=(H, W), extras=False); b = sup_predict(img, m_new, out_hw=(H, W), extras=False); c = old_ab(img, frozen, ab)
            row = {"old_A": c["old_A"], "old_B": c["old_B"], "old_C": a["old_sal"], "old_nw": a["old_nw"],
                   "uni_A": a["uni_A"], "uni_B": a["uni_B"], "uni_C": a["uni_C"], "new_A": b["uni_A"], "new_B": b["uni_B"], "new_C": b["uni_C"]}
            for k in store: store[k].append(row[k])
        np.savez_compressed(CACHE / f"masks_sup_{clip}.npz", **{k: pack(np.stack(v)) for k, v in store.items()}); print(clip, "sup done", round(time.time() - t0), "s", flush=True)


def cmd_base():
    tmp = CACHE / "tmp_frame.png"
    for clip, _, _ in SEGMENTS:
        f = CACHE / f"masks_base_{clip}.npz"; fr = np.load(CACHE / f"frames_{clip}.npy")
        km, dg = ([], []) if not f.exists() else (list(np.load(f)["kmeans"]), list(np.load(f)["dendro"]))
        t0 = time.time()
        for i in range(len(km), len(fr)):
            img = Image.fromarray(fr[i]); img.save(tmp)
            km.append(np.packbits(kmeans_border(str(tmp), (H, W)).reshape(-1))); dg.append(np.packbits(dendro_gap(img, (H, W)).reshape(-1)))
            if (i + 1) % 25 == 0:
                np.savez_compressed(f, kmeans=np.stack(km), dendro=np.stack(dg)); print(clip, i + 1, round(time.time() - t0), "s", flush=True)
        np.savez_compressed(f, kmeans=np.stack(km), dendro=np.stack(dg)); print(clip, "base done", flush=True)


def cmd_check():
    """Same pipeline on frames decoded at the exact frame index of dataset JPEGs: unified-C mask vs the mask computed earlier from the JPEG."""
    meta = pd.read_csv(ROOT / "data" / "dhwani_dataset" / "metadata.csv"); z = np.load(ROOT / "outputs" / "gopro" / "masks_sup.npz")
    models = (tta.load_models("pool"), {s: json.load(open(ROOT / "results" / "subject_v2" / "pool" / f"stage_{s}" / "config.json"))["thr"]["all"] for s in "ABC"}, load_old())
    ious = []
    for i in range(0, len(meta), 60):
        r = meta.iloc[i]; cap = cv2.VideoCapture(str(SRC / f"{r['clip']}.MP4")); cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame_idx)); ok, fr = cap.read()
        img = Image.fromarray(cv2.cvtColor(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        m = sup_predict(img, models, extras=False)["uni_C"]; ref = np.unpackbits(z["uni_C"][i])[: H * W].reshape(H, W).astype(bool)
        u = (m | ref).sum(); ious.append(float((m & ref).sum() / u) if u else 1.0); print(r.file, "IoU vs earlier mask %.3f" % ious[-1])
    print("mean", np.mean(ious))


def _writer(path, size, fps):
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened(): return w, cc
    raise RuntimeError("no video codec")


def _bar(img, text, h=36):
    bar = np.full((h, img.shape[1], 3), 28, np.uint8); cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([bar, img])


def cmd_render():
    (RES / "per_method").mkdir(parents=True, exist_ok=True); (RES / "grid").mkdir(exist_ok=True)
    data = {}
    for clip, _, _ in SEGMENTS:
        z = {**dict(np.load(CACHE / f"masks_sup_{clip}.npz")), **dict(np.load(CACHE / f"masks_base_{clip}.npz"))}
        data[clip] = (np.load(CACHE / f"frames_{clip}.npy"), z)
    rows = []
    for clip, t0, t1 in SEGMENTS:
        fr, z = data[clip]
        for m in METHODS: rows.append(dict(clip=clip, segment=f"{t0}-{t1}s", method=m, name=NAMES[m], mean_fg=float(np.mean([unpack(z[m], i).mean() for i in range(len(fr))]))))
    pd.DataFrame(rows).to_csv(RES / "foreground_share.csv", index=False)
    for n, m in enumerate(METHODS, 1):   # one video per method: original | overlay, six segments joined
        w, cc = _writer(RES / "per_method" / f"{n:02d}_{m}.mp4", (1920, 576), FPS)
        for clip, t0, t1 in SEGMENTS:
            fr, z = data[clip]
            for i in range(len(fr)):
                mk = unpack(z[m], i); frame = np.hstack([fr[i], overlay(fr[i], mk)])
                w.write(cv2.cvtColor(_bar(frame, f"{n:02d} {NAMES[m]} | {clip} t={t0 + i / FPS:5.1f} s | predicted {mk.mean():.0%} of the frame"), cv2.COLOR_RGB2BGR))
        w.release(); print("per-method", m, cc, flush=True)
    for clip, t0, t1 in SEGMENTS:        # grid: original + 12 methods
        fr, z = data[clip]; w, cc = _writer(RES / "grid" / f"{clip}_{t0}-{t1}s_grid.mp4", (1920, 1080), FPS)
        for i in range(len(fr)):
            tiles = [cv2.resize(fr[i], (480, 270))]; ttl = ["original"]
            for m in METHODS:
                mk = unpack(z[m], i); tiles.append(cv2.resize(overlay(fr[i], mk), (480, 270))); ttl.append(f"{NAMES[m]} {mk.mean():.0%}")
            for k, (t, ti) in enumerate(zip(ttl, tiles)):
                cv2.rectangle(ti, (0, 0), (480, 24), (28, 28, 28), -1); cv2.putText(ti, t, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info = np.full((270, 480, 3), 28, np.uint8)
            for k, line in enumerate([f"{clip}  t = {t0 + i / FPS:5.1f} s", "orange = predicted subject", "number = share of the frame", "no smoothing between frames"]):
                cv2.putText(info, line, (14, 60 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            tiles += [info] + [np.zeros((270, 480, 3), np.uint8)] * 2
            canvas = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
            w.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        w.release(); print("grid", clip, cc, flush=True)
    seg = "\n".join(f"| {c} | {a}-{b} s |" for c, a, b in SEGMENTS); fg = pd.DataFrame(rows).pivot(index="name", columns="clip", values="mean_fg").loc[[NAMES[m] for m in METHODS]]
    (RES / "README.md").write_text(
        "# Videos of every method on the tank footage\n\n"
        "Six 30 s segments, one per underwater clip, played at real-time speed with 5 frames per second (every sixth frame of the 30 fps footage).\n\n| clip | segment |\n|---|---|\n" + seg + "\n\n"
        "* `per_method/NN_<method>.mp4`: original | prediction (orange overlay with white outline), the six segments joined one after the other; the title bar shows method, clip, time and the share of the frame predicted as subject.\n"
        "* `grid/<clip>_<segment>_grid.mp4`: the original and all 12 methods tiled for the same segment, each tile titled with the method and its predicted share.\n"
        "* `foreground_share.csv`: mean predicted share of the frame per segment and method.\n"
        "* `earlier/`: the two first demo videos (unified C only).\n\n"
        "Methods: 1 k-means (border rule), 2 dendrogram (gap rule), 3-5 earlier Stage A / B / C (salient channel of the two-channel model, Sec. 21), 6 earlier Stage C not-water channel, "
        "7-9 unified A / B / C (Sec. 23), 10-12 new A / B / C (Sec. 26). Every model is frozen and each frame is predicted on its own, so masks may flicker between frames (no temporal smoothing).\n\n"
        "Not included: Stage D (SAM refinement; it gave no gain and equals Stage C), the k-means oracle (needs ground truth), test-time augmentation.\n\n"
        "No ground truth exists for these segments, so the videos show behaviour, not accuracy; scores on the tank frames are in `../report.md` and `results/subject_v3/tank/`.\n\n"
        "## Mean predicted share of the frame\n\n" + fg.round(3).mul(100).round(1).astype(str).add("%").to_markdown() + "\n")
    print("render done")


if __name__ == "__main__":
    {"frames": cmd_frames, "sup": cmd_sup, "base": cmd_base, "check": cmd_check, "render": cmd_render}[sys.argv[1]]()
