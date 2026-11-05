"""Build data/dhwani_dataset: ~300 frames of real underwater footage (indoor test tank, GoPro 4K) for use as a third, unseen test set.

Frames only (no masks). Protocol:
 1. In-water time windows with a scene tag per window (fish / floor / structure / waterline / empty) were fixed by looking at
    contact sheets of every clip (videos/underwater_selected/contact_sheets); tags are visual judgement, not verified by the user.
 2. Candidates every 0.5 s inside the windows (4K decoded sequentially, resized in memory to 1920 px long edge).
 3. Quality filter: not too dark / blown out; for non-empty tags sharpness above the 30th percentile of the tag x clip group.
 4. Diversity: DINOv2 CLS embeddings, greedy farthest-point sampling per tag with >= 1 s separation inside a clip and a per-clip cap per tag.
    No ground truth and no model prediction of subjects is used.
 5. Quotas (empty water capped at 30% of the dataset), JPEG q95, metadata.csv, contact sheets, README.md.

  python build_gopro_dataset.py            # writes data/dhwani_dataset
  python build_gopro_dataset.py --exclude  # also drops files listed in data/dhwani_dataset/exclude.txt (visual review) and refills quotas
"""
import argparse, csv, sys, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "supervised"))
from common import load_backbone, normalise, to_tensor, H, W  # noqa: E402
from PIL import Image  # noqa: E402

VID = ROOT / "videos" / "underwater_selected"
OUT = ROOT / "data" / "dhwani_dataset"
LONG = 1920
STEP = 0.5                      # candidate spacing (s)
MIN_GAP = 1.0                   # min spacing inside a clip (s)
REVIEW = []   # time-window corrections made after looking at the contact sheets (data/dhwani_dataset/review.csv: clip,t0,t1,action,tag,note)
if (OUT / "review.csv").exists():
    REVIEW = [(r["clip"], float(r["t0"]), float(r["t1"]), r["action"], r["tag"]) for r in csv.DictReader(open(OUT / "review.csv"))]
QUOTA = {"fish": 30, "floor": 50, "structure": 110, "waterline": 55, "empty": 80}    # 325 requested; empty is at most 30% of what is selected
SHARP_PCT = {"fish": 10, "floor": 10}   # rare tags keep more frames; others drop the blurriest 30%
DUP = {"empty": 0.985}                  # near-duplicate threshold (cosine similarity of DINOv2 CLS); default 0.97
CLIP_CAP = 0.6                  # one clip may fill at most this share of a tag's quota
# (clip, t0, t1, tag): in-water windows from the contact sheets
WINDOWS = [
    ("GX010214", 25, 52, "empty"), ("GX010214", 52, 62, "fish"), ("GX010214", 62, 68, "floor"), ("GX010214", 78, 101, "floor"),
    ("GX010214", 102, 112, "fish"), ("GX010214", 113, 163, "structure"),
    ("GX010216", 14, 104, "waterline"), ("GX010216", 106, 146, "structure"),
    ("GX010217", 24, 80, "empty"), ("GX010217", 84, 92, "structure"), ("GX010217", 92, 98, "waterline"), ("GX010217", 98, 103, "fish"),
    ("GX010217", 103, 121, "waterline"), ("GX010217", 121, 150, "structure"),
    ("GX010218", 14, 50, "empty"), ("GX010218", 50, 60, "floor"), ("GX010218", 78, 100, "floor"), ("GX010218", 102, 136, "structure"),
    ("GX010220", 14, 122, "waterline"), ("GX010220", 124, 134, "structure"),
    ("GX010221", 14, 94, "empty"), ("GX010221", 96, 138, "structure"),
]


def candidates():
    """-> list of dict(clip, t, tag, jpg(bytes), sharp, bright, bg, dark, blown) for every window, decoded once per clip."""
    out = []
    for clip in sorted({w[0] for w in WINDOWS}):
        cap = cv2.VideoCapture(str(VID / f"{clip}.MP4")); fps = cap.get(cv2.CAP_PROP_FPS)
        wins = sorted([w for w in WINDOWS if w[0] == clip], key=lambda w: w[1])
        for _, t0, t1, tag in wins:
            cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
            fi = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))); step = int(round(STEP * fps)); end = int(t1 * fps)
            while fi < end:
                ok = cap.grab()
                if not ok: break
                if (fi - int(t0 * fps)) % step == 0:
                    ok, fr = cap.retrieve()
                    if ok:
                        h, w = fr.shape[:2]; s = LONG / max(h, w)
                        fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                        g = cv2.cvtColor(cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
                        b, gc, r = (float(fr[..., k].mean()) for k in range(3))
                        out.append(dict(clip=clip, t=fi / fps, frame=fi, tag=tag, fr=fr, sharp=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                                        bright=float(g.mean()), bg=(b + gc) / (2 * r + 1e-6), dark=float((g < 20).mean()), blown=float((g > 240).mean())))
                fi += 1
        cap.release(); print(clip, "decoded", sum(c["clip"] == clip for c in out), flush=True)
    return out


def embed(cands, bs=16):
    try:
        bb = load_backbone()
    except Exception as e:      # hub check needs the network; fall back to the local cache
        print("hub load failed, using local cache:", type(e).__name__, flush=True)
        bb = torch.hub.load(str(Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"), "dinov2_vitb14", source="local").eval().cuda()
    embs = []
    with torch.no_grad():
        for i in range(0, len(cands), bs):
            x = torch.stack([to_tensor(Image.fromarray(cv2.cvtColor(c["fr"], cv2.COLOR_BGR2RGB)).resize((W, H))) for c in cands[i:i + bs]]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = bb.forward_features(normalise(x))["x_norm_clstoken"].float()
            embs.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(embs).numpy()


def select(cands, E, exclude):
    ok = np.array([0.02 <= c["bright"] / 255 <= 0.85 and c["dark"] < 0.5 and c["blown"] < 0.3 for c in cands])
    for c in cands:
        c["file"] = f"{c['clip']}_t{c['t']:07.2f}s.jpg"
        for clip, t0, t1, action, tag in REVIEW:
            if c["clip"] == clip and t0 <= c["t"] <= t1:
                if action == "retag": c["tag"] = tag
                else: exclude = exclude | {c["file"]}
    for i, c in enumerate(cands):
        if c["file"] in exclude: ok[i] = False
    # sharpness filter for non-empty tags: keep above the 30th percentile of tag x clip
    for key in {(c["tag"], c["clip"]) for c in cands}:
        ids = [i for i, c in enumerate(cands) if (c["tag"], c["clip"]) == key and ok[i]]
        if key[0] != "empty" and ids:
            thr = np.percentile([cands[i]["sharp"] for i in ids], SHARP_PCT.get(key[0], 30))
            for i in ids:
                if cands[i]["sharp"] < thr: ok[i] = False
    chosen = []
    for tag, q in QUOTA.items():
        pool = [i for i, c in enumerate(cands) if c["tag"] == tag and ok[i]]
        sel, per_clip = [], {}
        cap = max(1, int(CLIP_CAP * q))
        # farthest-point sampling: start from the sharpest frame, then maximise the distance to the chosen set
        if not pool: continue
        sel.append(max(pool, key=lambda i: cands[i]["sharp"])); per_clip[cands[sel[0]]["clip"]] = 1
        d = 1 - E[pool] @ E[sel[0]]
        dist = dict(zip(pool, d))
        while len(sel) < q:
            best = None
            for i in sorted(pool, key=lambda j: -dist[j]):
                c = cands[i]
                if i in sel or per_clip.get(c["clip"], 0) >= cap: continue
                if any(cands[j]["clip"] == c["clip"] and abs(cands[j]["t"] - c["t"]) < MIN_GAP for j in sel): continue
                if max(float(E[i] @ E[j]) for j in sel) > DUP.get(tag, 0.97): continue
                best = i; break
            if best is None: break
            sel.append(best); per_clip[cands[best]["clip"]] = per_clip.get(cands[best]["clip"], 0) + 1
            for i in pool: dist[i] = min(dist[i], 1 - float(E[i] @ E[best]))
        print(f"{tag:10s} quota {q:3d} got {len(sel):3d} from {len(pool):4d} candidates; per clip {per_clip}")
        chosen += sel
    return chosen


def sheets(cands, chosen):
    (OUT / "contact_sheets").mkdir(exist_ok=True)
    order = sorted(chosen, key=lambda i: (cands[i]["tag"], cands[i]["clip"], cands[i]["t"]))
    for s in range(0, len(order), 30):
        th = []
        for i in order[s:s + 30]:
            c = cands[i]; x = cv2.resize(c["fr"], (256, int(256 * c["fr"].shape[0] / c["fr"].shape[1])))
            x = cv2.copyMakeBorder(x, 0, 144 - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) if x.shape[0] < 144 else x[:144]
            cv2.putText(x, f"{order.index(i) + 1} {c['tag'][:4]} {c['clip'][-3:]}@{c['t']:.0f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            th.append(x)
        while len(th) % 6: th.append(np.zeros_like(th[0]))
        cv2.imwrite(str(OUT / "contact_sheets" / f"sheet_{s // 30 + 1:02d}.jpg"), np.vstack([np.hstack(th[k:k + 6]) for k in range(0, len(th), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return order


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exclude", action="store_true"); args = ap.parse_args()
    exclude = set()
    if args.exclude and (OUT / "exclude.txt").exists():
        exclude = {l.strip() for l in (OUT / "exclude.txt").read_text().splitlines() if l.strip()}
    t0 = time.time()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    cands = candidates(); print(len(cands), "candidates", round(time.time() - t0), "s", flush=True)
    E = embed(cands); print("embedded", flush=True)
    chosen = select(cands, E, exclude)
    for f in (OUT / "images").glob("*.jpg"): f.unlink()
    for i in chosen:
        c = cands[i]; cv2.imwrite(str(OUT / "images" / c["file"]), c["fr"], [cv2.IMWRITE_JPEG_QUALITY, 95])
    order = sheets(cands, chosen)
    # nearest-neighbour similarity inside the chosen set (near-duplicate check)
    Ec = E[chosen]; S = Ec @ Ec.T; np.fill_diagonal(S, -1)
    with open(OUT / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "clip", "time_s", "frame_idx", "width", "height", "sharpness", "brightness", "blue_green", "tag", "sheet_no"])
        for n, i in enumerate(order, 1):
            c = cands[i]; w.writerow([c["file"], c["clip"], f"{c['t']:.2f}", c["frame"], c["fr"].shape[1], c["fr"].shape[0], f"{c['sharp']:.1f}", f"{c['bright']:.1f}", f"{c['bg']:.3f}", c["tag"], n])
    print("selected", len(chosen), "max cosine similarity between two selected frames", round(float(S.max()), 3), "total", round(time.time() - t0), "s")
