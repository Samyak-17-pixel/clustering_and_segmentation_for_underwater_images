"""Empty-scene false-alarm test. SUIM has no water-only images, so negatives are crops with NO subject pixels:
  water   : >= 99% water (BW)
  seafloor: only water + sand/sea-floor classes, sand >= 30%
Crops (40% x 40% of the image, upscaled to the network input) are cut from SUIM val images (used to tune the
minimum-area rule) and SUIM test images (reported). A false alarm = predicted foreground > 1% of the crop.
Models: pooled A/B/C (unified subject label) and the label-free k-means border rule (always outputs a foreground).
"""
import json
import numpy as np, pandas as pd, torch
from peft import LoraConfig, get_peft_model
from common import *
from gt_utils import suim_class_map
from baseline_masks import kmeans_border
from export_examples import save_collage

V2 = ROOT / "results" / "subject_v2"
FA = OUT / "false_alarm"; FA.mkdir(exist_ok=True)
FRAC, MIN_FG = 0.4, 0.01
recs = build_records()


def find_crops(cls, kind):
    h, w = cls.shape
    ch, cw = int(h * FRAC), int(w * FRAC)
    if kind == "water":
        score = (cls == 0).astype(np.float32)
    else:
        score = np.isin(cls, (0, 7)).astype(np.float32)
    ii = np.pad(score.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sand = np.pad((cls == 7).astype(np.float32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = None
    for y in range(0, h - ch + 1, 8):
        for x in range(0, w - cw + 1, 8):
            f = (ii[y + ch, x + cw] - ii[y, x + cw] - ii[y + ch, x] + ii[y, x]) / (ch * cw)
            s = (sand[y + ch, x + cw] - sand[y, x + cw] - sand[y + ch, x] + sand[y, x]) / (ch * cw)
            ok = f >= 0.99 if kind == "water" else (f >= 0.999 and s >= 0.3)
            if ok and (best is None or (f if kind == "water" else s) > best[0]):
                best = ((f if kind == "water" else s), y, x, ch, cw)
    return best


def build():
    rows = []
    for r in recs:
        if r["ds"] != "suim" or r["split"] == "train": continue
        cls = suim_class_map(r["data"] / "masks" / f"{r['stem']}.bmp")
        for kind in ("water", "seafloor"):
            b = find_crops(cls[::2, ::2], kind)
            if b is None: continue
            _, y, x, ch, cw = b
            img = Image.open(r["img"]).convert("RGB"); W0, H0 = img.size
            sy, sx = H0 / cls[::2, ::2].shape[0], W0 / cls[::2, ::2].shape[1]
            box = (int(x * sx), int(y * sy), int((x + cw) * sx), int((y + ch) * sy))
            f = FA / f"{r['split']}_{kind}_{r['stem']}.png"
            img.crop(box).save(f)
            rows.append(dict(split=r["split"], kind=kind, stem=r["stem"], file=str(f)))
    d = pd.DataFrame(rows); d.to_csv(FA / "crops.csv", index=False)
    return d


def load_models():
    from common import load_backbone
    ck = {s: torch.load(OUT / "ckpt" / f"v2_pool_{s}.pt") for s in "ABC"}
    mk = lambda c: Head(**{k: v for k, v in c.items() if k not in ("lr", "wd")}, out=1).cuda().eval()
    heads = {}
    for s in "AB":
        heads[s] = mk(ck[s]["cfg"]); heads[s].load_state_dict(ck[s]["state"])
    frozen = load_backbone()
    bb = get_peft_model(load_backbone(), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["qkv", "proj"], bias="none")).eval()
    sd = bb.state_dict(); sd.update(ck["C"]["lora"]); bb.load_state_dict(sd)
    hc = mk(ck["C"]["head_cfg"]); hc.load_state_dict(ck["C"]["head"])
    return frozen, bb.base_model.model, heads, hc, {s: ck[s]["thr"]["all"] for s in "ABC"}


@torch.no_grad()
def predict(models, path):
    frozen, base, heads, hc, thr = models
    x = normalise(to_tensor(load_image(path))[None].cuda())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = extract(frozen, x).float()
        t = base.forward_features(x)["x_norm_patchtokens"]
    fc = t.float().transpose(1, 2).reshape(1, -1, H // 14, W // 14)
    return {"A": torch.sigmoid(up(heads["A"](f)))[0, 0].cpu().numpy(), "B": torch.sigmoid(up(heads["B"](f)))[0, 0].cpu().numpy(),
            "C": torch.sigmoid(up(hc(fc)))[0, 0].cpu().numpy()}


if __name__ == "__main__":
    d = build()
    print(d.groupby(["split", "kind"]).size())
    models = load_models()
    thr = models[4]
    out = []
    for _, r in d.iterrows():
        p = predict(models, r.file)
        row = dict(r)
        for s in "ABC":
            row[f"fg_{s}"] = float((p[s] > thr[s]).mean())
        km = kmeans_border(r.file, (TH, TW))
        row["fg_kmeans"] = float(km.mean()); row["_km"] = km
        row["_p"] = {s: p[s] > thr[s] for s in "ABC"}
        out.append(row)
    df = pd.DataFrame(out)
    # minimum-area rule: predicted foreground below tau (share of the image) is cleared; tau tuned on val crops
    # so that the false-alarm rate is <= 5%, then applied to test; positives re-scored from the stored pooled predictions.
    ev = [i for i, r in enumerate(recs) if r["split"] in ("val", "test")]
    evpos = {i: k for k, i in enumerate(ev)}
    taus = [0, .01, .02, .03, .05, .08, .12]
    md = ["# Empty-scene false alarms\n", f"Negatives: crops with no subject pixels (water-only: >= 99% water; sea-floor: water + sand only). A false alarm = predicted foreground > {MIN_FG:.0%} of the crop. val crops tune the minimum-area rule, test crops are reported.\n",
          "| Model | split | n crops | false-alarm rate | mean predicted fg share |", "|---|---|---|---|---|"]
    for s in ("kmeans", "A", "B", "C"):
        for sp in ("val", "test"):
            g = df[df.split == sp]; v = g[f"fg_{s}"]
            md.append(f"| {'k-means (border rule)' if s == 'kmeans' else 'model ' + s} | {sp} | {len(g)} | {(v > MIN_FG).mean():.0%} | {v.mean():.3f} |")
    md += ["", "By crop type (test, model C): " + ", ".join(f"{k}: {(g['fg_C'] > MIN_FG).mean():.0%} of {len(g)}" for k, g in df[df.split == 'test'].groupby('kind')) + "\n"]
    # tau tuning (model C)
    tune = {}
    for s in "ABC":
        val = df[df.split == "val"][f"fg_{s}"].values
        tau = next((t for t in taus if (val * (val >= t) > MIN_FG).mean() <= 0.05), taus[-1])
        tune[s] = tau
    md += ["## Minimum-area rule (tuned on val crops: smallest tau with <= 5% false alarms)\n", "| Model | tau | test false-alarm rate | USOD test IoU before -> after | SUIM test IoU before -> after |", "|---|---|---|---|---|"]
    for s in "ABC":
        P = np.load(pred_path(f"v2_pool_{s}", "evalset")).astype(np.float32)[:, 0]
        thrs = json.load(open(V2 / "pool" / f"stage_{s}" / "config.json"))["thr"]["all"]
        res = {}
        for ds in ("usod", "suim"):
            b, a = [], []
            for i in ev:
                r = recs[i]
                if r["ds"] != ds or r["split"] != "test": continue
                gt = load_gt(r["data"], r["stem"], "suim_subject" if ds == "suim" else "usod")
                pm = P[evpos[i]] > thrs
                b.append(mask_metrics(prob_to_mask(P[evpos[i]], gt.shape, thrs), gt)["iou"])
                keep = pm.mean() >= tune[s]
                a.append(mask_metrics(prob_to_mask(P[evpos[i]], gt.shape, thrs) & keep, gt)["iou"])
            res[ds] = (np.mean(b), np.mean(a))
        t = df[df.split == "test"][f"fg_{s}"].values
        md.append(f"| model {s} | {tune[s]:.0%} | {((t * (t >= tune[s])) > MIN_FG).mean():.0%} | {res['usod'][0]:.3f} -> {res['usod'][1]:.3f} | {res['suim'][0]:.3f} -> {res['suim'][1]:.3f} |")
    open(V2 / "false_alarm.md", "w").write("\n".join(md))
    df.drop(columns=["_km", "_p"]).to_csv(V2 / "false_alarm_per_crop.csv", index=False)
    # one collage per crop: the 8 test crops where model C has the largest predicted foreground
    outd = V2 / "examples" / "false_alarm"; outd.mkdir(parents=True, exist_ok=True)
    for _, r in df[df.split == "test"].sort_values("fg_C", ascending=False).head(8).iterrows():
        img = np.asarray(Image.open(r.file).convert("RGB").resize((TW, TH)))
        pan = [("k-means (border rule)", r["_km"], None)] + [(f"model {s}", r["_p"][s], None) for s in "ABC"]
        pan = [(f"{t}\nforeground {m.mean():.0%}", m, None) for t, m, _ in pan]
        save_collage(outd / f"{r.kind}_{r.stem}_fg{r.fg_C:.2f}.png", img, np.zeros((TH, TW), bool), pan, f"false alarm | {r.kind} | {r.stem} | no subject in this crop")
    print("\n".join(md))
