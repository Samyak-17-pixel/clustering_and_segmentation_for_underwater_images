"""Ground-truth loading shared by the DINOv2 scripts (USOD binary masks, SUIM derived masks)."""
from pathlib import Path

import numpy as np
from PIL import Image

GT_MODES = ("usod", "suim_nonwater", "suim_saliency", "suim_subject", "suim_subject2")

# SUIM class ids from the RGB bits (R=4, G=2, B=1): HD=1 PF=2 WR=3 RO=4 RI=5 FV=6 SR=7, BW=0.
SALIENT_CLASSES = (1, 3, 4, 6)   # divers, wrecks, robots, fish
REEF_CLASSES = (2, 5)            # plants, reefs
MIN_CLASS_FRAC = 0.002           # a class must cover this share of the image to decide the branch


def suim_class_map(path):
    rgb = np.asarray(Image.open(path).convert("RGB")) > 127
    return rgb[..., 0] * 4 + rgb[..., 1] * 2 + rgb[..., 2] * 1


def suim_subject(cls):
    """Subject rule: salient classes if present; otherwise reef/plants; otherwise nothing. Returns (mask, branch)."""
    for branch, ids in ((1, SALIENT_CLASSES), (2, REEF_CLASSES)):
        m = np.isin(cls, ids)
        if m.mean() >= MIN_CLASS_FRAC:
            return m, branch
    return np.zeros(cls.shape, bool), 3


def suim_subject2(cls):
    """Subject rule v3: only divers, wrecks, robots and fish are subjects; reef, plants, sand and water are background.
    Scenes without such a class (above MIN_CLASS_FRAC of the image) have no subject. Returns (mask, branch) with branch 1 or 3."""
    m = np.isin(cls, SALIENT_CLASSES)
    if m.mean() >= MIN_CLASS_FRAC:
        return m, 1
    return np.zeros(cls.shape, bool), 3


def load_gt(data, stem, mode="usod"):
    """Return a boolean foreground mask for image `stem` under `data`/masks."""
    masks = Path(data) / "masks"
    if mode == "usod":
        return np.asarray(Image.open(masks / f"{stem}.png").convert("L")) > 128
    if mode == "suim_saliency":
        return np.asarray(Image.open(masks / "Saliency" / f"{stem}.bmp").convert("L")) > 128
    if mode == "suim_nonwater":
        # SUIM colour code: BW (background waterbody) = black (0, 0, 0); HD = blue; foreground = every other class.
        rgb = np.asarray(Image.open(masks / f"{stem}.bmp").convert("RGB")).astype(int)
        is_bw = rgb.max(axis=-1) < 128
        return ~is_bw
    if mode == "suim_subject":
        return suim_subject(suim_class_map(masks / f"{stem}.bmp"))[0]
    if mode == "suim_subject2":
        return suim_subject2(suim_class_map(masks / f"{stem}.bmp"))[0]
    raise ValueError(f"unknown gt mode {mode!r}")


def water_fraction(data, stem):
    """Fraction of SUIM pixels labelled BW (background waterbody, black)."""
    rgb = np.asarray(Image.open(Path(data) / "masks" / f"{stem}.bmp").convert("RGB"))
    return float((rgb.max(axis=-1) < 128).mean())


def drop_degenerate(paths, data, mode, min_water_frac, out):
    """Drop images whose GT is degenerate: SUIM non-water GT with < `min_water_frac` water
    (all-foreground), or an empty SUIM saliency mask."""
    if mode == "suim_nonwater" and min_water_frac > 0:
        bad = lambda p: water_fraction(data, p.stem) < min_water_frac
        why = f"< {min_water_frac:.0%} water"
    elif mode == "suim_saliency":
        bad = lambda p: not load_gt(data, p.stem, mode).any()
        why = "an empty saliency mask"
    else:
        return paths
    keep, dropped = [], []
    for p in paths:
        (dropped if bad(p) else keep).append(p)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "excluded_images.txt").write_text("".join(f"{p.name}\n" for p in dropped))
    print(f"excluded {len(dropped)} images with {why}: {[p.stem for p in dropped]}")
    return keep
