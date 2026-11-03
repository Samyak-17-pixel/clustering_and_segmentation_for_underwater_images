"""Ground-truth loading shared by the DINOv2 scripts (USOD binary masks, SUIM derived masks)."""
from pathlib import Path

import numpy as np
from PIL import Image

GT_MODES = ("usod", "suim_nonwater", "suim_saliency")


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
