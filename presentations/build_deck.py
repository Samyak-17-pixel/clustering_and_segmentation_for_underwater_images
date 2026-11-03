"""Builds presentations/Review-2-Training-Free-Underwater-Object-Segmentation.pptx from the corrected Review 1 deck.

Slides 1-2 and the THANK YOU slide are the base deck's own slides (slide 1 gets the Review 2 label and 4 student lines);
slides 3-15 are new and use the base deck's fonts, colours and spacing (geometry below is written for a 12.19M-EMU-wide
slide and scaled by S = 1.5 to the base deck's 18.29M-EMU canvas). Run with the system python3 (python-pptx), after build_figures.py:

    python3 presentations/build_deck.py
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from lxml import etree
from pptx.util import Emu, Pt

REPO = Path(__file__).resolve().parents[1]
PPTS = REPO / "presentations"
FIG = PPTS / "figures"
TEMPLATE = PPTS / "Training-Free-Underwater-Object-Segmentation-Using-Pretrained-Vision-Transformers_review_2.pptx.pptx"
OUT = PPTS / "Review-2-Training-Free-Underwater-Object-Segmentation.pptx"
META = json.loads((FIG / "meta.json").read_text())

# template geometry and style
L, W = 457189, 11277318          # left margin, content width (EMU)
TOP_LABEL, TOP_TITLE = 471011, 772008
Y0, Y1 = 1500000, 6400000        # content area
HEAD, BODY = "Arimo", "M Plus 1m"
S = 1.5                          # base deck canvas is 1.5x the geometry used below
BG = "0D0B24"
WHITE, BODY_C, ACCENT, TEAL, MUTED = "FFFFFF", "E6EAF2", "6FD3F2", "0B5C75", "A9B4C8"
GREEN, ROSE, AMBER = "7BE0A8", "E8788A", "F2C46F"


def E(v):
    return Emu(int(round(v * S)))


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- numbers (all from the result files)
def csv_rows(path):
    return list(csv.DictReader(open(path)))


def km_mean(folder, feat, key):
    rows = [r for r in csv_rows(REPO / "results" / "usod" / folder / "metrics_union.csv") if r["feat"] == feat]
    return float(np.mean([float(r[key]) for r in rows]))


N = {}
for folder, tag in (("cluster_union_stride14_k4", "k4"), ("cluster_union_stride14_k6", "k6"), ("cluster_union_stride3_k6", "s3k6")):
    for feat in ("768", "rgb"):
        N[tag, feat, "oracle"] = km_mean(folder, feat, "oracle_iou")
        N[tag, feat, "border"] = km_mean(folder, feat, "border_iou")
ORACLE_BY_K = {feat: [km_mean("cluster_union_stride14_k6", feat, f"oracle_iou_k{k}") for k in range(2, 7)] for feat in ("768", "rgb")}
SUMMARY = {(r["feat"], r["spatial"], r["linkage"]): r for r in csv_rows(REPO / "results" / "dendrogram" / "usod/summary_s3.csv")}
STATS = {(r["feat"], r["spatial"], r["linkage"], r["rule"]): r for r in csv_rows(REPO / "results" / "dendrogram" / "usod/stats_s3.csv")}
BEST = STATS["768", "free", "average", "gap_border"]
_base = {r["image"]: float(r["border_iou"]) for r in csv_rows(REPO / "results/usod/cluster_union_stride3_k6/metrics_union.csv") if r["feat"] == "768"}
_new = {r["image"]: float(r["iou"]) for r in csv_rows(REPO / "results" / "dendrogram" / "usod/metrics_s3.csv")
        if (r["feat"], r["spatial"], r["linkage"], r["rule"]) == ("768", "free", "average", "gap_border")}
_d = np.array([_new[i] - _base[i] for i in _new])
WIN = dict(better=float((_d > 0.02).mean()), worse=float((_d < -0.02).mean()))
SUIM_BEST = {(r["feat"], r["spatial"], r["linkage"], r["rule"]): r
             for r in csv_rows(REPO / "results" / "dendrogram" / "suim_saliency/stats_s3.csv")}["768", "free", "average", "gap_border"]


_k4 = {r["image"]: float(r["border_iou"]) for r in csv_rows(REPO / "results/usod/cluster_union_stride14_k4/metrics_union.csv") if r["feat"] == "768"}
_k6 = {r["image"]: float(r["border_iou"]) for r in csv_rows(REPO / "results/usod/cluster_union_stride14_k6/metrics_union.csv") if r["feat"] == "768"}
SHARE_GOOD_K4 = float(np.mean([v >= 0.70 for v in _k4.values()]))
N_BETTER_K6 = sum(_k6[i] - _k4[i] > 0.10 for i in _k4)
N_WORSE_K6 = sum(_k4[i] - _k6[i] > 0.10 for i in _k4)
_o4 = {r["image"]: float(r["oracle_iou"]) for r in csv_rows(REPO / "results/usod/cluster_union_stride14_k4/metrics_union.csv") if r["feat"] == "768"}
_o6 = {r["image"]: float(r["oracle_iou"]) for r in csv_rows(REPO / "results/usod/cluster_union_stride14_k6/metrics_union.csv") if r["feat"] == "768"}
N_ORACLE_BETTER_K6 = sum(_o6[i] - _o4[i] > 0.10 for i in _o4)


def f2(x):
    return f"{float(x):.2f}"


def f3(x):
    return f"{float(x):.3f}"


# ---------------------------------------------------------------- drawing helpers
def set_run(run, size, color, font=BODY, bold=False, spc=None):
    if bold and font == BODY:
        font, bold = "M Plus 1m Bold", False
    run.font.size = Pt((size) * S)
    run.font.name = font
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    rPr = run.font._element
    if spc:
        rPr.set("spc", str(spc))
    for tag in ("a:ea", "a:cs"):
        el = rPr.makeelement(f"{{http://schemas.openxmlformats.org/drawingml/2006/main}}{tag[2:]}", {"typeface": font})
        rPr.append(el)


def text(slide, x, y, w, h, paras, size=11, color=BODY_C, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=0, line=1.08, name=None, wrap=True, spc=None):
    """paras: str or list of str / list of (text, overrides) / list of list of runs [(text, overrides)]."""
    tb = slide.shapes.add_textbox(E(x), E(y), E(w), E(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True if wrap is None else wrap
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt((space_after) * S)
        runs = para if isinstance(para, list) else [para]
        for r in runs:
            t, o = (r, {}) if isinstance(r, str) else r
            run = p.add_run()
            run.text = t
            set_run(run, o.get("size", size), o.get("color", color), o.get("font", font), o.get("bold", bold), o.get("spc", spc))
    return tb


def label(slide, t):
    return text(slide, L, TOP_LABEL, W, 173335, t.upper(), size=10.5, color=ACCENT, name="Label", spc=63)


def title(slide, t, size=34):
    return text(slide, L, TOP_TITLE, W, 560000, t, size=size, color=WHITE, font=HEAD, line=0.93, name="Title")


def box(slide, x, y, w, h, fill=TEAL, radius=0.09, name="Callout box"):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, E(x), E(y), E(w), E(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    s.name = name
    return s


def callout(slide, x, y, w, h, paras, size=13, color=WHITE, align=PP_ALIGN.LEFT, pad=190000, **kw):
    box(slide, x, y, w, h)
    return text(slide, x + pad, y, w - 2 * pad, h, paras, size=size, color=color, anchor=MSO_ANCHOR.MIDDLE, align=align, name="Callout text", **kw)


_badge_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)


def badged(name, badge):
    """Copy of a tile with a small label (kept on the picture so that the grid stays tidy)."""
    src = FIG / f"{name}.png"
    if not badge:
        return src
    out = FIG / "badged" / f"{name}__{hashlib.md5(badge.encode()).hexdigest()[:8]}.png"
    out.parent.mkdir(exist_ok=True)
    im = Image.open(src).convert("RGBA")
    d = ImageDraw.Draw(im, "RGBA")
    tw = d.textlength(badge, font=_badge_font)
    d.rounded_rectangle([14, 14, 14 + tw + 28, 14 + 52], radius=14, fill=(13, 11, 36, 215))
    d.text((28, 22), badge, font=_badge_font, fill=(255, 255, 255, 255))
    im.save(out)
    return out


def tile(slide, name, x, y, w, badge=None):
    pic = slide.shapes.add_picture(str(badged(name, badge)), E(x), E(y), width=E(w))
    pic.name = name
    return pic


def row_x(n, gap, left=L, total=W):
    w = (total - (n - 1) * gap) // n
    return [left + i * (w + gap) for i in range(n)], w


def add_chart(slide, x, y, w, h, categories, series, colors, number_format="0.00", legend=True, point_colors=None,
              maxv=0.8, label_size=12, cat_size=11, gap=60, kind=XL_CHART_TYPE.COLUMN_CLUSTERED):
    cd = CategoryChartData()
    cd.categories = categories
    for nm, vals in series:
        cd.add_series(nm, vals)
    gf = slide.shapes.add_chart(kind, E(x), E(y), E(w), E(h), cd)
    ch = gf.chart
    ch.font.name, ch.font.size = BODY, Pt((cat_size) * S)
    ch.font.color.rgb = rgb(BODY_C)
    ch.has_title = False
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM
        ch.legend.include_in_layout = False
        ch.legend.font.size, ch.legend.font.color.rgb, ch.legend.font.name = Pt((cat_size) * S), rgb(BODY_C), BODY
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = number_format, False
    dl.font.size, dl.font.color.rgb, dl.font.name = Pt((label_size) * S), rgb(WHITE), BODY
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
    if point_colors:
        for i, c in enumerate(point_colors):
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(c)
    cs = ch._chartSpace
    nofill = '<c:spPr xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>'
    cs.find(qn("c:chart")).addnext(etree.fromstring(nofill))
    cs.find(".//" + qn("c:plotArea")).append(etree.fromstring(nofill))
    va = ch.value_axis
    va.visible = False
    va.has_major_gridlines = False
    va.minimum_scale, va.maximum_scale = 0, maxv
    ca = ch.category_axis
    ca.format.line.color.rgb = rgb("6B7394")
    ca.tick_labels.font.size, ca.tick_labels.font.color.rgb, ca.tick_labels.font.name = Pt((cat_size) * S), rgb(BODY_C), BODY
    return gf


def rounded_photo(path, name):
    out = FIG / "badged" / name
    out.parent.mkdir(exist_ok=True)
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    im = im.crop((6, 6, w - 6, h - 6))
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=int(min(w, h) * 0.07), fill=255)
    im.putalpha(mask)
    im.save(out)
    return out


def new_slide(prs, layout, lab, ttl):
    s = prs.slides.add_slide(layout)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(BG)
    label(s, lab)
    title(s, ttl)
    return s


# ---------------------------------------------------------------- build
def main():
    prs = Presentation(str(TEMPLATE))
    layout = [l for l in prs.slide_layouts if l.name == "Blank"][0]
    ICONS, PHOTOS = FIG / "icons", FIG / "photos"      # icons come from the base deck; photos are the AUV / reef pictures

    # drop base slides 3-7 (objectives ... outcomes); keep 1, 2 and the THANK YOU slide, which is moved to the end below
    ids = prs.slides._sldIdLst
    for sld in list(ids)[2:7]:
        ids.remove(sld)
        prs.part.rels.pop(sld.rId)
    for rId, rel in list(prs.part.rels.items()):  # the base deck also lists every notes page in the presentation part's rels,
        if rel.reltype.endswith("/notesSlide"):    # which would keep the dropped slides alive; the slides keep their own notes rels
            prs.part.rels.pop(rId)
    thank_slide = prs.slides[2]                 # park its part name so that new slides cannot collide with it
    thank_slide.part.partname = PackURI("/ppt/slides/slide99.xml")

    # ---- slide 1: review label + 4 editable student lines
    s1 = prs.slides[0]
    lab = [sh for sh in s1.shapes if sh.has_text_frame and sh.text_frame.text.strip() == "PROJECT REVIEW 1"][0]
    lab.text_frame.paragraphs[0].runs[0].text = "PROJECT REVIEW 2"
    lines = [f"[Student name {i}] · [Roll number {i}]" for i in range(1, 5)]
    lines.append("Department of Naval Architecture and Ocean Engineering, IIT Madras · Machine Learning for Ocean Engineers · [Project guide, if applicable]")
    text(s1, L, 5600000, W, 900000, lines, size=10.5, color=WHITE, name="Students")

    # ---- slide 3: why it matters
    s = new_slide(prs, layout, "WHY IT MATTERS", "Why it matters, and the cost of ignoring it")
    rows = [("image4.png", "Labels are scarce", "Marking every pixel of underwater photos by hand is slow and expensive. USOD has only 300 images."),
            ("image7.png", "Water hides the object", "Light absorption, haze and colour cast defeat hand-made colour thresholds."),
            ("image6.png", "Vehicles need a cue", "An AUV that cannot tell something is ahead cannot avoid it, inspect it or follow it.")]
    y = Y0 + 100000
    for icon, head, body in rows:
        s.shapes.add_picture(str(ICONS / icon), E(L), E(y), width=E(440000)).name = "Icon"
        text(s, L + 650000, y - 20000, 6200000, 420000, head, size=20, font=HEAD, color=WHITE)
        text(s, L + 650000, y + 400000, 6200000, 760000, body, size=12)
        y += 1250000
    px, pw = L + W - 3900000, 3900000
    s.shapes.add_picture(str(PHOTOS / "auv.png"), E(px), E(Y0 + 40000), width=E(pw)).name = "AUV photo"
    s.shapes.add_picture(str(PHOTOS / "reef.png"), E(px), E(Y0 + 40000 + 2190000 + 100000), width=E(pw)).name = "Reef photo"
    callout(s, L, 5250000, 7000000, 1150000,
            [[("If we do not solve it: ", {"bold": True, "color": ACCENT}),
              "every new site or camera needs fresh labelled data, or the vehicle stays blind to what is in front of it."],
             [("AUV use is the motivation here, not yet a demonstrated result.", {"size": 10.5, "color": MUTED})]], size=13, space_after=4)

    # ---- slide 4: the method in plain words
    s = new_slide(prs, layout, "THE METHOD IN PLAIN WORDS", "How the method works, step by step")
    xs, w = row_x(5, 250000)
    steps = [("usod_test5_photo", "1 · Photo", "An underwater image goes in."),
             ("pipeline_patches", "2 · Patches", "Cut into 14 × 14 pixel squares: 42 × 32 = 1,344 patches."),
             ("usod_test5_pca14", "3 · Fingerprints", "DINOv2 gives each patch 768 numbers. Shown here squeezed into 3 colours."),
             ("pipeline_clusters", "4 · Groups", "Patches that look alike are put in the same group."),
             ("pipeline_mask", "5 · Object", "Pick the groups that are the object. That is the mask.")]
    for x, (nm, head, body) in zip(xs, steps):
        text(s, x, Y0, w, 330000, head, size=16, font=HEAD, color=WHITE)
        tile(s, nm, x, Y0 + 400000, w)
        text(s, x, Y0 + 400000 + int(w * 0.75) + 120000, w, 900000, body, size=12)
    for x in xs[1:]:
        text(s, x - 245000, Y0 + 400000 + int(w * 0.75) // 2 - 150000, 240000, 300000, "›", size=24, color=ACCENT, align=PP_ALIGN.CENTER, wrap=False)
    callout(s, L, 5250000, W, 1000000,
            [[("Think of sorting jigsaw pieces by how they look, ", {"bold": True, "color": ACCENT}),
              "then deciding which piles are the fish and which are the water. Nothing is trained: the DINOv2 model is used as it is."]], size=14)

    # ---- slide 5: data and metric
    s = new_slide(prs, layout, "DATA AND METRIC", "What we measure: overlap with the true mask")
    xs, w = row_x(4, 250000)
    ex = META["iou_example"]
    parts = [("usod_test5_photo", "USOD photo", None), ("usod_test5_gt", "Ground truth (drawn by annotators)", None),
             ("pipeline_mask", "Our mask", None), ("iou_overlap", "Overlap", None)]
    for x, (nm, cap, b) in zip(xs, parts):
        tile(s, nm, x, Y0, w)
        if nm == "iou_overlap":
            text(s, x, Y0 + int(w * 0.75) + 100000, w, 500000,
                 [[("hit ", {"color": GREEN, "bold": True}), ("false alarm ", {"color": ROSE, "bold": True}), ("missed", {"color": AMBER, "bold": True})]], size=12)
        else:
            text(s, x, Y0 + int(w * 0.75) + 100000, w, 500000, cap, size=12)
    callout(s, L, 4300000, 6700000, 1900000,
            [[("IoU = overlap ÷ area covered by either mask", {"bold": True, "size": 18, "font": HEAD})],
             [(f"This image: {ex['overlap']:,} ÷ {ex['union']:,} = {ex['iou']:.2f}", {"size": 16, "color": ACCENT})],
             [("1 means a perfect match, 0 means no overlap.", {"size": 12})]], size=14, space_after=6)
    text(s, 7500000, 4300000, 4234000, 1900000,
         [[("300", {"size": 40, "font": HEAD, "color": ACCENT}), ("  USOD underwater images, each with a true object mask", {"size": 12})],
          [("Every number later is a mean IoU over these 300 images.", {"size": 12})]], space_after=10, anchor=MSO_ANCHOR.MIDDLE)

    # ---- slide 6: PCA-RGB stride 14 vs 3
    s = new_slide(prs, layout, "FEATURE MAPS", "PCA-RGB maps: stride 14, then stride 3")
    tw, th, gap = 1930000, 1447500, 100000
    heads = ["Photo", "Stride 14: blocky", "Stride 3: smoother"]
    for i, hd in enumerate(heads):
        text(s, L + i * (tw + gap), Y0 - 40000, tw, 250000, hd, size=11, color=ACCENT)
    for r, stem in enumerate(("usod_test5", "usod_test205", "usod_test106")):
        yy = Y0 + 250000 + r * (th + gap)
        for i, nm in enumerate((f"{stem}_photo", f"{stem}_pca14", f"{stem}_pca3")):
            tile(s, nm, L + i * (tw + gap), yy, tw)
    xr = 6900000
    wr = L + W - xr
    text(s, xr, Y0 + 250000, wr, 1300000,
         [[("What you see: ", {"bold": True, "color": ACCENT}), "three colours made from the 768 numbers of every patch. Same colour means similar patch: diver, water and reef separate without any training."]], size=13)
    text(s, xr, Y0 + 1700000, wr, 1300000,
         [[("Stride 3: ", {"bold": True, "color": ACCENT}), "we slide the patch grid by 3 pixels instead of 14, so the map has finer detail and looks much better."]], size=13)
    callout(s, xr, Y0 + 3000000, wr, 1750000,
            [[("But the score barely moves: ", {"bold": True, "color": ACCENT}),
              f"best-case IoU {N['k6','768','oracle']:.3f} → {N['s3k6','768','oracle']:.3f} (raw 768-D), at about 17× the backbone time."]], size=13)

    # ---- slide 7: k-means up to 4 clusters (the same two images are used on slide 8)
    PAIR = ("usod_test279", "usod_test81")
    s = new_slide(prs, layout, "K-MEANS · UP TO 4 CLUSTERS", "k-means with up to 4 groups")
    xs, w = row_x(5, 100000)
    heads = ["Photo", "True mask", "Groups found (k ≤ 4)", "Best choice of groups (uses true mask)", "Our mask (no labels used)"]
    for x, hd in zip(xs, heads):
        text(s, x, Y0 - 40000, w, 400000, hd, size=11, color=ACCENT)
    for r, stem in enumerate(PAIR):
        yy = Y0 + 380000 + r * (int(w * 0.75) + 100000)
        k = META["kmeans14"][stem]["4"]
        tile(s, f"{stem}_photo", xs[0], yy, w)
        tile(s, f"{stem}_gt", xs[1], yy, w)
        tile(s, f"{stem}_km14_k4_clusters", xs[2], yy, w, badge=f"{k['k']} groups")
        tile(s, f"{stem}_km14_k4_oracle", xs[3], yy, w, badge=f"IoU {k['oracle_iou']:.2f}")
        tile(s, f"{stem}_km14_k4_mask", xs[4], yy, w, badge=f"IoU {k['real_iou']:.2f}")
    callout(s, L, 5750000, W, 600000,
            [[("With 4 groups the fish and the diver are split over several groups, so even the best choice is rough. ", {"bold": True, "color": ACCENT}),
              f"Mean over 300 images (raw 768-D): best possible {N['k4','768','oracle']:.3f} · without labels {N['k4','768','border']:.3f}.   ",
              ("PCA-RGB: ", {"color": ACCENT, "bold": True}), f"{N['k4','rgb','oracle']:.3f} · {N['k4','rgb','border']:.3f}"]], size=11.5)

    # ---- slide 8: k-means up to 6 clusters (same two images as slide 7; k <= 4 and k <= 6 side by side, no chart)
    s = new_slide(prs, layout, "K-MEANS · UP TO 6 CLUSTERS", "Same images: 4 groups versus 6 groups")
    gap = 70000
    tw = (W - 6 * gap) // 7
    th = int(tw * 0.75)
    cols = [x0 + i * (tw + gap) for i, x0 in enumerate([L] * 7)]
    heads = ["Photo", "Groups", "Best choice", "Our mask", "Groups", "Best choice", "Our mask"]
    for i, hd in enumerate(heads):
        text(s, cols[i], Y0 + 330000, tw, 250000, hd, size=10, color=ACCENT)
    text(s, cols[1], Y0 - 40000, 3 * tw + 2 * gap, 330000, "k ≤ 4", size=16, font=HEAD, color=WHITE)
    text(s, cols[4], Y0 - 40000, 3 * tw + 2 * gap, 330000, "k ≤ 6", size=16, font=HEAD, color=WHITE)
    for r, stem in enumerate(PAIR):
        yy = Y0 + 640000 + r * (th + 100000)
        k4, k6 = META["kmeans14"][stem]["4"], META["kmeans14"][stem]["6"]
        tile(s, f"{stem}_photo", cols[0], yy, tw)
        tile(s, f"{stem}_km14_k4_clusters", cols[1], yy, tw, badge=f"{k4['k']} groups")
        tile(s, f"{stem}_km14_k4_oracle", cols[2], yy, tw, badge=f"IoU {k4['oracle_iou']:.2f}")
        tile(s, f"{stem}_km14_k4_mask", cols[3], yy, tw, badge=f"IoU {k4['real_iou']:.2f}")
        tile(s, f"{stem}_km14_k6_clusters", cols[4], yy, tw, badge=f"{k6['k']} groups")
        tile(s, f"{stem}_km14_k6_oracle", cols[5], yy, tw, badge=f"IoU {k6['oracle_iou']:.2f}")
        tile(s, f"{stem}_km14_k6_mask", cols[6], yy, tw, badge=f"IoU {k6['real_iou']:.2f}")
    best = " and ".join(f"{META['kmeans14'][st]['4']['oracle_iou']:.2f} → {META['kmeans14'][st]['6']['oracle_iou']:.2f}" for st in PAIR)
    ours = " and ".join(f"{META['kmeans14'][st]['4']['real_iou']:.2f} → {META['kmeans14'][st]['6']['real_iou']:.2f}" for st in PAIR)
    callout(s, L, 5200000, W, 1150000,
            [[("More groups give a much better best choice: ", {"bold": True, "color": ACCENT}),
              f"{best} on these two images (mean {N['k4','768','oracle']:.3f} → {N['k6','768','oracle']:.3f}; better by more than 0.10 on {N_ORACLE_BETTER_K6} of 300 images)."],
             [("But our mask, with no labels, hardly follows: ", {"bold": True, "color": ACCENT}),
              f"{ours} here, and the mean over 300 images falls {N['k4','768','border']:.3f} → {N['k6','768','border']:.3f}."]], size=12, space_after=4)

    # ---- slide 9: the real problem
    s = new_slide(prs, layout, "THE REAL PROBLEM", "The groups are good; choosing them is hard")
    lw = 7000000
    w = (lw - 2 * 100000) // 3
    for i, c in enumerate(["Photo", "Best choice", "Our choice (no labels)"]):
        text(s, L + i * (w + 100000), Y0 - 40000, w, 400000, c, size=11, color=ACCENT)
    for r, stem in enumerate(("usod_test107", "usod_test106")):
        m = META["s3"][stem]
        yy = Y0 + 330000 + r * (int(w * 0.75) + 110000)
        tile(s, f"{stem}_photo", L, yy, w)
        tile(s, f"{stem}_km3_oracle", L + w + 100000, yy, w, badge=f"IoU {m['km_oracle']:.2f}")
        tile(s, f"{stem}_km3_real", L + 2 * (w + 100000), yy, w, badge=f"IoU {m['km_real']:.2f}")
    xr = L + lw + 350000
    add_chart(s, xr, Y0 - 40000, L + W - xr, 2900000, ["raw 768-D", "PCA-RGB"],
              [("Best choice", [round(N["s3k6", "768", "oracle"], 3), round(N["s3k6", "rgb", "oracle"], 3)]),
               ("Our choice (no labels)", [round(N["s3k6", "768", "border"], 3), round(N["s3k6", "rgb", "border"], 3)])],
              [AMBER, ACCENT], maxv=0.95, label_size=11, cat_size=10)
    callout(s, xr, Y0 + 2950000, L + W - xr, 1900000,
            [[(f"{N['s3k6','768','oracle']:.2f} vs {N['s3k6','768','border']:.2f}", {"size": 30, "font": HEAD, "color": WHITE})],
             [("Mean over 300 images. The groups already contain a good mask; our rule for picking the object is the weak link, so we try a dendrogram.", {"size": 11})]],
            space_after=4)

    # ---- slide 10: dendrogram explained
    s = new_slide(prs, layout, "DENDROGRAMS", "A dendrogram is a family tree of patches")
    s.shapes.add_picture(str(FIG / "toy_tree.png"), E(L), E(Y0), width=E(5300000)).name = "Toy tree"
    text(s, L, Y0 + 3600000, 5300000, 1250000,
         [[("1  ", {"color": ACCENT, "bold": True}), "Every patch starts alone; glue the closest groups, again and again."],
          [("2  ", {"color": ACCENT, "bold": True}), "Cutting the tree at a height gives k groups."],
          [("3  ", {"color": ACCENT, "bold": True}), "One tree gives every k: 2 groups, 3, 4 … each one a split of the last."]], size=12, space_after=6)
    gx, gw = 6100000, 2760000
    for i, (nm, bd) in enumerate((("usod_test153_photo", "photo"), ("tree_cut_k2", "2 groups"), ("tree_cut_k3", "3 groups"), ("tree_cut_k4", "4 groups"))):
        tile(s, nm, gx + (i % 2) * (gw + 120000), Y0 + (i // 2) * (int(gw * 0.75) + 120000), gw, badge=bd)
    text(s, gx, Y0 + 2 * (int(gw * 0.75) + 120000), 2 * gw + 120000, 500000, "A real image: one tree, cut at three heights.", size=11, color=MUTED)

    # ---- slide 11: variants
    s = new_slide(prs, layout, "DENDROGRAM VARIANTS", "Four ways to measure 'closest' between groups")
    xs, w = row_x(4, 120000)
    m = META["s3"]["usod_test153"]
    h = int(w * 0.75)
    tile(s, "usod_test153_photo", xs[0], Y0, w)
    tile(s, "usod_test153_gt", xs[1], Y0, w)
    tile(s, "usod_test153_km3_real", xs[2], Y0, w, badge=f"k-means {m['km_real']:.2f}")
    text(s, xs[3], Y0, w, h, [[("Same photo, same rule for picking groups. ", {"bold": True, "color": ACCENT}), "Only the way groups are compared changes."]], size=13, anchor=MSO_ANCHOR.MIDDLE)
    for x, (lk, nm) in zip(xs, (("ward", "Ward"), ("average", "Average"), ("complete", "Complete"), ("single", "Single"))):
        tile(s, f"usod_test153_dendro_{lk}_gap", x, Y0 + h + 100000, w, badge=(f"{nm} {m['dendro'][lk]:.2f}" + (": all marked object" if lk == "single" else "")))
    sg = SUMMARY["768", "free", "single"]
    callout(s, L, 5850000, W, 780000,
            [[("Why Single is all white: ", {"bold": True, "color": ACCENT}),
              "it chains patches into one huge group (99.7% of the patches), and the few left out sit on the border, so everything else is called object. ",
              f"The score is then just the fish's share of the picture, {m['dendro']['single']:.2f} (mean over 300 images {float(sg['gap_border_iou']):.2f}).  ",
              ("Average ", {"bold": True, "color": ACCENT}), f"works best on USOD ({float(SUMMARY['768','free','average']['gap_border_iou']):.2f})."]], size=11.5)

    # ---- slide 12: results summary
    s = new_slide(prs, layout, "RESULTS", "Dendrogram versus k-means on USOD")
    avg, ward, comp = (SUMMARY["768", "free", k] for k in ("average", "ward", "complete"))
    cats = ["k-means", "Ward", "Complete", "Average", "Avg + gap cut", "Best possible"]
    vals = [N["s3k6", "768", "border"], float(ward["sil_border_iou"]), float(comp["sil_border_iou"]), float(avg["sil_border_iou"]),
            float(avg["gap_border_iou"]), float(avg["oracle_sil_iou"])]
    add_chart(s, L, Y0 - 50000, 7100000, 4300000, cats, [("IoU", [round(v, 3) for v in vals])], [ACCENT], legend=False, maxv=0.85,
              point_colors=["8A93B2", ACCENT, ACCENT, ACCENT, GREEN, AMBER], label_size=14, cat_size=11, gap=45)
    text(s, L, Y0 + 4300000, 7100000, 500000, "Mean IoU on 300 USOD images, stride 3, raw 768-D, no labels used (grey: k-means, amber: upper bound).", size=10.5, color=MUTED)
    xr = 7900000
    wr = L + W - xr
    text(s, xr, Y0, wr, 1500000,
         [[(f"+{float(BEST['diff']):.2f}", {"size": 54, "font": HEAD, "color": GREEN})],
          [(f"mean IoU gain over k-means (95% interval +{float(BEST['ci_lo']):.2f} to +{float(BEST['ci_hi']):.2f})", {"size": 12})]], space_after=2)
    text(s, xr, Y0 + 1750000, wr, 1300000,
         [[(f"{100 * WIN['better']:.0f}% / {100 * WIN['worse']:.0f}%", {"size": 34, "font": HEAD, "color": WHITE})], [("of images clearly better / clearly worse than with k-means", {"size": 12})]], space_after=2)
    callout(s, xr, Y0 + 3250000, wr, 1500000,
            [[("Same trend on SUIM: ", {"bold": True, "color": ACCENT}),
              f"+{float(SUIM_BEST['diff']):.2f} on 108 images with its salient-object masks."]], size=12.5)

    # ---- slide 13: where it wins
    s = new_slide(prs, layout, "WHERE IT WINS", "Where the dendrogram fixes k-means mistakes")
    tw, gap = 1950000, 100000
    th = int(tw * 0.75)
    cols = ["Photo", "True mask", "k-means", "Dendrogram"]
    for i, c in enumerate(cols):
        text(s, L + i * (tw + gap), Y0 - 40000, tw, 250000, c, size=11, color=ACCENT)
    for r, stem in enumerate(("usod_test205", "usod_test53", "usod_test2")):
        yy = Y0 + 250000 + r * (th + gap)
        m = META["s3"][stem]
        tile(s, f"{stem}_photo", L, yy, tw)
        tile(s, f"{stem}_gt", L + tw + gap, yy, tw)
        tile(s, f"{stem}_km3_real", L + 2 * (tw + gap), yy, tw, badge=f"IoU {m['km_real']:.2f}")
        tile(s, f"{stem}_dendro_average_gap", L + 3 * (tw + gap), yy, tw, badge=f"IoU {m['dendro']['average']:.2f}")
    xr = L + 4 * (tw + gap) + 250000
    callout(s, xr, Y0 + 900000, L + W - xr, 3000000,
            [[("k-means picks the wrong group ", {"bold": True, "color": ACCENT}), "(IoU 0.07 to 0.11) on these three images."],
             [("The dendrogram, with average linkage and the gap cut, finds the object ", {"bold": True, "color": ACCENT}), "(IoU 0.72 to 0.84)."]],
            size=13, space_after=14)

    # ---- slide 14: failure
    s = new_slide(prs, layout, "A FAILURE, AND WHY", "When it fails: the same method on usod_test40")
    f = META["failure"]
    xs, w = row_x(4, 120000)
    h = int(w * 0.75)
    tile(s, "usod_test40_photo", xs[0], Y0, w)
    tile(s, "usod_test40_gt", xs[1], Y0, w)
    tile(s, "usod_test40_gap_mask", xs[2], Y0, w, badge=f"gap cut, 2 groups: IoU {f['iou_gap']:.2f}")
    tile(s, "usod_test40_sil_mask", xs[3], Y0, w, badge=f"silhouette, 4 groups: IoU {f['iou_sil']:.2f}")
    ty = Y0 + h + 100000
    s.shapes.add_picture(str(FIG / "failure_tree.png"), E(L), E(ty), height=E(Y1 - ty)).name = "Failure tree"
    c2 = f["counts"]["2"]
    text(s, 5100000, ty, L + W - 5100000, Y1 - ty,
         [[("Why: ", {"bold": True, "color": ACCENT}),
           f"the gap rule cuts at 2 groups, but the top split of the tree separates only {c2[0]} of {f['n_patches']:,} patches ({100 * c2[0] / f['n_patches']:.2f}%), a few odd patches."],
          [("The rest is then called the object, ", {"bold": True, "color": ACCENT}), "so almost the whole picture becomes object (IoU 0.21)."],
          [("Cutting at 4 groups works (0.83). ", {"bold": True, "color": ACCENT}), "About 1 image in 5 gets worse than with k-means."]],
         size=13, space_after=9)

    # ---- slide 15: conclusion and plan
    s = new_slide(prs, layout, "CONCLUSION AND PLAN", "Conclusion and plan for Review 3")
    xs, w = row_x(3, 200000)
    stats = [(f"{N['s3k6','768','border']:.2f} → {float(BEST['mean_iou']):.2f}", "mean IoU without labels, k-means → dendrogram (USOD)"),
             (f"{float(avg['oracle_sil_iou']):.2f}", "best possible with the same groups, so the gain is in how we choose them"),
             (f"{100 * WIN['better']:.0f}% / {100 * WIN['worse']:.0f}%", "of images clearly better / clearly worse")]
    for x, (big, small) in zip(xs, stats):
        callout(s, x, Y0, w, 1500000, [[(big, {"size": 34, "font": HEAD, "color": WHITE})], [(small, {"size": 12})]], space_after=2)
    y2 = Y0 + 1500000 + 300000
    wl = 5300000
    text(s, L, y2, wl, 3200000,
         [[("Limits", {"size": 22, "font": HEAD, "color": WHITE})],
          [("Best variant picked on the same images it is scored on (best of 64).", {"size": 14})],
          [("On SUIM 'everything except water' labels a trivial guess already scores 0.52.", {"size": 14})],
          [("Only 2 to 6 groups tried; rule constants not tuned; no real-time test yet.", {"size": 14})]], space_after=12)
    xr = L + wl + 400000
    text(s, xr, y2, L + W - xr, 3200000,
         [[("Plan for Review 3  ·  [date]", {"size": 22, "font": HEAD, "color": WHITE})],
          [("1  ", {"color": ACCENT, "bold": True, "size": 14}), ("Stop tiny outlier groups from winning the gap rule; test on held-out images.", {"size": 14})],
          [("2  ", {"color": ACCENT, "bold": True, "size": 14}), ("Ablations and a baseline comparison (SVAM-Net).", {"size": 14})],
          [("3  ", {"color": ACCENT, "bold": True, "size": 14}), ("Generalise: SUIM per-class evaluation and more data.", {"size": 14})],
          [("4  ", {"color": ACCENT, "bold": True, "size": 14}), ("Measure speed for onboard use; final report.", {"size": 14})]], space_after=12)

    thank = list(ids)[2]                      # the THANK YOU slide is now the third entry; move it behind the new slides
    ids.remove(thank)
    ids.append(thank)
    for i, sl in enumerate(prs.slides, 1):    # unique, in-order part names (python-pptx numbers new slides by count)
        sl.part.partname = PackURI(f"/ppt/slides/slide{i}.xml")
    prs.save(str(OUT))
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()
