"""Block diagram of the supervised pipeline -> results/doc_figure_sources/supervised_pipeline.png (copied to docs/figures by src/figures/make_readme_figures.py)"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FROZEN, TRAIN, NEUT, INK = "#cfe3f7", "#fde3c4", "#eeeeee", "#1f2937"
fig, ax = plt.subplots(figsize=(15, 4.6)); ax.set_xlim(0, 15); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, text, color, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)
def arrow(x0, x1, y=2.3):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
box(0.2, 1.7, 1.8, 1.2, "image\n462 x 616 x 3", NEUT); arrow(2.05, 2.55)
box(2.6, 1.3, 3.2, 2.0, "DINOv2 ViT-B/14\n(86.6 M weights)\nFROZEN in A, B\n+ 0.9 M LoRA adapters\nin C (trained)", FROZEN); arrow(5.85, 6.35)
box(6.4, 1.7, 2.0, 1.2, "patch tokens\n33 x 44 x 768", NEUT); arrow(8.45, 8.95)
box(9.0, 1.1, 2.6, 2.4, "HEAD (trained)\nA: one 1x1 layer, 769 weights\nB: 1x1 reduce + dilated\nresidual 3x3 blocks\n(0.4 M to 2 M weights)", TRAIN); arrow(11.65, 12.15)
box(12.2, 1.7, 1.3, 1.2, "scores\n33 x 44", NEUT); arrow(13.55, 14.05)
box(14.0, 1.7, 0.95, 1.2, "x7 up,\nsigmoid,\nthreshold", NEUT, 7.5)
ax.text(7.5, 4.25, "Supervised pipeline: what stays frozen (blue) and what we train (orange)", ha="center", fontsize=11, color=INK)
ax.text(7.5, 0.55, "Loss = BCE + Dice against the 231 x 308 ground-truth mask.  Stage A/B: features cached once, only the head trains.  "
        "Stage C: head + LoRA trained end to end on images.", ha="center", fontsize=8.5, color=INK)
ax.text(7.5, 0.15, "Output: one probability per pixel (subject vs water); threshold chosen on validation images only.", ha="center", fontsize=8.5, color=INK)
_out = Path(__file__).resolve().parents[2] / "results" / "doc_figure_sources"; _out.mkdir(parents=True, exist_ok=True)
plt.savefig(_out / "supervised_pipeline.png", dpi=130, bbox_inches="tight"); print("ok")
