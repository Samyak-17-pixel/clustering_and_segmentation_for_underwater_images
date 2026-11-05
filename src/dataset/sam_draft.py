"""Draft ground-truth masks for the tank fish frames with SAM (sam-vit-base) from hint boxes that were placed by looking at each frame.

The hints are NOT derived from any of our models, so the drafts do not favour them. SAM only turns a box into an outline; a human reviews every draft
(masks_draft_overlays/) before it becomes ground truth (masks/). Boxes are in a 960 x 540 view of the 1920 x 1080 frame: (x0, y0, x1, y1).

  python sam_draft.py            # writes data/dhwani_dataset/masks_draft/*.png, masks_draft_overlays/*.png, masks_draft/hints.json
"""
import json
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "dhwani_dataset"
HINTS = {   # frame -> list of boxes (one per fish; partial fish at the border included); [] = no subject in the frame
    "GX010214_t0052.99s": [(0, 160, 250, 370), (480, 245, 695, 345)],
    "GX010214_t0054.49s": [(0, 20, 108, 150), (298, 195, 578, 345), (738, 180, 960, 335)],
    "GX010214_t0055.49s": [(0, 0, 365, 170), (0, 130, 280, 310), (480, 175, 795, 320), (910, 160, 960, 265)],
    "GX010214_t0056.99s": [(330, 0, 790, 100), (335, 0, 405, 60), (290, 130, 662, 322), (762, 165, 960, 352)],
    "GX010214_t0057.99s": [(0, 0, 230, 230), (555, 0, 960, 285)],
    "GX010214_t0058.99s": [(0, 0, 468, 198)],
    "GX010214_t0060.49s": [(640, 0, 765, 92), (798, 0, 960, 168)],
    "GX010214_t0061.49s": [],
    "GX010214_t0102.47s": [(0, 0, 178, 140), (205, 0, 548, 142), (520, 0, 672, 55), (800, 0, 960, 100)],
    "GX010214_t0103.97s": [(0, 0, 318, 135), (428, 30, 672, 158), (765, 18, 960, 172)],
    "GX010214_t0104.97s": [(0, 0, 178, 120), (322, 58, 562, 197), (648, 122, 945, 245)],
    "GX010214_t0106.47s": [(412, 0, 592, 82), (0, 28, 232, 197), (212, 98, 455, 203), (598, 108, 852, 232)],
    "GX010214_t0107.97s": [(0, 78, 58, 138), (118, 108, 388, 268), (408, 138, 602, 252), (558, 18, 818, 208), (818, 178, 960, 282)],
    "GX010214_t0108.98s": [(0, 98, 268, 268), (272, 192, 492, 318), (503, 190, 702, 312), (708, 100, 960, 322)],
    "GX010214_t0109.98s": [(112, 178, 392, 402), (378, 252, 592, 398), (572, 266, 862, 412), (828, 232, 960, 472)],
    "GX010214_t0110.98s": [(0, 222, 102, 332), (218, 308, 502, 540), (482, 328, 698, 538), (702, 326, 960, 540)],
    "GX010217_t0099.00s": [(0, 415, 288, 540)],
    "GX010217_t0100.00s": [(468, 412, 722, 538)],
    "GX010217_t0101.50s": [(288, 495, 432, 540), (468, 462, 578, 512), (612, 480, 802, 540)],
    "GX010217_t0103.47s": [(672, 448, 892, 538), (878, 442, 960, 492)],
}


def main():
    dev = "cuda"
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base"); sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    (D / "masks_draft").mkdir(exist_ok=True); (D / "masks_draft_overlays").mkdir(exist_ok=True)
    for stem, boxes in HINTS.items():
        img = Image.open(D / "images" / f"{stem}.jpg").convert("RGB"); W, H = img.size
        mask = np.zeros((H, W), bool)
        if boxes:
            b = [[2 * v for v in bx] for bx in boxes]
            inp = proc(img, input_boxes=[b], return_tensors="pt").to(dev)
            with torch.no_grad():
                o = sam(pixel_values=inp["pixel_values"], input_boxes=inp["input_boxes"], multimask_output=True)
            ms = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu())[0]   # [n,3,H,W]
            sc = o.iou_scores[0].cpu()                                                                                                      # [n,3]
            for i in range(len(b)):
                mask |= ms[i, int(sc[i].argmax())].numpy().astype(bool)
        Image.fromarray(mask.astype(np.uint8) * 255).save(D / "masks_draft" / f"{stem}.png")
        # review picture: frame with hint boxes | draft outline
        a = cv2.resize(np.asarray(img), (960, 540)); bimg = a.copy()
        for bx in boxes: cv2.rectangle(a, bx[:2], bx[2:], (255, 255, 0), 2)
        m = cv2.resize(mask.astype(np.uint8), (960, 540), interpolation=cv2.INTER_NEAREST).astype(bool)
        bimg[m] = (0.5 * bimg[m] + 0.5 * np.array([230, 159, 0])).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(bimg, cs, -1, (255, 255, 255), 2)
        cv2.putText(a, f"{stem}  hints (yellow)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bimg, f"draft mask {m.mean():.1%} of frame", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(D / "masks_draft_overlays" / f"{stem}.png"), cv2.cvtColor(np.hstack([a, bimg]), cv2.COLOR_RGB2BGR))
        print(stem, len(boxes), "boxes", f"{mask.mean():.3f}", flush=True)
    (D / "masks_draft" / "hints.json").write_text(json.dumps(HINTS, indent=1))


if __name__ == "__main__":
    main()
