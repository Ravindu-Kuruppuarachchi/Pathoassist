"""ER/PR overlay: tumour nuclei coloured by intensity bin, non-tumour nuclei grey."""
import cv2
import numpy as np

from core.ihc.pipeline import TUMOUR, FieldAnalysis

# RGB
INTENSITY_COLORS = [(40, 110, 255), (255, 230, 0), (255, 140, 0), (230, 0, 0)]  # neg, weak, mod, strong
NON_TUMOUR = (170, 170, 170)


def draw_erpr_overlay(frame_rgb: np.ndarray, a: FieldAnalysis, thickness: int = 2, corrected=()) -> np.ndarray:
    out = frame_rgb.copy()
    for i, c in enumerate(a.coords):
        pts = np.round(c[::-1].T).astype(np.int32)
        if a.cell_type[i] == TUMOUR:
            col, th = INTENSITY_COLORS[int(a.intensity[i])], thickness
        else:
            col, th = NON_TUMOUR, 1
        cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
        if i in corrected:
            y, x = a.points[i]
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1)
    return out
