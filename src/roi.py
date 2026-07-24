from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

def clamp_roi(roi: ROI, image_shape: tuple[int, ...]) -> ROI:
    h, w = image_shape[:2]
    x = max(0, min(roi.x, w - 1))
    y = max(0, min(roi.y, h - 1))
    width = max(1, min(roi.width, w - x))
    height = max(1, min(roi.height, h - y))
    return ROI(x, y, width, height)

def crop_roi(image_rgb: np.ndarray, roi: ROI) -> np.ndarray:
    r = clamp_roi(roi, image_rgb.shape)
    return image_rgb[r.y:r.y+r.height, r.x:r.x+r.width].copy()

def draw_roi(image_rgb: np.ndarray, roi: ROI) -> np.ndarray:
    r = clamp_roi(roi, image_rgb.shape)
    out = image_rgb.copy()
    cv2.rectangle(out, (r.x, r.y), (r.x+r.width, r.y+r.height), (255, 255, 255), 2)
    return out
