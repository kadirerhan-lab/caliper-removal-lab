from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class CaliperConfig:
    brightness_threshold: int = 220
    saturation_max: int = 90
    min_area: int = 8
    max_area: int = 5000
    dilation: int = 2
    inpaint_radius: int = 3
    method: str = "TELEA"

def detect_caliper_mask(image_rgb: np.ndarray, cfg: CaliperConfig) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    bright = ((gray >= cfg.brightness_threshold) &
              (hsv[:, :, 1] <= cfg.saturation_max)).astype(np.uint8) * 255

    top_hat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    _, thin = cv2.threshold(top_hat, 28, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(bright, thin)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        elongated = max(w, h) >= 3 * max(1, min(w, h))
        if cfg.min_area <= area <= cfg.max_area or (elongated and area >= cfg.min_area):
            filtered[labels == i] = 255

    if cfg.dilation > 0:
        k = 2 * cfg.dilation + 1
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1
        )
    return filtered

def remove_calipers(image_rgb: np.ndarray, mask: np.ndarray, cfg: CaliperConfig) -> np.ndarray:
    method = cv2.INPAINT_NS if cfg.method == "NAVIER_STOKES" else cv2.INPAINT_TELEA
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(bgr, mask, float(cfg.inpaint_radius), method)
    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
