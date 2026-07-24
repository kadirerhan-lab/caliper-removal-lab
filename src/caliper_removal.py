from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CaliperRemovalConfig:
    brightness_threshold: int = 220
    saturation_max: int = 90
    min_component_area: int = 8
    max_component_area: int = 5000
    dilation_pixels: int = 2
    inpaint_radius: int = 3
    inpaint_method: str = "TELEA"
    detect_colored_overlays: bool = True


def _filter_components(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> tuple[np.ndarray, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    kept = 0

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        elongated = max(width, height) >= 3 * max(1, min(width, height))
        plausible = min_area <= area <= max_area

        if plausible or (elongated and area >= min_area):
            filtered[labels == label] = 255
            kept += 1

    return filtered, kept


def build_candidate_mask(
    image_rgb: np.ndarray,
    config: CaliperRemovalConfig,
) -> tuple[np.ndarray, dict]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Görüntü H×W×3 RGB formatında olmalıdır.")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    bright_neutral = (
        (gray >= config.brightness_threshold)
        & (hsv[:, :, 1] <= config.saturation_max)
    )
    candidate = bright_neutral.astype(np.uint8) * 255

    if config.detect_colored_overlays:
        local_bg = cv2.GaussianBlur(gray, (0, 0), 5)
        local_contrast = cv2.absdiff(gray, local_bg)
        colored = (
            (hsv[:, :, 1] >= 110)
            & (hsv[:, :, 2] >= 120)
            & (local_contrast >= 18)
        )
        candidate = cv2.bitwise_or(candidate, colored.astype(np.uint8) * 255)

    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    _, thin_bright = cv2.threshold(top_hat, 28, 255, cv2.THRESH_BINARY)
    candidate = cv2.bitwise_or(candidate, thin_bright)

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    )

    candidate, component_count = _filter_components(
        candidate,
        config.min_component_area,
        config.max_component_area,
    )

    if config.dilation_pixels > 0:
        size = 2 * config.dilation_pixels + 1
        candidate = cv2.dilate(
            candidate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=1,
        )

    mask_pixels = int(np.count_nonzero(candidate))
    total_pixels = int(candidate.size)
    return candidate, {
        "mask_pixels": mask_pixels,
        "modified_ratio": mask_pixels / max(total_pixels, 1),
        "component_count": component_count,
    }


def clean_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: CaliperRemovalConfig,
) -> np.ndarray:
    method = (
        cv2.INPAINT_NS
        if config.inpaint_method.upper() == "NAVIER_STOKES"
        else cv2.INPAINT_TELEA
    )

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cleaned_bgr = cv2.inpaint(
        image_bgr,
        mask.astype(np.uint8),
        float(config.inpaint_radius),
        method,
    )
    return cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)


def make_difference_view(
    original_rgb: np.ndarray,
    cleaned_rgb: np.ndarray,
) -> np.ndarray:
    difference = cv2.absdiff(original_rgb, cleaned_rgb)
    return cv2.convertScaleAbs(difference, alpha=4.0, beta=0)
