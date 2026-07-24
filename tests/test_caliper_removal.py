import cv2
import numpy as np

from src.caliper_removal import (
    CaliperRemovalConfig,
    build_candidate_mask,
    clean_image,
)


def test_synthetic_caliper_is_detected():
    image = np.full((160, 200, 3), 70, dtype=np.uint8)
    cv2.line(image, (90, 60), (110, 60), (255, 255, 255), 2)
    cv2.line(image, (100, 50), (100, 70), (255, 255, 255), 2)

    mask, diagnostics = build_candidate_mask(
        image,
        CaliperRemovalConfig(min_component_area=3),
    )

    assert diagnostics["mask_pixels"] > 0
    assert mask[60, 100] == 255


def test_cleaned_image_preserves_shape():
    image = np.full((100, 120, 3), 80, dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[40:45, 50:55] = 255

    cleaned = clean_image(image, mask, CaliperRemovalConfig())
    assert cleaned.shape == image.shape
