import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

def _largest_contour_mask(gray: np.ndarray, threshold: int) -> tuple[np.ndarray, np.ndarray | None]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(gray), None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask, contour

def extract_features(roi_rgb: np.ndarray, segmentation_threshold: int = 90) -> tuple[dict, np.ndarray]:
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    mask, contour = _largest_contour_mask(gray, segmentation_threshold)

    values = gray[mask > 0]
    if values.size == 0:
        values = gray.ravel()

    shape = {
        "width_px": roi_rgb.shape[1],
        "height_px": roi_rgb.shape[0],
        "aspect_ratio": roi_rgb.shape[1] / max(roi_rgb.shape[0], 1),
        "taller_than_wide": roi_rgb.shape[0] > roi_rgb.shape[1],
        "area_px": 0.0,
        "perimeter_px": 0.0,
        "circularity": 0.0,
        "solidity": 0.0,
    }

    if contour is not None:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        x, y, w, h = cv2.boundingRect(contour)
        shape.update({
            "width_px": int(w),
            "height_px": int(h),
            "aspect_ratio": w / max(h, 1),
            "taller_than_wide": h > w,
            "area_px": area,
            "perimeter_px": perimeter,
            "circularity": (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0,
            "solidity": area / hull_area if hull_area > 0 else 0.0,
        })

    quantized = (gray // 16).astype(np.uint8)
    glcm = graycomatrix(quantized, [1], [0], levels=16, symmetric=True, normed=True)
    edges = cv2.Canny(gray, 50, 150)

    texture = {
        "mean_intensity": float(values.mean()),
        "std_intensity": float(values.std()),
        "entropy": float(shannon_entropy(values)),
        "contrast": float(graycoprops(glcm, "contrast")[0, 0]),
        "homogeneity": float(graycoprops(glcm, "homogeneity")[0, 0]),
        "energy": float(graycoprops(glcm, "energy")[0, 0]),
        "correlation": float(graycoprops(glcm, "correlation")[0, 0]),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float(np.count_nonzero(edges) / edges.size),
    }
    return {**shape, **texture}, mask
