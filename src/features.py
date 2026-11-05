
"""Handcrafted descriptors for exploratory analysis of dermoscopic images."""
from pathlib import Path
import cv2
import numpy as np
from scipy.stats import entropy


def _safe_stats(values, prefix):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }


def _heuristic_lesion_mask(image_rgb):
    """
    Estimate a foreground region from colour distance to the image border.

    This is deliberately labelled a heuristic. It is not a clinically validated
    segmentation method and can fail on vignetting, dark frames, hair and shadows.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    border = np.concatenate([
        lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]
    ], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Restrict tiny isolated noise and select the largest connected component.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    center = np.array([w / 2, h / 2])
    candidates = []
    for idx in range(1, count):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        cx, cy = centroids[idx]
        centrality = np.linalg.norm((np.array([cx, cy]) - center) / np.array([w, h]))
        if area >= 0.005 * h * w:
            candidates.append((area / (1 + 3 * centrality), idx))
    if not candidates:
        return np.zeros((h, w), dtype=np.uint8)

    chosen = max(candidates)[1]
    return (labels == chosen).astype(np.uint8)


def extract_features(image_path):
    """Extract image-level descriptors from one RGB image."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    features = {}
    for name, image in [("r", rgb[:, :, 0]), ("g", rgb[:, :, 1]), ("b", rgb[:, :, 2]),
                        ("h", hsv[:, :, 0]), ("s", hsv[:, :, 1]), ("v", hsv[:, :, 2]),
                        ("lab_l", lab[:, :, 0]), ("lab_a", lab[:, :, 1]), ("lab_b", lab[:, :, 2])]:
        features.update(_safe_stats(image, name))

    gray_f = gray.astype(np.float32) / 255.0
    features["darkness_mean"] = float(1.0 - gray_f.mean())
    features["darkness"] = features["darkness_mean"]  # retained name from the original project
    features["color_variance"] = float(np.mean([np.var(rgb[:, :, i].astype(np.float32)) for i in range(3)]))
    features["darkness_q10"] = float(1.0 - np.quantile(gray_f, 0.10))
    features["darkness_q25"] = float(1.0 - np.quantile(gray_f, 0.25))
    features["intensity_entropy"] = float(entropy(np.histogram(gray, bins=32, range=(0, 256))[0] + 1))

    # Local texture and edge descriptors.
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    features["edge_density"] = float(np.mean(magnitude > np.quantile(magnitude, 0.85)))
    features["gradient_mean"] = float(magnitude.mean())
    features["gray_std"] = float(gray_f.std())

    # Heuristic mask-derived shape descriptors.
    mask = _heuristic_lesion_mask(rgb)
    area = int(mask.sum())
    features["mask_area_fraction"] = float(area / mask.size)
    if area > 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(max((cv2.arcLength(c, True) for c in contours), default=0.0))
        features["mask_perimeter"] = perimeter
        features["mask_compactness"] = float(4 * np.pi * area / (perimeter ** 2 + 1e-8))
        features["mask_irregularity"] = float(perimeter ** 2 / (4 * np.pi * area + 1e-8))
        features["irregularity"] = features["mask_irregularity"]  # heuristic analogue of the original descriptor
        x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        features["elongation"] = float(max(bw, bh) / (min(bw, bh) + 1e-8))
        for channel, name in [(rgb[:, :, 0], "lesion_r"), (rgb[:, :, 1], "lesion_g"),
                              (rgb[:, :, 2], "lesion_b"), (gray_f, "lesion_gray")]:
            features.update(_safe_stats(channel[mask > 0], name))
    else:
        features.update({
            "mask_perimeter": 0.0, "mask_compactness": 0.0, "mask_irregularity": 0.0,
            "irregularity": 0.0, "elongation": 0.0,
            "lesion_r_mean": np.nan, "lesion_r_std": np.nan,
            "lesion_g_mean": np.nan, "lesion_g_std": np.nan,
            "lesion_b_mean": np.nan, "lesion_b_std": np.nan,
            "lesion_gray_mean": np.nan, "lesion_gray_std": np.nan,
        })
    return features
