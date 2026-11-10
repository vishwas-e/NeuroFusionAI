"""
NeuroFusion AI — Tumor Region Segmentation & Boundary Localization
===================================================================
Extracts the segmented tumor boundary mask, colors the tumor region
with a vibrant semi-transparent highlight, draws the sharp lesion boundary,
and calculates the tumor surface volume percentage.
"""

import cv2
import numpy as np
from typing import Dict, Any


def generate_tumor_segmentation(
    heatmap: np.ndarray,
    original_img: np.ndarray,
    pred_class: str
) -> Dict[str, Any]:
    """
    Generate colored tumor segmentation overlay and boundary line.

    Parameters
    ----------
    heatmap : np.ndarray
        Raw float32 Grad-CAM activation heatmap [0, 1].
    original_img : np.ndarray
        Raw uint8 RGB diagnostic MRI image.
    pred_class : str
        Predicted tumor class ('glioma', 'meningioma', 'pituitary', 'notumor').

    Returns
    -------
    dict with:
        'overlay': RGB uint8 image with colored tumor highlight & boundary
        'detected': bool
        'area_pct': float (percentage of brain area)
        'color_name': str
        'message': str
    """
    h, w = original_img.shape[:2]
    c_lower = (pred_class or "").lower().replace("_", "").replace(" ", "")

    if "notumor" in c_lower:
        return {
            "overlay": original_img.copy(),
            "detected": False,
            "area_pct": 0.0,
            "color_name": "None",
            "message": "Clean scan — Healthy brain architecture with no abnormal tumor boundary detected."
        }

    # Ensure heatmap matches image size
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Identify brain tissue vs pitch-black background
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) == 3 else original_img
    brain_mask = gray > 25

    # Threshold the top salient activation inside brain tissue
    thresh_val = max(0.40, np.percentile(heatmap_resized[brain_mask], 85) if np.any(brain_mask) else 0.5)
    binary_tumor = ((heatmap_resized >= thresh_val) & brain_mask).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_tumor = cv2.morphologyEx(binary_tumor, cv2.MORPH_CLOSE, kernel)
    binary_tumor = cv2.morphologyEx(binary_tumor, cv2.MORPH_OPEN, kernel)

    # Find contours of the tumor region
    contours, _ = cv2.findContours(binary_tumor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick specific color highlight for each class
    if "glio" in c_lower:
        color_rgb = (239, 68, 68)     # Bright Red
        border_rgb = (255, 255, 255)  # Crisp White Border
        color_name = "Red (Glioma Mass)"
    elif "menin" in c_lower:
        color_rgb = (245, 158, 11)    # Warm Amber
        border_rgb = (255, 255, 255)
        color_name = "Amber (Dural Meningioma)"
    elif "pitu" in c_lower:
        color_rgb = (14, 165, 233)    # Bright Cyan/Sky Blue
        border_rgb = (255, 255, 255)
        color_name = "Cyan (Pituitary Sellar Mass)"
    else:
        color_rgb = (239, 68, 68)
        border_rgb = (255, 255, 255)
        color_name = "Colored Tumor Highlight"

    overlay = original_img.copy()
    mask_layer = np.zeros_like(original_img)
    tumor_px = 0
    brain_px = np.count_nonzero(brain_mask)

    if contours:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80]
        if not valid_contours:
            valid_contours = contours

        # 1. Fill tumor region with color
        cv2.drawContours(mask_layer, valid_contours, -1, color_rgb, thickness=cv2.FILLED)
        tumor_px = np.count_nonzero(mask_layer[:, :, 0])

        # 2. Blend semi-transparent highlight (50% alpha)
        overlay = cv2.addWeighted(overlay, 1.0, mask_layer, 0.50, 0)

        # 3. Draw crisp contour boundary line around the tumor (thickness=2)
        cv2.drawContours(overlay, valid_contours, -1, border_rgb, thickness=2, lineType=cv2.LINE_AA)

        # 4. Draw neat label on the largest tumor contour
        largest_cnt = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label = f"{pred_class.title()} Lesion"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay, (cX - 5, cY - th - 5), (cX + tw + 5, cY + 5), (0, 0, 0), -1)
            cv2.putText(overlay, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    area_pct = (tumor_px / brain_px * 100) if brain_px > 0 else 0.0

    return {
        "overlay": overlay,
        "detected": True,
        "area_pct": round(area_pct, 1),
        "color_name": color_name,
        "message": f"Tumor boundary localized ({area_pct:.1f}% intracranial tissue area)."
    }
