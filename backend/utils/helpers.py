"""
NeuroFusion AI — Utility Helpers
==================================
General-purpose utility functions used across the project.
"""

import os, sys, re, json, logging, base64, io
from datetime import datetime
from typing import Any
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Image utilities
# ──────────────────────────────────────────────────────────────

def numpy_to_base64(img_array: np.ndarray, fmt: str = "PNG") -> str:
    """
    Convert a numpy RGB uint8 array to a base64-encoded image string.
    Useful for embedding images in Streamlit HTML widgets.
    """
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a numpy uint8 RGB array."""
    return np.array(pil_image.convert("RGB"), dtype=np.uint8)


def resize_for_display(img_array: np.ndarray,
                       max_dim: int = 400) -> np.ndarray:
    """
    Proportionally resize an image so the larger dimension ≤ max_dim.
    Input/output: numpy uint8 RGB.
    """
    h, w = img_array.shape[:2]
    if max(h, w) <= max_dim:
        return img_array
    scale = max_dim / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = Image.fromarray(img_array).resize((new_w, new_h), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
#  Prediction utilities
# ──────────────────────────────────────────────────────────────

def probabilities_to_dict(probs: np.ndarray,
                           class_names: list) -> dict:
    """
    Map a probability vector to a {class: probability} dict.

    Parameters
    ----------
    probs       : np.ndarray  shape (num_classes,)
    class_names : list[str]

    Returns
    -------
    dict  sorted by probability descending
    """
    result = {cls: float(p) for cls, p in zip(class_names, probs)}
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def format_confidence(confidence: float) -> str:
    """Return a percentage string, e.g. 0.937 → '93.7%'."""
    return f"{confidence * 100:.1f}%"


def get_confidence_color(confidence: float) -> str:
    """Return a CSS colour based on confidence level."""
    if confidence >= 0.85:
        return "#27ae60"   # green
    elif confidence >= 0.65:
        return "#f39c12"   # orange
    else:
        return "#e74c3c"   # red


# ──────────────────────────────────────────────────────────────
#  ID / string helpers
# ──────────────────────────────────────────────────────────────

def generate_patient_id(prefix: str = "NF") -> str:
    """Generate a timestamped patient ID, e.g. 'NF-20241015-0042'."""
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{prefix}-{ts}"


def sanitise_patient_id(raw: str) -> str:
    """Strip unsafe characters from a patient ID string."""
    return re.sub(r"[^\w\-]", "", raw.strip()).upper()


def truncate_text(text: str, max_len: int = 50) -> str:
    """Truncate text to max_len characters with ellipsis."""
    return text if len(text) <= max_len else text[:max_len - 3] + "…"


# ──────────────────────────────────────────────────────────────
#  JSON helpers
# ──────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_loads(text: str, fallback: Any = None) -> Any:
    """Parse JSON string without raising; return fallback on error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback


# ──────────────────────────────────────────────────────────────
#  File utilities
# ──────────────────────────────────────────────────────────────

def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def list_files(directory: str, extensions: list = None) -> list:
    """
    Recursively list files under *directory*, optionally filtered by extension.
    """
    result = []
    if not os.path.isdir(directory):
        return result
    for root, _, files in os.walk(directory):
        for fname in files:
            if extensions is None or os.path.splitext(fname)[1].lower() in extensions:
                result.append(os.path.join(root, fname))
    return sorted(result)


def format_file_size(size_bytes: int) -> str:
    """Return a human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ──────────────────────────────────────────────────────────────
#  Streamlit state helpers
# ──────────────────────────────────────────────────────────────

def get_session(key: str, default=None):
    """Safely retrieve a Streamlit session_state value."""
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def set_session(key: str, value):
    """Safely set a Streamlit session_state value."""
    try:
        import streamlit as st
        st.session_state[key] = value
    except Exception:
        pass
