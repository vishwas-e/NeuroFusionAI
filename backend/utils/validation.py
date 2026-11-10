"""
NeuroFusion AI — Input Validation
====================================
Validates user inputs before processing to prevent errors
and provide clear, user-friendly feedback.
"""

import os, sys, re, logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Result container
# ──────────────────────────────────────────────────────────────

class ValidationResult:
    """Holds the outcome of a validation check."""

    def __init__(self, is_valid: bool = True, errors: list = None):
        self.is_valid = is_valid
        self.errors   = errors or []

    def add_error(self, message: str):
        self.is_valid = False
        self.errors.append(message)

    def __bool__(self):
        return self.is_valid

    def __repr__(self):
        status = "OK" if self.is_valid else f"FAILED: {self.errors}"
        return f"ValidationResult({status})"


# ──────────────────────────────────────────────────────────────
#  Patient information validation
# ──────────────────────────────────────────────────────────────

def validate_patient_info(patient_info: dict) -> ValidationResult:
    """
    Validate the patient information dictionary from the UI form.

    Required fields: patient_id, age, gender
    """
    result = ValidationResult()

    # Patient ID
    pid = patient_info.get("patient_id", "").strip()
    if not pid:
        result.add_error("Patient ID is required.")
    elif len(pid) < 2:
        result.add_error("Patient ID must be at least 2 characters long.")
    elif not re.match(r"^[\w\-]+$", pid):
        result.add_error(
            "Patient ID may only contain letters, digits, underscores, and hyphens."
        )

    # Age
    age = patient_info.get("age")
    if age is None:
        result.add_error("Age is required.")
    else:
        try:
            age_val = int(age)
            if not (0 < age_val < 130):
                result.add_error("Age must be between 1 and 129.")
        except (ValueError, TypeError):
            result.add_error("Age must be a valid integer.")

    # Gender
    gender = patient_info.get("gender", "").strip()
    if not gender:
        result.add_error("Gender is required.")
    elif gender not in config.GENDER_OPTIONS:
        result.add_error(f"Gender must be one of: {config.GENDER_OPTIONS}.")

    return result


# ──────────────────────────────────────────────────────────────
#  MRI image validation
# ──────────────────────────────────────────────────────────────

def validate_mri_upload(uploaded_file) -> ValidationResult:
    """
    Validate a Streamlit UploadedFile object for MRI images.
    """
    result = ValidationResult()

    if uploaded_file is None:
        result.add_error("No MRI image was uploaded. Please upload an MRI file.")
        return result

    # Check extension
    filename = getattr(uploaded_file, "name", "")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.SUPPORTED_IMAGE_FORMATS:
        result.add_error(
            f"Unsupported file format '{ext}'. "
            f"Accepted formats: {', '.join(config.SUPPORTED_IMAGE_FORMATS)}."
        )
        return result

    # Check file size (max 15 MB)
    MAX_SIZE_MB = 15
    file_size = getattr(uploaded_file, "size", None)
    if file_size is not None and file_size > MAX_SIZE_MB * 1024 * 1024:
        result.add_error(
            f"File is too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_SIZE_MB} MB."
        )
        return result

    # Try to open the image
    try:
        from PIL import Image
        import io
        content = uploaded_file.read()
        uploaded_file.seek(0)           # Reset stream for later use
        img = Image.open(io.BytesIO(content))
        img.verify()
    except Exception as exc:
        result.add_error(
            "The uploaded file could not be read as an image. "
            "Please ensure it is a valid, uncorrupted MRI image."
        )
        logger.error("Image validation error: %s", exc)

    return result


# ──────────────────────────────────────────────────────────────
#  Model readiness validation
# ──────────────────────────────────────────────────────────────

def validate_model_ready(model_type: str = "mri") -> ValidationResult:
    """
    Check that the required trained model files exist.

    Parameters
    ----------
    model_type : str  'mri', 'clinical', or 'fusion'
    """
    result = ValidationResult()
    paths = {
        "mri"     : config.MRI_MODEL_PATH,
        "clinical": config.CLINICAL_MODEL_PATH,
        "fusion"  : config.FUSION_MODEL_PATH,
    }
    path = paths.get(model_type, config.MRI_MODEL_PATH)
    if not os.path.exists(path):
        result.add_error(
            f"Trained model not found: {os.path.basename(path)}. "
            f"Please run the corresponding training script first."
        )
    return result


# ──────────────────────────────────────────────────────────────
#  Image dimension validation
# ──────────────────────────────────────────────────────────────

def validate_image_dimensions(img_array: np.ndarray,
                               expected_shape: tuple = None) -> ValidationResult:
    """
    Validate that a preprocessed image array has the expected shape.

    Parameters
    ----------
    img_array      : np.ndarray
    expected_shape : tuple  Default: (*config.IMAGE_SIZE, 3)
    """
    result = ValidationResult()
    if expected_shape is None:
        expected_shape = (*config.IMAGE_SIZE, 3)

    if img_array is None:
        result.add_error("Image array is None.")
        return result

    if img_array.shape != expected_shape:
        result.add_error(
            f"Unexpected image shape {img_array.shape}. "
            f"Expected {expected_shape}."
        )
    return result


# ──────────────────────────────────────────────────────────────
#  Clinical feature validation
# ──────────────────────────────────────────────────────────────

def validate_clinical_features(patient_info: dict) -> ValidationResult:
    """
    Warn if optional clinical feature fields are missing.
    These fields are not strictly required but improve prediction quality.
    """
    result = ValidationResult()
    optional_fields = config.CLINICAL_CATEGORICAL_FEATURES

    missing = [f for f in optional_fields if not patient_info.get(f)]
    if missing:
        # Non-blocking: warn only
        logger.info("Optional clinical fields not provided: %s", missing)
    return result
