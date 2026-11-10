"""
NeuroFusion AI - Central Configuration
======================================
All project-wide settings and constants are defined here.
Modify this file to adapt the system to a different dataset.
"""

import os, sys

# ── Windows DLL fix: load TF's bundled VC++ 14.50 runtime before System32's
#    older version. Fixes "DLL initialization routine failed" on Windows.
if sys.platform == "win32":
    _tf_py = os.path.join(os.path.dirname(sys.executable),
                          "Lib", "site-packages", "tensorflow", "python")
    if os.path.isdir(_tf_py):
        os.add_dll_directory(_tf_py)

# ── Ensure KERAS_HOME points to local directory avoiding Windows permission issues with C:\Users\<user>\.keras
_local_keras = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras")
os.makedirs(_local_keras, exist_ok=True)
os.environ.setdefault("KERAS_HOME", _local_keras)

# ─────────────────────────────────────────────
#  Paths & Segregated Architecture
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR    = os.path.join(BASE_DIR, "frontend")
BACKEND_DIR     = os.path.join(BASE_DIR, "backend")
ML_DIR          = os.path.join(BASE_DIR, "ml")

# Ensure root, backend, and ml are on Python path for clean cross-module imports
for _p in [BASE_DIR, BACKEND_DIR, ML_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
CLINICAL_DIR    = os.path.join(DATA_DIR, "clinical")
TEMPLATES_DIR   = os.path.join(FRONTEND_DIR, "templates")

# Machine Learning & Models
MODELS_DIR      = os.path.join(ML_DIR, "models")
TRAINED_DIR     = os.path.join(MODELS_DIR, "trained")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")
DATABASE_PATH   = os.path.join(BACKEND_DIR, "database", "neurofusion.db")

# Saved model artifacts
MRI_MODEL_PATH      = os.path.join(TRAINED_DIR, "mri_model.h5")
CLINICAL_MODEL_PATH = os.path.join(TRAINED_DIR, "clinical_model.h5")
FUSION_MODEL_PATH   = os.path.join(TRAINED_DIR, "fusion_model.h5")
LABEL_ENCODER_PATH  = os.path.join(TRAINED_DIR, "label_encoder.pkl")
SCALER_PATH         = os.path.join(TRAINED_DIR, "scaler.pkl")
CLASS_NAMES_PATH    = os.path.join(TRAINED_DIR, "class_names.json")
METRICS_PATH        = os.path.join(TRAINED_DIR, "metrics.json")

# ─────────────────────────────────────────────
#  Image / Model Settings
# ─────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)          # ResNet50 input size
IMAGE_CHANNELS  = 3
BATCH_SIZE      = 32
EPOCHS          = 30
LEARNING_RATE   = 1e-4
FINE_TUNE_LR    = 1e-5

# ResNet50 feature-extraction layer
GRADCAM_LAYER   = "conv5_block3_out"  # Last conv block in ResNet50

# ─────────────────────────────────────────────
#  Train / Validation / Test Split
# ─────────────────────────────────────────────
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
#  Tumor Classes  (overridden at runtime from class_names.json)
# ─────────────────────────────────────────────
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ─────────────────────────────────────────────
#  Clinical Feature Configuration
#  Change this list to match the actual dataset columns used.
# ─────────────────────────────────────────────
CLINICAL_NUMERICAL_FEATURES = ["age"]

CLINICAL_CATEGORICAL_FEATURES = [
    "gender",
    "headache",
    "vision_problems",
    "seizures",
    "nausea",
    "memory_loss",
]

CLINICAL_ALL_FEATURES = CLINICAL_NUMERICAL_FEATURES + CLINICAL_CATEGORICAL_FEATURES

GENDER_OPTIONS  = ["Male", "Female", "Other"]
BINARY_OPTIONS  = ["No", "Yes"]

# ─────────────────────────────────────────────
#  Feature Fusion Dimensions
# ─────────────────────────────────────────────
MRI_FEATURE_DIM      = 512   # Dense output from MRI branch
CLINICAL_FEATURE_DIM = 64    # Dense output from clinical branch
FUSED_FEATURE_DIM    = MRI_FEATURE_DIM + CLINICAL_FEATURE_DIM

# ─────────────────────────────────────────────
#  Augmentation Settings
# ─────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range"        : 15,
    "width_shift_range"     : 0.10,
    "height_shift_range"    : 0.10,
    "horizontal_flip"       : True,
    "zoom_range"            : 0.10,
    "brightness_range"      : [0.85, 1.15],
}

# ─────────────────────────────────────────────
#  Application Settings
# ─────────────────────────────────────────────
APP_TITLE       = "NeuroFusion AI"
APP_SUBTITLE    = "Clinical Decision Support System — Multimodal Brain Tumor Classification & Explainable AI"
APP_VERSION     = "1.0.0"
DISCLAIMER      = (
    "🏥 Clinical Decision Support System (CDSS) — This system assists medical specialists "
    "by extracting radiological and clinical indicators. Diagnostic predictions serve as decision support; "
    "final clinical diagnosis rests with the attending medical professional."
)

# Clinician access credentials
DEMO_USERNAME   = "doctor"
DEMO_PASSWORD   = "neurofusion2024"

# ─────────────────────────────────────────────
#  Supported Image Formats
# ─────────────────────────────────────────────
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
