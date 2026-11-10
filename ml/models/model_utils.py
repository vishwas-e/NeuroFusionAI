"""
NeuroFusion AI — Model Utilities
=================================
Builds the MRI (ResNet50), clinical, and fusion models.
Provides load / save helpers used throughout the project.

Algorithms:
  - CNN + Transfer Learning (ResNet50)
  - Clinical Dense Feature Extractor
  - Concatenation-based Feature Fusion
  - Dense Classifier + Softmax
"""

import os, sys, json, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  MRI Branch  (ResNet50 Transfer Learning)
# ──────────────────────────────────────────────────────────────

def build_mri_feature_extractor(num_classes: int,
                                 trainable_base: bool = False) -> Model:
    """
    Build the MRI-only classification model using ResNet50 as the
    pretrained backbone (Transfer Learning).

    Architecture
    ────────────
    Input (224×224×3)
    → ResNet50 (pretrained on ImageNet, base frozen by default)
    → GlobalAveragePooling2D
    → Dense(512, ReLU) ← MRI feature vector
    → Dropout(0.4)
    → Dense(num_classes, Softmax)

    Parameters
    ----------
    num_classes     : int   Number of tumor categories.
    trainable_base  : bool  If True, allow all ResNet50 layers to
                            update (for fine-tuning phase).

    Returns
    -------
    keras.Model
    """
    base_model = ResNet50(
        weights    = "imagenet",
        include_top= False,
        input_shape= (*config.IMAGE_SIZE, config.IMAGE_CHANNELS),
    )
    base_model.trainable = trainable_base

    inputs  = keras.Input(shape=(*config.IMAGE_SIZE, config.IMAGE_CHANNELS),
                          name="mri_input")
    x       = base_model(inputs, training=trainable_base)
    x       = layers.GlobalAveragePooling2D(name="gap")(x)
    x       = layers.Dense(512, activation="relu", name="mri_features")(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="mri_output")(x)

    model = Model(inputs, outputs, name="MRI_ResNet50_Classifier")
    logger.info("MRI feature extractor built  (base trainable=%s)", trainable_base)
    return model


def build_mri_feature_model() -> Model:
    """
    Return a model that outputs the 512-D MRI feature vector
    (before the final softmax head).  Used during fusion.
    """
    # Load or build full MRI model, then strip the output head
    if os.path.exists(config.MRI_MODEL_PATH):
        full_model = keras.models.load_model(config.MRI_MODEL_PATH, compile=False)
    else:
        raise FileNotFoundError(
            f"Trained MRI model not found at {config.MRI_MODEL_PATH}. "
            "Run training/train_mri.py first."
        )

    feature_model = Model(
        inputs  = full_model.input,
        outputs = full_model.get_layer("mri_features").output,
        name    = "MRI_Feature_Extractor",
    )
    feature_model.trainable = False
    return feature_model


# ──────────────────────────────────────────────────────────────
#  Clinical Branch  (Dense Feature Extractor)
# ──────────────────────────────────────────────────────────────

def build_clinical_feature_extractor(input_dim: int,
                                      num_classes: int) -> Model:
    """
    Build a small dense network to extract features from the
    clinical/patient feature vector.

    Architecture
    ────────────
    Input (input_dim,)
    → Dense(128, ReLU)
    → BatchNormalization
    → Dropout(0.3)
    → Dense(64, ReLU)  ← Clinical feature vector
    → Dense(num_classes, Softmax)

    Parameters
    ----------
    input_dim   : int   Number of preprocessed clinical features.
    num_classes : int   Number of tumor categories.
    """
    inputs = keras.Input(shape=(input_dim,), name="clinical_input")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", name="clinical_features")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="clinical_output")(x)

    model = Model(inputs, outputs, name="Clinical_Feature_Extractor")
    logger.info("Clinical feature extractor built (input_dim=%d)", input_dim)
    return model


def build_clinical_feature_model(input_dim: int) -> Model:
    """
    Return a model that outputs the 64-D clinical feature vector.
    Used during fusion.
    """
    if os.path.exists(config.CLINICAL_MODEL_PATH):
        full_model = keras.models.load_model(config.CLINICAL_MODEL_PATH, compile=False)
    else:
        raise FileNotFoundError(
            f"Trained clinical model not found at {config.CLINICAL_MODEL_PATH}. "
            "Run training/train_clinical.py first."
        )

    feature_model = Model(
        inputs  = full_model.input,
        outputs = full_model.get_layer("clinical_features").output,
        name    = "Clinical_Feature_Extractor",
    )
    feature_model.trainable = False
    return feature_model


# ──────────────────────────────────────────────────────────────
#  Fusion Model  (MRI + Clinical → Final Classifier)
# ──────────────────────────────────────────────────────────────

def build_fusion_model(mri_feature_dim: int,
                       clinical_feature_dim: int,
                       num_classes: int) -> Model:
    """
    Build the multimodal fusion model.

    Architecture
    ────────────
    MRI feature vector (512,)  ─┐
                                 ├─ Concatenation (576,)
    Clinical feature vector (64,)┘
    → Dense(256, ReLU)
    → BatchNormalization
    → Dropout(0.4)
    → Dense(128, ReLU)
    → Dropout(0.3)
    → Dense(num_classes, Softmax)  ← Final tumor classification

    Parameters
    ----------
    mri_feature_dim      : int
    clinical_feature_dim : int
    num_classes          : int
    """
    mri_input      = keras.Input(shape=(mri_feature_dim,),      name="mri_feat_input")
    clinical_input = keras.Input(shape=(clinical_feature_dim,), name="clinical_feat_input")

    # ── Concatenation-based Feature Fusion
    fused = layers.Concatenate(name="feature_fusion")([mri_input, clinical_input])

    x = layers.Dense(256, activation="relu")(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="fusion_output")(x)

    model = Model(
        inputs  = [mri_input, clinical_input],
        outputs = outputs,
        name    = "NeuroFusion_Classifier",
    )
    logger.info(
        "Fusion model built (mri=%d + clinical=%d → %d classes)",
        mri_feature_dim, clinical_feature_dim, num_classes,
    )
    return model


# ──────────────────────────────────────────────────────────────
#  Load / Save utilities
# ──────────────────────────────────────────────────────────────

def load_model_safe(model_path: str):
    """Load a Keras model; return None if the file does not exist."""
    if not os.path.exists(model_path):
        logger.warning("Model not found: %s", model_path)
        return None
    try:
        model = keras.models.load_model(model_path, compile=False)
        logger.info("Loaded model from %s", model_path)
        return model
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_path, exc)
        return None


def save_class_names(class_names: list,
                     path: str = config.CLASS_NAMES_PATH):
    """Persist the ordered list of class names as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f)
    logger.info("Class names saved: %s", class_names)


def load_class_names(path: str = config.CLASS_NAMES_PATH) -> list:
    """Load class names from JSON; fall back to config defaults."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    logger.warning("class_names.json not found; using defaults.")
    return config.DEFAULT_CLASSES


def get_model_summary(model) -> str:
    """Return model summary as a string."""
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)
