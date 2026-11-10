"""
NeuroFusion AI — Training: Clinical Model
==========================================
Train a dense neural network on clinical/patient features only.

Run:
    python training/train_clinical.py

Algorithm: Dense Neural Network, Categorical Encoding, Feature Scaling
"""

import os, sys, json, logging, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.clinical_preprocessing import (
    ClinicalPreprocessor, generate_synthetic_clinical_data
)
from models.model_utils import build_clinical_feature_extractor, load_class_names
from training.evaluate import evaluate_model_from_arrays, save_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_clinical_model(csv_path: str = None,
                         epochs: int = 60,
                         batch_size: int = 32):
    """
    Train the clinical-data classifier.

    Parameters
    ----------
    csv_path   : str  Path to CSV with clinical features + 'tumor_class' column.
                      If None, uses synthetic demo data (clearly labelled).
    epochs     : int
    batch_size : int
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder

    logger.info("=" * 60)
    logger.info("NeuroFusion AI — Clinical Model Training")
    logger.info("=" * 60)

    class_names = load_class_names()
    num_classes  = len(class_names)

    # ── Load data
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info("Loaded clinical CSV: %s (%d rows)", csv_path, len(df))
    else:
        logger.warning(
            "No clinical CSV supplied. Using SYNTHETIC demo data. "
            "Do NOT use these results in a real clinical setting."
        )
        df = generate_synthetic_clinical_data(n_samples=1000,
                                              class_names=class_names)

    # ── Preprocess features
    preprocessor = ClinicalPreprocessor()
    X = preprocessor.fit_transform(df)
    preprocessor.save()

    # ── Encode labels
    le = LabelEncoder()
    le.fit(class_names)
    y_int = le.transform(df["tumor_class"].values)
    y_cat = to_categorical(y_int, num_classes=num_classes)

    # ── Split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y_cat, test_size=1 - config.TRAIN_RATIO,
        random_state=config.RANDOM_SEED, stratify=y_int
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
    )
    logger.info("Split → train:%d  val:%d  test:%d", len(X_tr), len(X_val), len(X_te))

    # ── Build model
    input_dim = X.shape[1]
    model = build_clinical_feature_extractor(input_dim, num_classes)
    model.compile(
        optimizer = "adam",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()

    # ── Train
    os.makedirs(config.TRAINED_DIR, exist_ok=True)
    history = model.fit(
        X_tr, y_tr,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = [
            ModelCheckpoint(config.CLINICAL_MODEL_PATH,
                            monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=5, verbose=1),
        ],
        verbose=1,
    )

    # ── Evaluate
    metrics = evaluate_model_from_arrays(model, X_te, y_te,
                                         class_names, model_name="Clinical-only")

    # ── Save history
    hist_path = os.path.join(config.TRAINED_DIR, "clinical_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    logger.info("=" * 60)
    logger.info("Clinical model training complete.")
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  F1-Score : %.4f", metrics["f1"])
    logger.info("Model saved → %s", config.CLINICAL_MODEL_PATH)
    logger.info("=" * 60)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NeuroFusion AI clinical feature classifier"
    )
    parser.add_argument("--csv", default=None,
                        help="Path to clinical CSV (tumor_class column required).")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train_clinical_model(csv_path=args.csv, epochs=args.epochs)
