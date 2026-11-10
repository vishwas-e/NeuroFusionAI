"""
NeuroFusion AI — Training: Fusion Model
========================================
Trains the multimodal Feature Fusion classifier using frozen
MRI and Clinical feature extractors.

Run:
    python training/train_fusion.py

Algorithm: Feature Fusion (Concatenation), Dense Classifier, Softmax
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
from preprocessing.image_preprocessing import (
    split_dataset, load_and_preprocess_image
)
from models.model_utils import (
    build_fusion_model, build_mri_feature_model, build_clinical_feature_model,
    load_class_names
)
from training.evaluate import evaluate_model_from_arrays, save_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Feature extraction helpers
# ──────────────────────────────────────────────────────────────

def extract_mri_features_from_dir(split_dir: str,
                                   mri_feature_model,
                                   class_names: list,
                                   max_samples_per_class: int = 150) -> tuple:
    """
    Walk train/val/test splits and extract MRI feature vectors
    using the trained MRI feature extractor.

    Returns (X_features, y_int, image_paths)
    """
    X, y, paths = [], [], []
    for split in ("train", "val", "test"):
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = os.path.join(split_dir, split, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            count = 0
            for fname in os.listdir(cls_dir):
                if max_samples_per_class and count >= max_samples_per_class:
                    break
                fpath = os.path.join(cls_dir, fname)
                try:
                    img = load_and_preprocess_image(fpath)
                    X.append(img)
                    y.append(cls_idx)
                    paths.append((split, fpath))
                    count += 1
                except Exception as exc:
                    logger.warning("Skipping %s: %s", fpath, exc)

    X_arr = np.array(X, dtype=np.float32)
    # Extract features in batches
    features = mri_feature_model.predict(X_arr, batch_size=32, verbose=1)
    return features, np.array(y), paths


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def train_fusion_model(data_dir   : str = config.RAW_DIR,
                       csv_path   : str = None,
                       epochs     : int = 40,
                       batch_size : int = 32,
                       max_samples_per_class: int = 150):
    """
    Train the Feature Fusion classifier.

    Prerequisites:
      - MRI model trained    (config.MRI_MODEL_PATH must exist)
      - Clinical model trained (config.CLINICAL_MODEL_PATH must exist)
    """
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    )
    from sklearn.preprocessing import LabelEncoder

    logger.info("=" * 60)
    logger.info("NeuroFusion AI — Fusion Model Training")
    logger.info("=" * 60)

    class_names = load_class_names()
    num_classes  = len(class_names)

    # ── Validate prerequisites
    for req_path, name in [
        (config.MRI_MODEL_PATH,      "MRI model"),
        (config.CLINICAL_MODEL_PATH, "Clinical model"),
    ]:
        if not os.path.exists(req_path):
            raise FileNotFoundError(
                f"{name} not found at {req_path}. "
                "Train it first with training/train_mri.py or train_clinical.py."
            )

    # ── Load feature extractors
    logger.info("Loading MRI feature extractor …")
    mri_feat_model = build_mri_feature_model()
    mri_feat_dim   = mri_feat_model.output_shape[-1]

    logger.info("Loading Clinical feature extractor …")
    clin_preprocessor = ClinicalPreprocessor.load()
    clin_feat_model   = build_clinical_feature_model(clin_preprocessor.feature_dim)
    clin_feat_dim     = clin_feat_model.output_shape[-1]

    # ── Prepare MRI splits and extract features
    split_dir = os.path.join(config.PROCESSED_DIR, "splits")
    if not os.path.exists(split_dir):
        split_dataset(data_dir, split_dir)

    cache_path = os.path.join(config.PROCESSED_DIR, "mri_features_cache.npz")
    if os.path.exists(cache_path):
        logger.info("Loading cached MRI features from %s …", cache_path)
        data_cache = np.load(cache_path)
        mri_feats, y_mri = data_cache["feats"], data_cache["y"]
    else:
        logger.info("Extracting MRI features …")
        mri_feats, y_mri, img_paths = extract_mri_features_from_dir(
            split_dir, mri_feat_model, class_names, max_samples_per_class=max_samples_per_class
        )
        np.savez_compressed(cache_path, feats=mri_feats, y=y_mri)
        logger.info("Cached extracted MRI features to %s", cache_path)

    # ── Load and preprocess clinical data
    if csv_path and os.path.exists(csv_path):
        df_clinical = pd.read_csv(csv_path)
    else:
        logger.warning("Using SYNTHETIC clinical data for fusion training (DEMO).")
        df_clinical = generate_synthetic_clinical_data(
            n_samples=len(mri_feats), class_names=class_names
        )
        # Align labels with MRI labels
        le_label = LabelEncoder().fit(class_names)
        df_clinical["tumor_class"] = le_label.inverse_transform(y_mri % num_classes)

    clin_vecs = clin_preprocessor.transform(df_clinical)
    clin_extracted = clin_feat_model.predict(clin_vecs, verbose=0)

    # ── Use MRI label ordering (primary)
    n = min(len(mri_feats), len(clin_extracted))
    X_mri  = mri_feats[:n]
    X_clin = clin_extracted[:n]
    y_all  = y_mri[:n]
    y_cat  = to_categorical(y_all, num_classes=num_classes)

    # ── Train / Val / Test split
    idx = np.arange(n)
    idx_tr, idx_tmp = train_test_split(idx, test_size=1 - config.TRAIN_RATIO,
                                        random_state=config.RANDOM_SEED)
    idx_val, idx_te = train_test_split(
        idx_tmp,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
    )

    logger.info("Fusion dataset split → train:%d  val:%d  test:%d",
                len(idx_tr), len(idx_val), len(idx_te))

    # ── Build fusion model
    model = build_fusion_model(mri_feat_dim, clin_feat_dim, num_classes)
    model.compile(
        optimizer = "adam",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()

    # ── Train
    os.makedirs(config.TRAINED_DIR, exist_ok=True)
    history = model.fit(
        [X_mri[idx_tr], X_clin[idx_tr]],
        y_cat[idx_tr],
        validation_data = ([X_mri[idx_val], X_clin[idx_val]], y_cat[idx_val]),
        epochs     = epochs,
        batch_size = batch_size,
        callbacks  = [
            ModelCheckpoint(config.FUSION_MODEL_PATH,
                            monitor="val_accuracy", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_accuracy", patience=10,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=5, verbose=1),
        ],
        verbose=1,
    )

    # ── Evaluate
    y_pred_probs = model.predict([X_mri[idx_te], X_clin[idx_te]], verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = np.argmax(y_cat[idx_te], axis=1)

    metrics = evaluate_model_from_arrays(
        model,
        [X_mri[idx_te], X_clin[idx_te]],
        y_cat[idx_te],
        class_names,
        model_name="Fusion",
    )

    # ── Save history
    hist_path = os.path.join(config.TRAINED_DIR, "fusion_history.json")
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)

    logger.info("=" * 60)
    logger.info("Fusion model training complete.")
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  F1-Score : %.4f", metrics["f1"])
    logger.info("Model saved → %s", config.FUSION_MODEL_PATH)
    logger.info("=" * 60)
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NeuroFusion AI multimodal fusion classifier"
    )
    parser.add_argument("--data-dir", default=config.RAW_DIR)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-samples-per-class", type=int, default=150)
    args = parser.parse_args()
    train_fusion_model(args.data_dir, args.csv, args.epochs, max_samples_per_class=args.max_samples_per_class)
