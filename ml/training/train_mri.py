"""
NeuroFusion AI — Training: MRI Model (ResNet50 Transfer Learning)
=================================================================
Phase 1: Train MRI-only classifier on the brain-tumor dataset.
Phase 2: Fine-tune top ResNet50 layers for higher accuracy.

Run from the project root:
    python training/train_mri.py

Algorithm: CNN, Transfer Learning, ResNet50, Data Augmentation,
           Train/Val/Test Split, Accuracy, Precision, Recall, F1
"""

import os, sys, json, logging, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing.image_preprocessing import split_dataset, create_data_generators
from models.model_utils import (
    build_mri_feature_extractor, save_class_names, load_class_names
)
from training.evaluate import evaluate_model, save_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Callbacks
# ──────────────────────────────────────────────────────────────

def build_callbacks(model_name: str = "mri"):
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    )
    os.makedirs(config.TRAINED_DIR, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath   = config.MRI_MODEL_PATH,
            monitor    = "val_accuracy",
            save_best_only= True,
            verbose    = 1,
        ),
        EarlyStopping(
            monitor    = "val_accuracy",
            patience   = 8,
            restore_best_weights= True,
            verbose    = 1,
        ),
        ReduceLROnPlateau(
            monitor    = "val_loss",
            factor     = 0.3,
            patience   = 4,
            min_lr     = 1e-7,
            verbose    = 1,
        ),
        CSVLogger(
            os.path.join(config.TRAINED_DIR, f"{model_name}_training_log.csv"),
            append=True,
        ),
    ]


# ──────────────────────────────────────────────────────────────
#  Main training routine
# ──────────────────────────────────────────────────────────────

def train_mri_model(data_dir: str = config.RAW_DIR,
                    epochs_phase1: int = config.EPOCHS,
                    epochs_phase2: int = 10):
    """
    Train the MRI ResNet50 classifier.

    Phase 1 — Feature extraction (base frozen)
    Phase 2 — Fine-tuning (top layers unfrozen)

    Parameters
    ----------
    data_dir      : str   Directory with <class>/<images> sub-folders.
    epochs_phase1 : int   Epochs for frozen-base training.
    epochs_phase2 : int   Epochs for fine-tuning.
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    logger.info("=" * 60)
    logger.info("NeuroFusion AI — MRI Model Training")
    logger.info("=" * 60)

    # ── 1. Prepare dataset splits
    processed_dir = config.PROCESSED_DIR
    split_dir     = os.path.join(processed_dir, "splits")

    logger.info("Splitting dataset from %s …", data_dir)
    split_dataset(data_dir, split_dir)

    train_dir = os.path.join(split_dir, "train")
    val_dir   = os.path.join(split_dir, "val")
    test_dir  = os.path.join(split_dir, "test")

    # ── 2. Create data generators
    train_gen, val_gen, test_gen, class_names = create_data_generators(
        train_dir, val_dir, test_dir, batch_size=config.BATCH_SIZE
    )
    num_classes = len(class_names)
    save_class_names(class_names)
    logger.info("Classes: %s  (%d)", class_names, num_classes)

    # ── 3. Build model (Phase 1 — frozen base)
    model = build_mri_feature_extractor(num_classes, trainable_base=False)
    model.compile(
        optimizer = Adam(learning_rate=config.LEARNING_RATE),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    model.summary()

    # ── 4. Phase 1 training
    logger.info("\n[Phase 1] Training with frozen ResNet50 base …")
    history1 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_phase1,
        callbacks       = build_callbacks("mri_phase1"),
        verbose         = 1,
    )

    # ── 5. Fine-tuning (Phase 2)
    logger.info("\n[Phase 2] Fine-tuning top ResNet50 layers …")
    # Unfreeze top 30 layers of ResNet50
    base = model.get_layer("resnet50")
    for layer in base.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer = Adam(learning_rate=config.FINE_TUNE_LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = epochs_phase2,
        callbacks       = build_callbacks("mri_phase2"),
        initial_epoch   = len(history1.history["loss"]),
        verbose         = 1,
    )

    # ── 6. Evaluate on test set
    logger.info("\n[Evaluation] Evaluating on test set …")
    metrics = evaluate_model(model, test_gen, class_names,
                             model_name="MRI-only")

    # ── 7. Save combined training history
    combined_history = {
        "phase1": {k: [float(v) for v in vals]
                   for k, vals in history1.history.items()},
        "phase2": {k: [float(v) for v in vals]
                   for k, vals in history2.history.items()},
    }
    hist_path = os.path.join(config.TRAINED_DIR, "mri_history.json")
    with open(hist_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    logger.info("Training history saved → %s", hist_path)

    logger.info("=" * 60)
    logger.info("MRI model training complete.")
    logger.info("  Accuracy : %.4f", metrics["accuracy"])
    logger.info("  F1-Score : %.4f", metrics["f1"])
    logger.info("Model saved → %s", config.MRI_MODEL_PATH)
    logger.info("=" * 60)

    return model, metrics


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train NeuroFusion AI MRI classifier (ResNet50)"
    )
    parser.add_argument(
        "--data-dir", default=config.RAW_DIR,
        help="Directory containing <class>/<image> sub-folders."
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS,
        help="Number of Phase-1 training epochs."
    )
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=10,
        help="Number of Phase-2 fine-tuning epochs."
    )
    args = parser.parse_args()

    # Check for actual image class subdirectories (ignore placeholder files)
    class_dirs = [
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
        and not d.startswith("__")
        and not d.startswith(".")
    ] if os.path.isdir(args.data_dir) else []

    if not class_dirs:
        print(
            f"\n{'='*60}\n"
            f"[ERROR] No dataset class folders found in '{args.data_dir}'.\n"
            f"{'='*60}\n\n"
            "The MRI dataset has not been downloaded yet.\n\n"
            "STEP 1 — Download the dataset:\n"
            "  https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset\n\n"
            "STEP 2 — Extract it into the project's data/raw/ folder\n"
            "  so that the structure looks like:\n\n"
            "  data/raw/\n"
            "  ├── glioma/        (MRI images)\n"
            "  ├── meningioma/    (MRI images)\n"
            "  ├── notumor/       (MRI images)\n"
            "  └── pituitary/     (MRI images)\n\n"
            "STEP 3 — Then re-run this command:\n"
            "  python training/train_mri.py --data-dir data/raw\n"
            f"{'='*60}\n"
        )
        sys.exit(1)

    print(f"\nFound {len(class_dirs)} class folders: {class_dirs}")

    train_mri_model(args.data_dir, args.epochs, args.fine_tune_epochs)
