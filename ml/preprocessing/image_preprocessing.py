"""
NeuroFusion AI — MRI Image Preprocessing
=========================================
Handles image loading, validation, resizing, normalisation,
and data-augmentation pipeline for ResNet50 input.

Algorithm: MRI Feature Extraction / Data Augmentation
"""

import os
import io
import logging
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Core image helpers
# ──────────────────────────────────────────────────────────────

def validate_image_file(file_path_or_bytes) -> bool:
    """
    Validate that the supplied file is a readable image with a
    supported extension.

    Parameters
    ----------
    file_path_or_bytes : str | bytes | BytesIO
        A filesystem path, raw bytes, or a BytesIO object.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    try:
        if isinstance(file_path_or_bytes, str):
            ext = os.path.splitext(file_path_or_bytes)[1].lower()
            if ext not in config.SUPPORTED_IMAGE_FORMATS:
                logger.warning("Unsupported image extension: %s", ext)
                return False
            img = Image.open(file_path_or_bytes)
        else:
            img = Image.open(io.BytesIO(file_path_or_bytes)
                             if isinstance(file_path_or_bytes, bytes)
                             else file_path_or_bytes)

        img.verify()          # Raises if corrupt
        return True
    except Exception as exc:
        logger.error("Image validation failed: %s", exc)
        return False


def load_and_preprocess_image(image_source, target_size=None) -> np.ndarray:
    """
    Load an MRI image from a file path, bytes, or file-like object,
    resize it to *target_size* and apply ResNet50 preprocessing.

    Parameters
    ----------
    image_source : str | bytes | BytesIO | PIL.Image
        The image to load.
    target_size : tuple[int, int], optional
        (width, height).  Defaults to config.IMAGE_SIZE.

    Returns
    -------
    np.ndarray
        Preprocessed image array, shape (H, W, 3).
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE

    # ── Load
    if isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source)
    elif isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source))
    elif hasattr(image_source, "read"):           # file-like / UploadedFile
        img = Image.open(image_source)
    else:
        img = Image.open(str(image_source))

    # ── Ensure RGB (handles grayscale / RGBA MRI scans)
    img = img.convert("RGB")

    # ── Resize
    img = img.resize(target_size, Image.LANCZOS)

    # ── To numpy + ResNet50 preprocessing (scales to [-1, 1])
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return arr


def load_image_for_display(image_source, target_size=None) -> np.ndarray:
    """
    Load an MRI image for display purposes (0-255 uint8, RGB).
    Used by Grad-CAM overlay and the Streamlit UI.
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE

    if isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source))
    elif hasattr(image_source, "read"):
        content = image_source.read()
        img = Image.open(io.BytesIO(content))
    elif isinstance(image_source, Image.Image):
        img = image_source
    else:
        img = Image.open(str(image_source))

    img = img.convert("RGB").resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
#  Batch / dataset loaders
# ──────────────────────────────────────────────────────────────

def create_data_generators(train_dir: str,
                           val_dir: str,
                           test_dir: str,
                           batch_size: int = config.BATCH_SIZE):
    """
    Create Keras ImageDataGenerators for training, validation, and test
    datasets organised as  <split>/<class_name>/<image_file>.

    Training generator applies data augmentation as per config.AUGMENTATION.
    Validation and test generators apply only ResNet50 preprocessing.

    Parameters
    ----------
    train_dir, val_dir, test_dir : str
        Root directories for each split.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    tuple
        (train_gen, val_gen, test_gen, class_names)
    """
    aug = config.AUGMENTATION

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=aug["rotation_range"],
        width_shift_range=aug["width_shift_range"],
        height_shift_range=aug["height_shift_range"],
        horizontal_flip=aug["horizontal_flip"],
        zoom_range=aug["zoom_range"],
        brightness_range=aug["brightness_range"],
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=config.IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_gen.class_indices.keys())
    logger.info("Classes detected: %s", class_names)

    return train_gen, val_gen, test_gen, class_names


def split_dataset(source_dir: str,
                  output_dir: str,
                  train_ratio: float = config.TRAIN_RATIO,
                  val_ratio: float   = config.VAL_RATIO):
    """
    Organise a flat dataset (source_dir/<class>/<images>) into
    train / val / test sub-directories for use with Keras generators.

    Existing splits are not overwritten.

    Parameters
    ----------
    source_dir : str
        Directory containing one sub-folder per class.
    output_dir : str
        Destination directory; train/, val/, test/ will be created inside.
    train_ratio : float
        Fraction of data for training.
    val_ratio : float
        Fraction of data for validation (remainder → test).
    """
    import shutil, random

    random.seed(config.RANDOM_SEED)

    splits = {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio}

    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path)
                 if os.path.splitext(f)[1].lower() in config.SUPPORTED_IMAGE_FORMATS]
        random.shuffle(files)

        n      = len(files)
        n_tr   = int(n * train_ratio)
        n_val  = int(n * val_ratio)

        boundaries = {
            "train": files[:n_tr],
            "val"  : files[n_tr:n_tr + n_val],
            "test" : files[n_tr + n_val:],
        }

        for split, split_files in boundaries.items():
            dest = os.path.join(output_dir, split, class_name)
            os.makedirs(dest, exist_ok=True)
            for fname in split_files:
                src = os.path.join(class_path, fname)
                dst = os.path.join(dest, fname)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        logger.info("Class %s → train:%d  val:%d  test:%d",
                    class_name,
                    len(boundaries["train"]),
                    len(boundaries["val"]),
                    len(boundaries["test"]))

    logger.info("Dataset split complete → %s", output_dir)


def preprocess_batch(image_paths: list) -> np.ndarray:
    """
    Preprocess a list of image file paths into a batch tensor.

    Parameters
    ----------
    image_paths : list[str]

    Returns
    -------
    np.ndarray  shape (N, H, W, 3)
    """
    images = []
    for path in image_paths:
        try:
            arr = load_and_preprocess_image(path)
            images.append(arr)
        except Exception as exc:
            logger.warning("Skipping %s — %s", path, exc)
    return np.array(images, dtype=np.float32)
