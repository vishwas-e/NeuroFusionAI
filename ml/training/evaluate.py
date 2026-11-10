"""
NeuroFusion AI — Evaluation Module
====================================
Computes and persists all classification metrics:
  Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

Algorithms: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import os, sys, json, logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database.database import save_model_metrics

logger = logging.getLogger(__name__)


def evaluate_model(model, test_generator, class_names: list,
                   model_name: str = "Model") -> dict:
    """
    Evaluate a Keras model using a Keras generator (for MRI model).

    Parameters
    ----------
    model         : keras.Model
    test_generator: Keras data generator
    class_names   : list[str]
    model_name    : str  Label for the metrics snapshot.

    Returns
    -------
    dict  with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    import tensorflow as tf

    logger.info("Evaluating %s on test generator …", model_name)

    y_true, y_pred = [], []
    test_generator.reset()

    for batch_x, batch_y in test_generator:
        preds  = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds,   axis=1))
        y_true.extend(np.argmax(batch_y, axis=1))
        if len(y_true) >= test_generator.n:
            break

    y_true = np.array(y_true[:test_generator.n])
    y_pred = np.array(y_pred[:test_generator.n])

    return _compute_and_save(y_true, y_pred, class_names, model_name)


def evaluate_model_from_arrays(model,
                                X_test,
                                y_test_cat,
                                class_names: list,
                                model_name: str = "Model") -> dict:
    """
    Evaluate a Keras model from numpy arrays.

    Parameters
    ----------
    model       : keras.Model
    X_test      : np.ndarray or list (for multi-input models)
    y_test_cat  : np.ndarray  one-hot labels
    class_names : list[str]
    model_name  : str
    """
    logger.info("Evaluating %s on test arrays …", model_name)
    preds  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds,      axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    return _compute_and_save(y_true, y_pred, class_names, model_name)


def _compute_and_save(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      class_names: list,
                      model_name: str) -> dict:
    """Internal: compute metrics, log, save to DB and JSON."""

    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted",
                                 zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="weighted",
                              zero_division=0))
    f1   = float(f1_score(y_true, y_pred, average="weighted",
                          zero_division=0))
    cm   = confusion_matrix(y_true, y_pred).tolist()

    per_class = {}
    for i, cls in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            per_class[cls] = {
                "precision": float(precision_score(y_true == i, y_pred == i,
                                                   zero_division=0)),
                "recall"   : float(recall_score(y_true == i, y_pred == i,
                                                zero_division=0)),
                "f1"       : float(f1_score(y_true == i, y_pred == i,
                                            zero_division=0)),
                "support"  : int(mask.sum()),
            }

    metrics = {
        "model_name"       : model_name,
        "accuracy"         : round(acc,  4),
        "precision"        : round(prec, 4),
        "recall"           : round(rec,  4),
        "f1"               : round(f1,   4),
        "confusion_matrix" : cm,
        "per_class_metrics": per_class,
        "num_samples"      : len(y_true),
    }

    logger.info(
        "%s — Acc:%.4f  Prec:%.4f  Rec:%.4f  F1:%.4f",
        model_name, acc, prec, rec, f1
    )

    # Persist to database and JSON
    try:
        save_model_metrics(model_name, metrics)
    except Exception as exc:
        logger.warning("Could not save metrics to DB: %s", exc)

    save_metrics(metrics)
    return metrics


def save_metrics(metrics: dict,
                 path: str = config.METRICS_PATH):
    """
    Append/update metrics JSON file (keeps history for all models).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    model_name = metrics.get("model_name", "Unknown")
    existing[model_name] = metrics

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info("Metrics saved to %s", path)


def load_all_metrics(path: str = config.METRICS_PATH) -> dict:
    """Load all saved model metrics from JSON."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def compare_models(metrics_path: str = config.METRICS_PATH) -> dict:
    """
    Return a comparison table for the three models:
      MRI-only, Clinical-only, Fusion.

    Results come only from actual model execution.
    Returns an empty dict if no metrics have been saved yet.
    """
    all_metrics = load_all_metrics(metrics_path)
    comparison  = {}
    for name in ("MRI-only", "Clinical-only", "Fusion"):
        if name in all_metrics:
            m = all_metrics[name]
            comparison[name] = {
                "accuracy" : m.get("accuracy",  "N/A"),
                "precision": m.get("precision", "N/A"),
                "recall"   : m.get("recall",    "N/A"),
                "f1"       : m.get("f1",        "N/A"),
            }
    return comparison
