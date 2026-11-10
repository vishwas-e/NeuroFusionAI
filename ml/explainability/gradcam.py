"""
NeuroFusion AI — Grad-CAM Explainability
==========================================
Generates gradient-weighted class-activation maps (Grad-CAM)
for the MRI ResNet50 branch.

Algorithm: Grad-CAM (Gradient-weighted Class Activation Mapping)

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.
"""

import os, sys, logging
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Core Grad-CAM implementation
# ──────────────────────────────────────────────────────────────

class GradCAM:
    """
    Compute Grad-CAM heatmaps for a given Keras model.
    Supports both flat models and architectures with nested backbones (e.g. ResNet50).

    Parameters
    ----------
    model       : keras.Model   Full MRI classification model.
    layer_name  : str           Name of the target convolutional layer.
                                Defaults to config.GRADCAM_LAYER.
    """

    def __init__(self, model: keras.Model,
                 layer_name: str = config.GRADCAM_LAYER):
        self.model      = model
        self.layer_name = layer_name
        self._is_nested = False
        self._sub_backbone = None
        self._head_layers = []
        self._grad_model = None

        # 1. Check if the layer exists inside a nested submodel (e.g. resnet50)
        for layer in model.layers:
            if hasattr(layer, "layers") and any(l.name == layer_name for l in layer.layers):
                self._is_nested = True
                backbone = layer
                target_layer = backbone.get_layer(layer_name)
                self._sub_backbone = keras.Model(
                    inputs=backbone.input,
                    outputs=[target_layer.output, backbone.output]
                )
                idx = model.layers.index(backbone)
                self._head_layers = model.layers[idx + 1:]
                logger.info("GradCAM attached to nested layer '%s' inside '%s'", layer_name, backbone.name)
                break

        # 2. Check direct model layers or fallback
        if not self._is_nested:
            layer_names = [l.name for l in model.layers]
            if layer_name in layer_names:
                self._grad_model = keras.Model(
                    inputs=model.inputs,
                    outputs=[model.get_layer(self.layer_name).output, model.output]
                )
            else:
                # Look deeper into any nested layer for any conv layer
                found = False
                for layer in model.layers:
                    if hasattr(layer, "layers"):
                        conv_nested = [l.name for l in layer.layers if "conv" in l.name.lower()]
                        if conv_nested:
                            self._is_nested = True
                            target_layer = layer.get_layer(conv_nested[-1])
                            self._sub_backbone = keras.Model(
                                inputs=layer.input,
                                outputs=[target_layer.output, layer.output]
                            )
                            idx = model.layers.index(layer)
                            self._head_layers = model.layers[idx + 1:]
                            found = True
                            logger.info("GradCAM fallback to nested conv layer '%s' in '%s'", conv_nested[-1], layer.name)
                            break
                if not found:
                    conv_layers = [l.name for l in model.layers
                                   if isinstance(l, keras.layers.Conv2D) or "conv" in l.name.lower()]
                    if conv_layers:
                        self.layer_name = conv_layers[-1]
                        self._grad_model = keras.Model(
                            inputs=model.inputs,
                            outputs=[model.get_layer(self.layer_name).output, model.output]
                        )
                    else:
                        raise ValueError("No convolutional layer found in model.")

    def compute_heatmap(self,
                        img_array: np.ndarray,
                        class_idx: int = None,
                        eps: float = 1e-8) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for *img_array*.

        Parameters
        ----------
        img_array : np.ndarray
            Pre-processed image with shape (H, W, 3) or (1, H, W, 3).
        class_idx : int, optional
            Class index to explain. Defaults to argmax (predicted class).
        eps : float
            Small value to prevent division by zero.

        Returns
        -------
        np.ndarray  Heatmap in [0, 1], shape (H, W).
        """
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, 0)

        img_tensor = tf.cast(img_array, tf.float32)

        if self._is_nested:
            with tf.GradientTape() as tape:
                conv_outputs, backbone_out = self._sub_backbone(img_tensor, training=False)
                x = backbone_out
                for layer in self._head_layers:
                    if "dropout" in layer.name:
                        x = layer(x, training=False)
                    else:
                        x = layer(x)
                predictions = x
                if class_idx is None:
                    class_idx = int(tf.argmax(predictions[0]))
                class_score = predictions[:, class_idx]
        else:
            with tf.GradientTape() as tape:
                tape.watch(img_tensor)
                conv_outputs, predictions = self._grad_model(img_tensor, training=False)
                if class_idx is None:
                    class_idx = int(tf.argmax(predictions[0]))
                class_score = predictions[:, class_idx]

        # Gradients of class score w.r.t. feature map activations
        grads = tape.gradient(class_score, conv_outputs)  # (1, H', W', C)

        # Global average pooling of gradients  →  importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weight the feature maps
        conv_outputs = conv_outputs[0]                         # (H', W', C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # (H', W', 1)
        heatmap = tf.squeeze(heatmap)                          # (H', W')

        # ReLU + normalise
        heatmap = tf.nn.relu(heatmap).numpy()
        denom = float(heatmap.max() - heatmap.min())
        if denom > 0:
            heatmap = (heatmap - heatmap.min()) / (denom + eps)
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap.astype(np.float32)

    def overlay_heatmap(self,
                        heatmap    : np.ndarray,
                        original_img: np.ndarray,
                        alpha      : float = 0.45,
                        colormap   : int   = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Resize *heatmap* to match *original_img* and overlay it.

        Parameters
        ----------
        heatmap      : np.ndarray   Float [0,1], shape (H', W').
        original_img : np.ndarray   uint8 RGB, shape (H, W, 3).
        alpha        : float        Blending weight for the heatmap.
        colormap     : int          OpenCV colormap constant.

        Returns
        -------
        np.ndarray  RGB uint8 overlay, same shape as original_img.
        """
        h, w = original_img.shape[:2]

        # Resize heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

        # Apply colour map
        heatmap_coloured = cv2.applyColorMap(heatmap_resized, colormap)
        heatmap_coloured = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

        # Blend
        if original_img.dtype != np.uint8:
            original_img = np.clip(original_img, 0, 255).astype(np.uint8)

        overlay = cv2.addWeighted(original_img, 1 - alpha,
                                  heatmap_coloured, alpha, 0)
        return overlay.astype(np.uint8)

    def generate(self,
                 preprocessed_img: np.ndarray,
                 original_img    : np.ndarray,
                 class_idx       : int = None,
                 alpha           : float = 0.45) -> dict:
        """
        High-level method: compute heatmap and produce overlay.

        Parameters
        ----------
        preprocessed_img : np.ndarray
            ResNet50-preprocessed image (input to the model).
        original_img     : np.ndarray
            Raw uint8 RGB image for display.
        class_idx        : int, optional
        alpha            : float

        Returns
        -------
        dict with keys:
            'heatmap'        – raw float32 heatmap (H', W')
            'overlay'        – RGB uint8 overlay (H, W, 3)
            'class_idx'      – explained class index
        """
        heatmap   = self.compute_heatmap(preprocessed_img, class_idx)
        if class_idx is None:
            # Recompute with actual argmax
            preds     = self.model.predict(
                np.expand_dims(preprocessed_img, 0), verbose=0
            )
            class_idx = int(np.argmax(preds[0]))

        overlay = self.overlay_heatmap(heatmap, original_img, alpha)

        return {
            "heatmap"  : heatmap,
            "overlay"  : overlay,
            "class_idx": class_idx,
        }


# ──────────────────────────────────────────────────────────────
#  Convenience wrapper
# ──────────────────────────────────────────────────────────────

def generate_gradcam(model       : keras.Model,
                     preprocessed: np.ndarray,
                     original_img: np.ndarray,
                     class_idx   : int = None,
                     layer_name  : str = config.GRADCAM_LAYER,
                     alpha       : float = 0.45) -> dict:
    """
    Convenience function to generate a Grad-CAM overlay.

    Parameters
    ----------
    model        : keras.Model  Full MRI classifier.
    preprocessed : np.ndarray   ResNet50-preprocessed image.
    original_img : np.ndarray   Raw uint8 RGB image for overlay.
    class_idx    : int, optional  If None, uses predicted class.
    layer_name   : str          Target convolutional layer.
    alpha        : float        Heatmap blend strength.

    Returns
    -------
    dict  {'heatmap': ..., 'overlay': ..., 'class_idx': ...}
    """
    cam = GradCAM(model, layer_name)
    return cam.generate(preprocessed, original_img, class_idx, alpha)
