"""
Car Object Detection — Inference Script
Usage: python src/predict.py --image path/to/image.jpg --model outputs/best_model_Adam.keras
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = 224


def load_model(model_path: str):
    """Load a saved Keras model."""
    print(f"[Model] Loading from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    return model


def preprocess_image(img_path: str, img_size: int = IMG_SIZE):
    """Load & normalise one image."""
    img = load_img(img_path, target_size=(img_size, img_size))
    arr = img_to_array(img) / 255.0
    return arr.astype(np.float32), img_to_array(img).astype(np.uint8)


def predict_bbox(model, img_array: np.ndarray):
    """Run inference; returns [xmin, ymin, xmax, ymax] in [0,1]."""
    x = np.expand_dims(img_array, axis=0)
    bbox = model.predict(x, verbose=0)[0]
    return np.clip(bbox, 0.0, 1.0)


def visualize_result(orig_img: np.ndarray, bbox: np.ndarray,
                     save_path: str = None):
    """Draw predicted bounding box on the original image."""
    h, w = orig_img.shape[:2]
    x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
    x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)

    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.imshow(orig_img)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=3, edgecolor="#FF4444", facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(x1, y1 - 8, "Car", fontsize=12, color="#FF4444",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_title("Car Detection — ResNet50", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    plt.show()


def compute_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Compute IoU between two [xmin,ymin,xmax,ymax] boxes (normalised)."""
    ix1 = max(gt[0], pred[0]); iy1 = max(gt[1], pred[1])
    ix2 = min(gt[2], pred[2]); iy2 = min(gt[3], pred[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_gt   = (gt[2] - gt[0])   * (gt[3] - gt[1])
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    union = area_gt + area_pred - inter + 1e-7
    return inter / union


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Detection Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--output", default="outputs/prediction.png",
                        help="Where to save the visualisation")
    args = parser.parse_args()

    model        = load_model(args.model)
    norm_img, orig_img = preprocess_image(args.image)
    bbox         = predict_bbox(model, norm_img)

    print(f"\n[Result] Predicted BBox (normalised): "
          f"xmin={bbox[0]:.3f} ymin={bbox[1]:.3f} "
          f"xmax={bbox[2]:.3f} ymax={bbox[3]:.3f}")

    os.makedirs("outputs", exist_ok=True)
    visualize_result(orig_img, bbox, save_path=args.output)
