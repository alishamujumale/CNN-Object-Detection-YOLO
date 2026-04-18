"""
Car Object Detection using ResNet50
Dataset: https://www.kaggle.com/datasets/sshikamaru/car-object-detection
Compares SGD, Adam, RMSprop, Adagrad optimizers
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # ← saves to file without opening a window
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE     = 224          # ResNet50 default input size
BATCH_SIZE   = 16
EPOCHS       = 30           # Increase for better accuracy
NUM_CLASSES  = 1            # Binary: car vs background (or multi-class)
TRAIN_DIR    = "data/training_images"
TEST_DIR     = "data/testing_images"
ANNOT_FILE   = "data/train_solution_bounding_boxes (1).csv"
SEED         = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
class CarDetectionDataset:
    """Loads images + bounding box annotations."""

    def __init__(self, img_dir, annot_csv, img_size=IMG_SIZE):
        self.img_dir  = img_dir
        self.img_size = img_size
        self.df       = pd.read_csv(annot_csv)
        self._preprocess()

    def _preprocess(self):
        """Normalize bbox coords to [0,1] relative to image dimensions."""
        # Expected CSV columns: image, xmin, ymin, xmax, ymax
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        # Group: one row per image (take first bbox for classification head)
        self.df_grouped = (
            self.df.groupby("image")
                   .agg(xmin=("xmin","min"), ymin=("ymin","min"),
                        xmax=("xmax","max"), ymax=("ymax","max"),
                        count=("image","count"))
                   .reset_index()
        )
        print(f"[Dataset] Total unique images: {len(self.df_grouped)}")

    def load_sample(self, row):
        """Load one image + bbox as tensors."""
        img_path = os.path.join(self.img_dir, row["image"])
        img = load_img(img_path, target_size=(self.img_size, self.img_size))
        img = img_to_array(img) / 255.0  # Normalize to [0,1]

        # Normalise bbox to [0,1] (assuming original size 676×380 from dataset)
        orig_w, orig_h = 676, 380
        bbox = np.array([
            row["xmin"] / orig_w,
            row["ymin"] / orig_h,
            row["xmax"] / orig_w,
            row["ymax"] / orig_h,
        ], dtype=np.float32)

        return img.astype(np.float32), bbox

    def build_arrays(self):
        """Returns (X, y_bbox) numpy arrays."""
        X, Y = [], []
        for _, row in self.df_grouped.iterrows():
            try:
                img, bbox = self.load_sample(row)
                X.append(img)
                Y.append(bbox)
            except Exception as e:
                pass  # Skip missing files
        return np.array(X), np.array(Y)


# ─────────────────────────────────────────────
# MODEL: ResNet50 + Bounding Box Regression Head
# ─────────────────────────────────────────────
def build_resnet50_detector(freeze_base=True):
    """
    ResNet50 backbone (ImageNet weights) + custom regression head
    for bounding-box prediction [xmin, ymin, xmax, ymax].
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = not freeze_base  # Transfer learning

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # ── Backbone ──
    x = base_model(inputs, training=False)

    # ── Detection Head ──
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(4, activation="sigmoid", name="bbox")(x)
    # sigmoid → outputs in [0,1] matching our normalised labels

    model = keras.Model(inputs, outputs, name="ResNet50_CarDetector")
    return model


# ─────────────────────────────────────────────
# METRIC: IoU (Intersection over Union)
# ─────────────────────────────────────────────
class IoUMetric(keras.metrics.Metric):
    def __init__(self, name="iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.iou_sum   = self.add_weight(name="iou_sum",   initializer="zeros")
        self.count     = self.add_weight(name="count",     initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        inter_x1 = tf.maximum(y_true[:,0], y_pred[:,0])
        inter_y1 = tf.maximum(y_true[:,1], y_pred[:,1])
        inter_x2 = tf.minimum(y_true[:,2], y_pred[:,2])
        inter_y2 = tf.minimum(y_true[:,3], y_pred[:,3])

        inter_area = tf.maximum(0.0, inter_x2 - inter_x1) * \
                     tf.maximum(0.0, inter_y2 - inter_y1)

        area_true = (y_true[:,2]-y_true[:,0]) * (y_true[:,3]-y_true[:,1])
        area_pred = (y_pred[:,2]-y_pred[:,0]) * (y_pred[:,3]-y_pred[:,1])
        union_area = area_true + area_pred - inter_area + 1e-7

        iou = inter_area / union_area
        self.iou_sum.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.iou_sum / self.count

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)


# ─────────────────────────────────────────────
# OPTIMISER COMPARISON EXPERIMENT
# ─────────────────────────────────────────────
OPTIMIZERS = {
    
    "SGD":     keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    "Adam":    keras.optimizers.Adam(learning_rate=1e-4),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=1e-4),
    "Adagrad": keras.optimizers.Adagrad(learning_rate=1e-3),
    
    
}

def augment(image, bbox):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, bbox

def run_experiment(X_train, y_train, X_val, y_val, epochs=EPOCHS):
    """Train ResNet50 with each optimiser and collect histories."""
    results = {}

    for opt_name, optimizer in OPTIMIZERS.items():
        print(f"\n{'='*50}")
        print(f"  Training with {opt_name}")
        print(f"{'='*50}")

        model = build_resnet50_detector(freeze_base=True)
        model.compile(
            optimizer=optimizer,
            loss="mse",           # Mean Squared Error for bbox regression
            metrics=[IoUMetric(), "mae"]
        )

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_iou", patience=5,
                restore_best_weights=True, mode="max"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-7, verbose=1
            ),
        ]

        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.map(augment).shuffle(200).batch(BATCH_SIZE).prefetch(1)

        history = model.fit(
            train_ds,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=cb,
            verbose=1
        )

        val_iou  = max(history.history.get("val_iou", [0]))
        val_loss = min(history.history["val_loss"])
        results[opt_name] = {
            "history": history.history,
            "model":   model,
            "val_iou": val_iou,
            "val_loss": val_loss,
        }
        print(f"  → Best Val IoU : {val_iou:.4f}")
        print(f"  → Best Val Loss: {val_loss:.4f}")

    return results


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_optimizer_comparison(results, save_path="outputs/optimizer_comparison.png"):
    """Side-by-side training curves for all optimisers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors    = {"SGD":"#E63946","Adam":"#2A9D8F","RMSprop":"#E9C46A","Adagrad":"#264653"}

    metrics = [("loss","Train Loss"),("val_loss","Validation Loss"),("val_iou","Validation IoU")]

    for ax, (key, title) in zip(axes, metrics):
        for name, res in results.items():
            h = res["history"]
            if key in h:
                ax.plot(h[key], label=name, color=colors[name], linewidth=2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("ResNet50 Car Detector — Optimiser Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")


def plot_predictions(model, X_val, y_val, n=6,
                     save_path="outputs/sample_predictions.png"):
    """Visualise predicted vs ground-truth bounding boxes."""
    preds = model.predict(X_val[:n])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        img = X_val[i]
        h, w = img.shape[:2]
        ax.imshow(img)

        # Ground-truth (green)
        gt = y_val[i]
        rect_gt = patches.Rectangle(
            (gt[0]*w, gt[1]*h), (gt[2]-gt[0])*w, (gt[3]-gt[1])*h,
            linewidth=2, edgecolor="#00FF88", facecolor="none", label="GT"
        )
        # Prediction (red)
        pr = preds[i]
        rect_pr = patches.Rectangle(
            (pr[0]*w, pr[1]*h), (pr[2]-pr[0])*w, (pr[3]-pr[1])*h,
            linewidth=2, edgecolor="#FF4444", facecolor="none", label="Pred",
            linestyle="--"
        )
        ax.add_patch(rect_gt)
        ax.add_patch(rect_pr)
        ax.legend(fontsize=8)
        ax.axis("off")

    plt.suptitle("Predicted (red) vs Ground Truth (green) Bounding Boxes",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {save_path}")


def print_summary_table(results):
    """Print a ranked comparison table."""
    print("\n" + "="*55)
    print(f"{'Optimiser':<12} {'Val IoU':>10} {'Val Loss':>12} {'Rank':>6}")
    print("="*55)
    ranked = sorted(results.items(), key=lambda x: x[1]["val_iou"], reverse=True)
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"{name:<12} {res['val_iou']:>10.4f} {res['val_loss']:>12.4f} {rank:>6}")
    print("="*55)
    best = ranked[0][0]
    print(f"\n✅ Best Optimiser: {best} with IoU = {ranked[0][1]['val_iou']:.4f}")
    return ranked[0][0], ranked[0][1]["model"]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Car Object Detection — ResNet50")
    print("="*55)

    # 1. Load data
    print("\n[1/4] Loading dataset …")
    dataset = CarDetectionDataset(TRAIN_DIR, ANNOT_FILE)
    X, y    = dataset.build_arrays()
    print(f"      X shape: {X.shape}  |  y shape: {y.shape}")

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"      Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 3. Experiment
    print("\n[2/4] Running optimiser comparison …")
    results = run_experiment(X_train, y_train, X_val, y_val, epochs=EPOCHS)

    # 4. Plots & summary
    print("\n[3/4] Generating plots …")
    plot_optimizer_comparison(results)

    best_opt, best_model = print_summary_table(results)

    print("\n[4/4] Visualising predictions from best model …")
    plot_predictions(best_model, X_val, y_val)

    # Save best model
    best_model.save(f"outputs/best_model_{best_opt}.keras")
    print(f"\n✅ Best model saved → outputs/best_model_{best_opt}.keras")
