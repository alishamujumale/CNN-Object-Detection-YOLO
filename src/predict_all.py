import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ── Config ──
IMG_SIZE   = 224
TEST_DIR   = "data/testing_images"
MODEL_PATH = "outputs/best_model_Adam.keras"
OUTPUT_DIR = "outputs/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ──
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded")

# ── Get all test images ──
test_images = [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')]
print(f"Found {len(test_images)} test images")

# ── Predict each image ──
for i, img_name in enumerate(test_images):
    img_path = os.path.join(TEST_DIR, img_name)

    # Load & preprocess
    img      = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_arr  = img_to_array(img) / 255.0
    orig_img = img_to_array(load_img(img_path)).astype(np.uint8)

    # Predict
    bbox = model.predict(np.expand_dims(img_arr, axis=0), verbose=0)[0]
    bbox = np.clip(bbox, 0.0, 1.0)

    # Draw
    h, w = orig_img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.imshow(orig_img)
    ax.add_patch(patches.Rectangle(
        (bbox[0]*w, bbox[1]*h),
        (bbox[2]-bbox[0])*w,
        (bbox[3]-bbox[1])*h,
        linewidth=2, edgecolor="#FF4444", facecolor="none"
    ))
    ax.text(bbox[0]*w, bbox[1]*h - 8, "Car",
            fontsize=11, color="#FF4444", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.axis("off")
    ax.set_title(img_name, fontsize=9)
    plt.tight_layout()

    # Save
    save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

    print(f"[{i+1}/{len(test_images)}] ✅ {img_name}")

print(f"\n✅ All predictions saved → {OUTPUT_DIR}/")