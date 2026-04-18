# 🚗 Car Object Detection — ResNet50

> Transfer-learning based bounding-box detector with **optimizer comparison** (SGD · Adam · RMSprop · Adagrad)

---

## 📁 Project Structure

```
car_detection/
├── data/
│   ├── training_images/        ← 1000 training images from Kaggle
│   ├── testing_images/         ← test images
│   └── train_solution_bounding_boxes (1).csv
├── src/
│   ├── train.py                ← Main training + experiment runner
│   └── predict.py              ← Single-image inference
├── notebooks/
│   └── car_detection.ipynb     ← Interactive step-by-step notebook
├── outputs/                    ← Saved models + plots (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone / download this project
cd car_detection

# 2. Create virtual environment
python -m venv venv
source .\venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset from Kaggle
#    https://www.kaggle.com/datasets/sshikamaru/car-object-detection
#    Place contents inside the data/ folder
```

---

## 🚀 Running the Project

### Option A — Full Training Script
```bash
python src/train.py
```
This will:
1. Load & preprocess all 1000 training images
2. Train ResNet50 with **4 different optimisers** sequentially
3. Save comparison plots → `outputs/optimizer_comparison.png`
4. Save best model → `outputs/best_model_<OPT>.keras`

### Option B — Interactive Notebook
```bash
jupyter notebook notebooks/car_detection.ipynb
```

### Option C — Inference on a Single Image
```bash
python src/predict.py \
  --image data/testing_images/vid_5_10000.jpg \
  --model outputs/best_model_Adam.keras \
  --output outputs/result.png
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
ResNet50 Backbone (ImageNet weights, frozen for transfer learning)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, ReLU) → BatchNorm → Dropout(0.4)
    ↓
Dense(256, ReLU) → Dropout(0.3)
    ↓
Dense(4, Sigmoid)  ← [xmin, ymin, xmax, ymax] normalised to [0,1]
```

---

## 📊 Optimizer Comparison

| Optimizer | Learning Rate | Notes |
|-----------|--------------|-------|
| **SGD** | 1e-3 (momentum=0.9) | Classic, stable but slow |
| **Adam** | 1e-4 | Adaptive — usually best for vision tasks |
| **RMSprop** | 1e-4 | Good for noisy gradients |
| **Adagrad** | 1e-3 | Accumulates gradients; can slow down |

**Loss function:** MSE (Mean Squared Error) on normalised bbox coords  
**Metric:** IoU (Intersection over Union)  
**Early Stopping:** patience=5 on val_iou

---

## 📈 Training Strategy

1. **Phase 1 — Transfer Learning:** Backbone frozen, only head trained
2. **Phase 2 — Fine-Tuning:** Top 30 ResNet layers unfrozen, Adam @ 1e-5

**Callbacks used:**
- `EarlyStopping` (monitor: val_iou, mode: max, patience: 5)
- `ReduceLROnPlateau` (factor: 0.5, patience: 3)

---

## 🔬 Key Concepts

| Concept | Description |
|---------|-------------|
| **ResNet50** | 50-layer deep residual network with skip connections to avoid vanishing gradients |
| **Transfer Learning** | Pre-trained ImageNet weights give strong visual features out-of-the-box |
| **Bounding Box Regression** | Predict 4 continuous values [xmin,ymin,xmax,ymax] normalised to [0,1] |
| **IoU** | Standard object-detection metric: area of overlap / area of union |
| **MSE Loss** | Penalises large bbox prediction errors quadratically |

---

## 📦 Outputs

| File | Description |
|------|-------------|
| `outputs/optimizer_comparison.png` | Side-by-side loss + IoU curves for all 4 optimisers |
| `outputs/sample_predictions.png` | Val images with GT (green) vs Pred (red) boxes |
| `outputs/best_model_<OPT>.keras` | Best model weights from experiment |
| `outputs/final_model.keras` | Fine-tuned final model |

---

## 💡 Tips to Improve Accuracy

- Increase `EPOCHS` to 30–50 and let early stopping decide
- Add data augmentation (flips, brightness, zoom) via `tf.keras.preprocessing`
- Use anchor-based detection (YOLO / SSD) for multi-car images
- Unfreeze more ResNet layers during fine-tuning

---

*Built for educational purposes — Mini Project, Computer Vision*
