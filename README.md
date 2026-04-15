# 🩸 AnemiaAI — Intelligent Anemia Detection System

> **Final Year Project** — Image-based anemia risk assessment using Machine Learning

---

## Overview

AnemiaAI detects anemia risk from conjunctiva (inner eyelid) and nail bed images by analyzing color pallor,
saturation, and texture biomarkers — the same visual indicators clinicians examine during physical assessment.

### Risk Levels
| Range | Level | Action |
|-------|-------|--------|
| 0–40% | 🟢 Low Risk | Routine monitoring |
| 40–70% | 🟡 Medium Risk | Clinical blood test recommended |
| 70–100% | 🔴 High Risk | Urgent medical consultation |

---

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python dataset_generator.py
```

### 3. Train Model
```bash
python train.py
```

### 4. Run Web App
```bash
python app.py
```
Then open: **http://localhost:5000**

---

## Project Structure

```
anemia_project/
├── app.py                  # Flask web application
├── train.py                # Model training pipeline
├── predict.py              # Prediction module
├── dataset_generator.py    # Synthetic dataset generator
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── model/
│   ├── best_model.pkl      # Trained GBM model (generated)
│   └── model_metadata.json # Metrics & config (generated)
│
├── dataset/
│   ├── anemic/             # Anemic images (generated)
│   └── normal/             # Normal images (generated)
│
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Results page
│
├── static/
│   ├── training_curves.png # Training plot (generated)
│   ├── confusion_matrix.png# CM plot (generated)
│   └── uploads/            # Uploaded images (runtime)
│
└── logs/
    └── training.log        # Training log (generated)
```

---

## Technical Details

### Feature Extraction
Each image is processed through three feature extractors:

1. **Color Histograms (BGR + HSV)** — 192 features
   - 32-bin histograms per channel (Blue, Green, Red, Hue, Saturation, Value)
   - Captures the fundamental color distribution indicating pallor

2. **Statistical Features** — 15 features
   - Per-channel mean, std, quartiles (P10, P25, P75, P90)
   - **Pallor Index** = G_mean / R_mean (elevated in anemia)
   - **Yellowness Index** = (R + G) / (2B) (jaundice indicator)
   - HSV saturation statistics

3. **Texture Features** — 8 features
   - Laplacian variance (sharpness/texture measure)
   - Sobel gradient magnitude (edge strength)
   - Local Binary Pattern approximation

**Total: ~215 features per image**

### Model
- **Algorithm:** Gradient Boosting Classifier (sklearn)
- **n_estimators:** 200, **learning_rate:** 0.05
- **Pipeline:** StandardScaler → GradientBoostingClassifier
- **Early stopping:** Via cross-validation + best model saving
- **Evaluation:** Train/Val/Test split + AUC-ROC

### Dataset
- **Synthetic** dataset generated using OpenCV
- Anemic images: pale, low-saturation, yellowish hue
- Normal images: healthy pink-red, well-vascularized
- Augmentation: flip, rotation (±10°, ±20°), brightness ±20%, color jitter
- ~800 base images, ~1600+ after augmentation

---

## API

```bash
# JSON API endpoint
curl -X POST http://localhost:5000/api/predict \
  -F "images=@your_image.jpg"
```

Response:
```json
{
  "risk_level": "Medium Risk",
  "anemic_probability": 0.543,
  "confidence_pct": 54.3,
  "num_images": 1
}
```

---

## Disclaimer

This tool is for **educational purposes only**. It does not constitute medical advice and cannot replace
professional clinical diagnosis. Anemia must be confirmed through laboratory tests (CBC, hemoglobin, ferritin).
Always consult a licensed healthcare provider.

---

*Built with Flask · OpenCV · Scikit-learn · Python 3*
