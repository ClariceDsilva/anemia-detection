# predict.py

import pickle
import numpy as np
import cv2
from pathlib import Path

MODEL_DIR = Path("model")
IMG_SIZE = 224
CLASSES = ["anemic", "normal"]

def load_model():
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError("Run train.py first")

    with open(path, "rb") as f:
        model = pickle.load(f)

    return model, {}

def extract_features(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten() / 255.0

def predict_single(model, img):
    feats = extract_features(img).reshape(1, -1)
    probs = model.predict_proba(feats)[0]

    return {
        "label": CLASSES[np.argmax(probs)],
        "anemic_prob": float(probs[0]),
        "normal_prob": float(probs[1]),
        "confidence": float(np.max(probs) * 100)
    }

def get_risk_and_advice(avg_prob):
    if avg_prob < 0.35:
        return (
            "Low",
            "green",
            "No major signs of anemia. Maintain a healthy diet rich in iron (spinach, dates, legumes)."
        )
    elif avg_prob < 0.65:
        return (
            "Medium",
            "orange",
            "Possible early signs of anemia. Consider iron-rich foods and consult a doctor if symptoms persist."
        )
    else:
        return (
            "High",
            "red",
            "High risk of anemia detected. Strongly recommended to consult a doctor and take a blood test immediately."
        )

def predict_multiple(model, images):
    results = [predict_single(model, img) for img in images]

    avg_prob = np.mean([r["anemic_prob"] for r in results])

    risk, color, advice = get_risk_and_advice(avg_prob)

    return {
        "risk_level": risk,
        "risk_color": color,
        "doctor_advice": advice,
        "anemic_probability": avg_prob,
        "confidence_pct": round(avg_prob * 100, 2),
        "individual_results": results
    }

def apply_symptom_modifier(result, score):
    if score == 0:
        return result

    new_prob = min(1.0, result["anemic_probability"] + score * 0.03)

    result["anemic_probability"] = new_prob
    result["confidence_pct"] = round(new_prob * 100, 2)

    # Recalculate risk after symptom boost
    risk, color, advice = get_risk_and_advice(new_prob)

    result["risk_level"] = risk
    result["risk_color"] = color
    result["doctor_advice"] = advice

    return result