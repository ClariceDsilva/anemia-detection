# predict.py

import pickle
import numpy as np
import cv2
from pathlib import Path

MODEL_DIR = Path("model")
IMG_SIZE = 224
CLASSES = ["anemic", "normal"]


# ============================================================
# Pre-prediction image validation
# ============================================================
#
# The application accepts the following intended image types:
#
#   1. Palm / hand
#   2. Fingernail / nail-bed
#   3. Eyelid / inner eyelid
#
# This validator is intentionally tolerant because nail-bed and
# eyelid images may contain much less skin-colored area than a
# palm photograph.
#
# It is a heuristic filter, NOT a trained image classifier.
#


MIN_IMAGE_DIMENSION = 32

# Previous value was 0.15, which was too strict for nail-bed
# and eyelid images. A lower threshold allows these valid inputs.
MIN_SKIN_RATIO = 0.05

# Increased from 0.05 because legitimate nail/eyelid photos can
# contain more edges and fine details.
MAX_EDGE_DENSITY = 0.12

# Wider chrominance range to accommodate:
# - normal skin
# - pale skin
# - anemic pallor
# - nail beds
# - fingernails
# - eyelid/inner-eyelid tones
SKIN_CR_RANGE = (105, 220)
SKIN_CB_RANGE = (75, 150)


def is_relevant_hand_image(img):
    """
    Validate an uploaded image before it reaches the anemia model.

    Intended inputs:
        - palm / hand
        - fingernail / nail-bed
        - eyelid

    Returns:
        (True, "ok") on success
        (False, reason) on rejection
    """

    # --------------------------------------------------------
    # 1. Image decode check
    # --------------------------------------------------------
    if img is None:
        return False, "corrupt_or_undecodable"

    # Make sure this actually looks like an OpenCV image.
    if not isinstance(img, np.ndarray):
        return False, "corrupt_or_undecodable"

    if img.ndim != 3 or img.shape[2] != 3:
        return False, "corrupt_or_undecodable"

    # --------------------------------------------------------
    # 2. Minimum image size
    # --------------------------------------------------------
    h, w = img.shape[:2]

    if h < MIN_IMAGE_DIMENSION or w < MIN_IMAGE_DIMENSION:
        return False, "image_too_small"

    # --------------------------------------------------------
    # 3. Skin / tissue color check
    # --------------------------------------------------------
    #
    # We deliberately use a LOW threshold here.
    #
    # A palm image may contain lots of skin.
    # A nail-bed image may contain mostly nail/background.
    # An eyelid image may contain mostly eye/nail-like tissue.
    #
    # Therefore 5% is used instead of the old 15%.
    #

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    cr = ycrcb[:, :, 1].astype(np.int32)
    cb = ycrcb[:, :, 2].astype(np.int32)

    skin_mask = (
        (cr >= SKIN_CR_RANGE[0])
        & (cr <= SKIN_CR_RANGE[1])
        & (cb >= SKIN_CB_RANGE[0])
        & (cb <= SKIN_CB_RANGE[1])
    )

    skin_ratio = (
        float(np.count_nonzero(skin_mask))
        / float(skin_mask.size)
    )

    if skin_ratio < MIN_SKIN_RATIO:
        return False, f"low_skin_ratio_{skin_ratio:.3f}"

    # --------------------------------------------------------
    # 4. Edge-density check
    # --------------------------------------------------------
    #
    # This is intentionally much more tolerant than before.
    #
    # Nail and eyelid photographs can naturally contain:
    # - eyelashes
    # - nail boundaries
    # - wrinkles
    # - skin texture
    # - fine edges
    #
    # We mainly want to reject extremely noisy/document-like
    # images rather than normal medical-image detail.
    #

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edge_density = (
        float(np.count_nonzero(edges))
        / float(edges.size)
    )

    if edge_density > MAX_EDGE_DENSITY:
        return False, f"high_edge_density_{edge_density:.3f}"

    # --------------------------------------------------------
    # Image passed validation
    # --------------------------------------------------------
    return True, "ok"


# ============================================================
# Existing model functions
# ============================================================


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

    new_prob = min(
        1.0,
        result["anemic_probability"] + score * 0.03
    )

    result["anemic_probability"] = new_prob
    result["confidence_pct"] = round(new_prob * 100, 2)

    # Recalculate risk after symptom boost
    risk, color, advice = get_risk_and_advice(new_prob)

    result["risk_level"] = risk
    result["risk_color"] = color
    result["doctor_advice"] = advice

    return result