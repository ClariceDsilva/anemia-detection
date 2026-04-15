# app.py

import os
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime

import cv2
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from predict import load_model, predict_multiple, apply_symptom_modifier

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)
app.secret_key = "secret"
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

try:
    MODEL, _ = load_model()
    log.info("Model Loaded")
except:
    MODEL = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def img_to_b64(img):
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL is not None)

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        flash("Model not loaded")
        return redirect(url_for("index"))

    files = request.files.getlist("images")

    images = []
    previews = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = UPLOAD_FOLDER / f"{uuid.uuid4()}_{filename}"
            file.save(path)

            img = cv2.imread(str(path))
            img = cv2.resize(img, (224, 224))

            images.append(img)
            previews.append(img_to_b64(img))

    if not images:
        flash("No valid images")
        return redirect(url_for("index"))

    result = predict_multiple(MODEL, images)

    score = int(request.form.get("symptom_score", 0))
    if score > 0:
        result = apply_symptom_modifier(result, score)

    return render_template(
        "result.html",
        anemic_probability=result["anemic_probability"],
        risk_level=result["risk_level"],
        risk_color=result["risk_color"],
        doctor_advice=result["doctor_advice"],
        confidence_pct=result["confidence_pct"],
        previews=previews,
        individual=result["individual_results"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

if __name__ == "__main__":
    app.run(debug=True)