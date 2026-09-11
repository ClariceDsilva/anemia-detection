# app.py

import os
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from predict import load_model, predict_multiple, apply_symptom_modifier, is_relevant_hand_image
from extensions import db, login_manager
from models import User, PredictionHistory
from auth import auth_bp

BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024

app = Flask(__name__)

# SECRET_KEY: read from the environment in real deployments. The literal
# string below is ONLY a fallback for local development so the app still
# runs out of the box — it is not a real secret and must not be relied on
# outside local dev.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-key")

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# --- Database ---
# Local development: SQLite (default, zero-config).
# Deployment: set DATABASE_URL to a PostgreSQL connection string.
# Some hosting providers (e.g. Render, Heroku) hand out DATABASE_URL using
# the legacy "postgres://" scheme, but SQLAlchemy 1.4+ only accepts
# "postgresql://" — normalize it here so either form works without any
# extra configuration on the deployment side.
os.makedirs(app.instance_path, exist_ok=True)
database_url = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(app.instance_path, 'anemiaai.db')}",
)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
login_manager.init_app(app)

app.register_blueprint(auth_bp)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Create tables automatically for local development if they don't exist yet.
with app.app_context():
    db.create_all()

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
def landing():
    return render_template("landing.html")

@app.route("/tool")
@login_required
def index():
    return render_template("index.html", model_loaded=MODEL is not None)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

@app.route("/history")
@login_required
def history():
    # Strictly scoped to the logged-in user — never another user's data.
    # Newest first, capped at the latest 50 for this initial version.
    predictions = (
        PredictionHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template("history.html", predictions=predictions)

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if MODEL is None:
        flash("Model not loaded")
        return redirect(url_for("index"))

    files = request.files.getlist("images")

    if not files or all(f.filename == "" for f in files):
        flash("No valid images")
        return redirect(url_for("index"))

    # --- Validate every uploaded file BEFORE saving or predicting anything ---
    # Files are decoded straight from the uploaded bytes in memory (never
    # written to disk here), so a rejected image is never persisted, even
    # temporarily. If ANY file in the batch is invalid/corrupt/irrelevant,
    # the ENTIRE batch is rejected and the model is never called.
    decoded = []          # (img, filename) pairs for files that pass every check
    rejection_reasons = []

    for file in files:
        if not file or file.filename == "":
            continue

        if not allowed_file(file.filename):
            rejection_reasons.append("disallowed_extension")
            continue

        raw_bytes = file.read()
        img_array = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        is_valid, reason = is_relevant_hand_image(img)
        if not is_valid:
            rejection_reasons.append(reason)
            continue

        decoded.append((img, secure_filename(file.filename)))

    if rejection_reasons:
        content_rejected = any(
            reason == "face_detected"
            or reason.startswith("low_skin_ratio")
            or reason.startswith("high_edge_density")
            for reason in rejection_reasons
        )
        if content_rejected:
            flash("Please upload only clear images of a hand, palm, or fingernail.")
        else:
            flash("One or more files are not valid images. Please upload clear JPG, JPEG, or PNG images.")
        return redirect(url_for("index"))

    if not decoded:
        flash("No valid images")
        return redirect(url_for("index"))

    # --- Every image in the batch passed validation — now (and only now)
    # save and run the existing, unchanged prediction flow ---
    images = []
    previews = []

    for img, filename in decoded:
        path = UPLOAD_FOLDER / f"{uuid.uuid4()}_{filename}"
        cv2.imwrite(str(path), img)

        resized = cv2.resize(img, (224, 224))
        images.append(resized)
        previews.append(img_to_b64(resized))

    result = predict_multiple(MODEL, images)

    score = int(request.form.get("symptom_score", 0))
    if score > 0:
        result = apply_symptom_modifier(result, score)

    # Save a metadata-only history record for this prediction. The ML
    # result the user sees is already fully computed above and does not
    # depend on this succeeding — if history storage fails (e.g. the
    # database is temporarily unreachable), we roll back that failed
    # transaction, log it clearly, and still return the user's real,
    # correctly-computed prediction below. A history-write failure must
    # never block, alter, or partially report the actual prediction result.
    try:
        history_entry = PredictionHistory(
            user_id=current_user.id,
            risk_level=result["risk_level"],
            risk_color=result["risk_color"],
            anemic_probability=result["anemic_probability"],
            confidence_pct=result["confidence_pct"],
            symptom_score=score,
            num_images=len(images),
            doctor_advice=result["doctor_advice"],
        )
        db.session.add(history_entry)
        db.session.commit()
    except Exception:
        db.session.rollback()
        log.exception("Failed to save prediction history — continuing without it")

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
    app.run(host="0.0.0.0", port=10000)