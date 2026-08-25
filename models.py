# models.py

from datetime import datetime

from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from extensions import db


class User(UserMixin, db.Model):
    """
    A registered AnemiaAI user.

    UserMixin (from Flask-Login) supplies the properties/methods Flask-Login
    expects (is_authenticated, is_active, is_anonymous, get_id()) so this
    model works directly with login_user() / current_user / @login_required.

    Passwords are NEVER stored as plain text — only a salted hash
    (via Werkzeug's generate_password_hash) is persisted.
    """

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def set_password(self, password):
        """Hash and store the given plain-text password. Never stores it as-is."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify a plain-text password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r}>"


class PredictionHistory(db.Model):
    """
    A record of one completed anemia-risk prediction, tied to the user who
    ran it.

    Metadata only — deliberately does NOT store the uploaded images, image
    file paths, or any base64/binary image data. Only the numeric/textual
    outcome of the prediction is kept, which keeps rows small and portable
    between SQLite (local dev) and PostgreSQL (deployment).
    """

    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    risk_level = db.Column(db.String(20), nullable=False)        # "Low" / "Medium" / "High"
    risk_color = db.Column(db.String(20), nullable=False)        # "green" / "orange" / "red"
    anemic_probability = db.Column(db.Float, nullable=False)
    confidence_pct = db.Column(db.Float, nullable=False)
    symptom_score = db.Column(db.Integer, nullable=False, default=0)
    num_images = db.Column(db.Integer, nullable=False)
    doctor_advice = db.Column(db.Text, nullable=False)

    user = db.relationship("User", backref=db.backref("predictions", lazy=True))

    def __repr__(self):
        return f"<PredictionHistory id={self.id} user_id={self.user_id} risk_level={self.risk_level!r}>"
