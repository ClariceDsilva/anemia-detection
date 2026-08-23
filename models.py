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
