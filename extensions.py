# extensions.py
#
# Shared Flask extension instances.
#
# These are created here (unbound) and initialized against the actual Flask
# app later, in app.py, via db.init_app(app) / login_manager.init_app(app).
# Keeping them in their own module (rather than in app.py or models.py)
# avoids circular imports: models.py can import `db` from here without
# needing to import app.py, and app.py can import both `extensions` and
# `models` without either of those needing to import app.py back.

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()

login_manager = LoginManager()
# Endpoint to redirect unauthenticated users to. "auth.login" because the
# login route is defined on the "auth" Blueprint (see auth.py).
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "error"
