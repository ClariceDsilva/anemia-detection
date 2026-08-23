# auth.py
#
# Authentication routes: /signup, /login, /logout.
#
# Registered with no url_prefix, so the routes live at exactly /signup,
# /login, /logout as specified — Flask still namespaces the *endpoint names*
# as "auth.signup", "auth.login", "auth.logout" internally (that's just how
# Blueprints work), which is why url_for() calls below use the "auth."
# prefix even though the URLs themselves are not prefixed.

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user

from extensions import db
from models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        # --- Validation ---
        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return render_template("signup.html", username=username, email=email)

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", username=username, email=email)

        if User.query.filter_by(username=username).first() is not None:
            flash("That username is already taken.", "error")
            return render_template("signup.html", username=username, email=email)

        if User.query.filter_by(email=email).first() is not None:
            flash("An account with that email already exists.", "error")
            return render_template("signup.html", username=username, email=email)

        # --- Create the user (password is hashed, never stored as plain text) ---
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Account created — welcome to AnemiaAI!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        password = request.form.get("password") or ""

        user = None
        if identifier:
            user = User.query.filter(
                (User.username == identifier) | (User.email == identifier.lower())
            ).first()

        # Deliberately generic error — never reveal whether the username/email
        # specifically exists, only whether the credentials as a whole were valid.
        if user is None or not user.check_password(password):
            flash("Invalid username/email or password.", "error")
            return render_template("login.html", identifier=identifier)

        login_user(user)
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("landing"))
