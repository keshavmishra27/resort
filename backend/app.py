# backend/app.py

from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_user, logout_user, current_user
from backend.credentials import User
from backend.class_pred import classify_image
from backend.obj_count import universal_object_counter
from backend.forms import RegisterForm, LoginForm
from backend import db
from datetime import datetime
import os

app_blueprint = Blueprint('app_blueprint', __name__)

UPLOAD_FOLDER = os.path.join("backend", "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app_blueprint.context_processor
def inject_user():
    return dict(current_user=current_user)

@app_blueprint.route('/')
@app_blueprint.route('/home')
def home_page():
    return render_template('home.html')

@app_blueprint.route("/upload")
def upload_page():
    return render_template("upload.html")

@app_blueprint.route("/analyze", methods=["POST"])
def analyze():
    if current_user.is_authenticated:
        current_user.score = session.get("score", 0)
        db.session.commit()

    file = request.files.get("image")
    if not file:
        return "No file uploaded", 400

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    input_filename = f"uploaded_{timestamp}.jpg"
    output_filename = f"object_detected_{timestamp}.jpg"

    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)

    file.save(input_path)

    object_class, confidence = classify_image(input_path)
    if confidence is None:
        confidence = 0.0

    count, _ = universal_object_counter(input_path, output_path=output_path)

    previous = session.get("score", 0)
    session["score"] = previous + count
    total_score = session["score"]

    image_url = url_for("static", filename=f"uploads/{output_filename}")

    return render_template(
        "result.html",
        object_class=object_class,
        count=count,
        total_score=total_score,
        confidence=confidence,
        image_url=image_url
    )

@app_blueprint.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email_address=form.email_adress.data,
                              password=form.pswd.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully. You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('app_blueprint.home_page'))

    if form.errors != {}:
        for err_msg in form.errors.values():
            flash(err_msg, category='danger')
    return render_template('register.html', form=form)

@app_blueprint.route('/login', methods=['GET', 'POST'])
def login_page():
    flash('Please log in', category='info')
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_pswd_correction(form.pswd.data):
            login_user(attempted_user)
            flash('Logged in successfully!', category='info')
            return redirect(url_for('app_blueprint.home_page'))
        else:
            flash('Invalid username or password!', category='danger')
    return render_template('login.html', form=form)

@app_blueprint.route('/logout')
def logout_page():
    logout_user()
    flash('Logged out successfully!', category='info')
    return redirect(url_for('app_blueprint.home_page'))
