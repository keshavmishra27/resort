from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user
from backend.credentials import User
from backend.class_pred import classify_image
from backend.obj_count import universal_object_classifier
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
    uploaded_files = request.files.getlist('images')
    results = []
    total_score_gain = 0

    for image in uploaded_files:
        if image.filename == '':
            continue

        # Save the uploaded file
        filename = datetime.now().strftime("object_detected_%Y%m%d%H%M%S") + "_" + image.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)

        # Get object count and image score from object classifier
        object_count, image_score = universal_object_classifier(filepath, min_area=500, show_result=False)
        total_score_gain += image_score

        # Get predicted class for display purposes
        predicted_class, confidence = classify_image(filepath)

        results.append({
            'filename': filename,
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'object_count': object_count,
            'image_score': image_score
        })

    # Update current user’s score
    if current_user.is_authenticated:
        current_user.score += total_score_gain
        db.session.commit()

    return render_template("result.html", results=results)


@app_blueprint.route('/leaderboard')
def leaderboard_page():
    users = User.query.order_by(User.score.desc()).all()
    return render_template('leaderboard.html', users=users)


@app_blueprint.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(
            username=form.username.data,
            email_address=form.email_address.data,
            password=form.pswd.data,
            score=0  # Initial score set to 0
        )
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully. You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('app_blueprint.home_page'))

    if form.errors:
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
