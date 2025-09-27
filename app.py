# app.py
import os, time, requests, shutil
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
CSV_FILE = "readings.csv"
MODEL_FILE = "gauge_reader_model.h5"
IMG_SIZE = 128
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 6 * 1024 * 1024   # 6 MB max upload

MODEL_URL = os.getenv("MODEL_URL")  # Optional: download model from remote

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download model if not exists
if MODEL_URL and not os.path.exists(MODEL_FILE):
    print("Downloading model from MODEL_URL...")
    r = requests.get(MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    print("Model downloaded.")

# Load model
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"{MODEL_FILE} not found. Upload it or set MODEL_URL.")
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

# ---------------- APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.getenv("FLASK_SECRET", "replace-this-secret")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Unable to read image")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img.astype("float32"), axis=0)


def predict_reading(local_path):
    arr = preprocess_image(local_path)
    pred = model.predict(arr, verbose=0)[0][0]
    return float(pred)


def append_row(filename, reading, reporting_id):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        next_id = int(df["id"].max()) + 1 if ("id" in df.columns and not df.empty) else 1
    else:
        df = pd.DataFrame(columns=["id", "filename", "reading", "reporting_id"])
        next_id = 1
    new_row = {"id": next_id, "filename": filename, "reading": reading, "reporting_id": reporting_id}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return next_id


@app.route("/", methods=["GET", "POST"])
def index():
    last_entry = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["image"]
        reporting_id = request.form.get("reporting_id", "").strip()
        if not reporting_id:
            flash("Please enter a Reporting ID")
            return redirect(request.url)
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            base, ext = os.path.splitext(filename)
            safe_name = f"{base}_{int(time.time()*1000)}{ext}"
            local_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
            file.save(local_path)

            try:
                reading = predict_reading(local_path)
            except Exception as e:
                flash(f"Prediction error: {e}")
                return redirect(request.url)

            new_id = append_row(safe_name, reading, reporting_id)

            last_entry = {
                "id": new_id,
                "filename": safe_name,
                "reading": reading,
                "reporting_id": reporting_id,
            }

            return render_template("result.html",
                                   id=new_id,
                                   filename=safe_name,
                                   predicted=reading,
                                   final_reading=reading,
                                   reporting_id=reporting_id,
                                   last_entry=last_entry)
        else:
            flash("Allowed image types: png, jpg, jpeg")
            return redirect(request.url)

    # Show last entry if CSV exists
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            last_row = df.iloc[-1]
            last_entry = {
                "id": last_row["id"],
                "filename": last_row["filename"],
                "reading": last_row["reading"],
                "reporting_id": last_row["reporting_id"],
            }

    return render_template("index.html", last_entry=last_entry)


# ✅ Correction Route
@app.route("/correct", methods=["POST"])
def correct():
    reporting_id = request.form.get("reporting_id")
    correct_value = request.form.get("correct_reading")

    if not reporting_id or not correct_value:
        flash("Missing reporting ID or correct reading")
        return redirect(url_for("index"))

    if not os.path.exists(CSV_FILE):
        flash("CSV file not found")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            flash("CSV file is empty")
            return redirect(url_for("index"))

        # Update the row(s) where reporting_id matches
        mask = df["reporting_id"].astype(str) == str(reporting_id)
        if mask.any():
            df.loc[mask, "reading"] = float(correct_value)
            df.to_csv(CSV_FILE, index=False)
            flash(f"Updated reading for Reporting ID {reporting_id} to {correct_value}")
        else:
            flash(f"Reporting ID {reporting_id} not found in CSV")
    except Exception as e:
        flash(f"Error updating CSV: {e}")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
