from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

# ============================================
# Project Paths
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = BASE_DIR / "frontend" / "templates"
STATIC_DIR = BASE_DIR / "frontend" / "static"

MODEL_DIR = BASE_DIR / "backend" / "models"

MODEL_PATH = MODEL_DIR / "flood_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"



app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR)
)


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)



FEATURES = [
    "Temp",
    "Humidity",
    "Cloud Cover",
    "ANNUAL",
    "Jan-Feb",
    "Mar-May",
    "Jun-Sep",
    "Oct-Dec",
    "avgjune",
    "sub"
]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    try:

        values = []

        for feature in FEATURES:
            values.append(float(request.form[feature]))

        input_df = pd.DataFrame([values], columns=FEATURES)

        scaled_data = scaler.transform(input_df)

        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            return render_template("chance.html")

        return render_template("no_chance.html")

    except Exception as e:
        return f"<h2>Error</h2><p>{e}</p>"



if __name__ == "__main__":
    app.run(debug=True)