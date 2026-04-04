# =============================================
# app.py  –  Flask backend
# =============================================

from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np
import os

app = Flask(__name__)

# ── Load scaler & accuracies once at startup ─────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODELS_DIR, "accuracies.json")) as f:
    ACCURACIES = json.load(f)

MODEL_FILES = {
    "Logistic Regression": "logistic.pkl",
    "Random Forest":       "random_forest.pkl",
    "KNN":                 "knn.pkl",
    "SVM":                 "svm.pkl",
}

def load_model(name):
    path = os.path.join(MODELS_DIR, MODEL_FILES[name])
    with open(path, "rb") as f:
        return pickle.load(f)

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

FEATURE_RANGES = {
    # (min, max)
    "age":      (20,  80),
    "sex":      (0,   1),
    "cp":       (0,   3),
    "trestbps": (80,  200),
    "chol":     (100, 600),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (60,  220),
    "exang":    (0,   1),
    "oldpeak":  (0.0, 7.0),
    "slope":    (0,   2),
    "ca":       (0,   4),
    "thal":     (0,   3),
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           models=list(MODEL_FILES.keys()),
                           accuracies=ACCURACIES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # ── Validate model choice ─────────────────────────────────────────────
        model_name = data.get("model", "Random Forest")
        if model_name not in MODEL_FILES:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400

        # ── Parse & validate each feature ─────────────────────────────────────
        features = []
        errors   = []

        for feat in FEATURE_NAMES:
            raw = data.get(feat)
            if raw is None or str(raw).strip() == "":
                errors.append(f"'{feat}' is required.")
                continue
            try:
                val = float(raw)
            except ValueError:
                errors.append(f"'{feat}' must be a number.")
                continue

            lo, hi = FEATURE_RANGES[feat]
            if not (lo <= val <= hi):
                errors.append(f"'{feat}' must be between {lo} and {hi}.")
                continue

            features.append(val)

        if errors:
            return jsonify({"error": " | ".join(errors)}), 422

        # ── Scale & predict ────────────────────────────────────────────────────
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        model = load_model(model_name)
        prediction = int(model.predict(X_scaled)[0])

        # probability (all 4 models support predict_proba)
        try:
            proba = model.predict_proba(X_scaled)[0]
            confidence = round(float(proba[prediction]) * 100, 1)
        except Exception:
            confidence = None

        return jsonify({
            "prediction":  prediction,
            "label":       "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "confidence":  confidence,
            "model":       model_name,
            "accuracy":    ACCURACIES.get(model_name),
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
