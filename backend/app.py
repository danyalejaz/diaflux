"""
DiaFlux ML Backend
==================
Flask API that serves the trained diabetes risk model (models/diabetes_model.pkl)
and its StandardScaler (models/scaler.pkl) produced by the project notebooks.

Endpoints
---------
GET  /api/health           -> service + model status
POST /api/predict          -> diabetes risk prediction for a single patient
POST /api/simulate         -> compare risk before/after lifestyle modifications
POST /api/recommendations  -> dietary / exercise / medical guidance for a risk level

All responses follow the exact JSON contract expected by the React frontend
(see diaflux_frontend/src/types.ts).
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Silence sklearn "model trained on a different version" notices; the
# LogisticRegression / StandardScaler artifacts unpickle fine across minor versions.
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

PORT = int(os.environ.get("PORT", "5000"))

# Exact feature order the model & scaler were trained on (from the notebook
# preprocessing pipeline: numeric columns first, then one-hot gender, then
# one-hot smoking_history).
FEATURE_ORDER = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_Female",
    "gender_Male",
    "gender_Other",
    "smoking_history_No Info",
    "smoking_history_current",
    "smoking_history_ever",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_not current",
]

# --------------------------------------------------------------------------- #
# Load model artifacts once at startup
# --------------------------------------------------------------------------- #
MODEL = None
SCALER = None
MODEL_LOAD_ERROR = None
try:
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    print(f"[DiaFlux] Loaded model ({type(MODEL).__name__}) and scaler from {MODELS_DIR}")
except Exception as exc:  # pragma: no cover - defensive startup guard
    MODEL_LOAD_ERROR = str(exc)
    print(f"[DiaFlux] WARNING: could not load model artifacts: {exc}")

# Index of the "diabetes = 1" class inside predict_proba output.
POSITIVE_CLASS_INDEX = 1
try:
    if MODEL is not None and hasattr(MODEL, "classes_"):
        POSITIVE_CLASS_INDEX = int(list(MODEL.classes_).index(1))
except (ValueError, TypeError):
    POSITIVE_CLASS_INDEX = 1

app = Flask(__name__)
CORS(app)


# --------------------------------------------------------------------------- #
# Feature engineering + inference
# --------------------------------------------------------------------------- #
def build_feature_frame(metrics: dict) -> pd.DataFrame:
    """Turn a raw frontend metrics payload into the 15-column model input."""
    row = {name: 0 for name in FEATURE_ORDER}

    row["age"] = float(metrics.get("age", 0))
    row["hypertension"] = int(metrics.get("hypertension", 0))
    row["heart_disease"] = int(metrics.get("heart_disease", 0))
    row["bmi"] = float(metrics.get("bmi", 0))
    row["HbA1c_level"] = float(metrics.get("HbA1c_level", 0))
    row["blood_glucose_level"] = float(metrics.get("blood_glucose_level", 0))

    gender = str(metrics.get("gender", "Female"))
    gender_col = f"gender_{gender}"
    if gender_col in row:
        row[gender_col] = 1

    smoking = str(metrics.get("smoking_history", "No Info"))
    smoking_col = f"smoking_history_{smoking}"
    if smoking_col in row:
        row[smoking_col] = 1

    return pd.DataFrame([[row[name] for name in FEATURE_ORDER]], columns=FEATURE_ORDER)


def predict_probability(metrics: dict) -> float:
    """Return the model's probability of diabetes (class 1) for one patient."""
    if MODEL is None or SCALER is None:
        raise RuntimeError("Model artifacts are not loaded.")

    features = build_feature_frame(metrics)
    scaled = SCALER.transform(features)
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_ORDER)
    proba = MODEL.predict_proba(scaled_df)[0][POSITIVE_CLASS_INDEX]

    # Keep within (0, 1) for numerically stable downstream display.
    return float(min(0.999, max(0.001, proba)))


def risk_level_for(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


def confidence_for(prob: float) -> int:
    """How confident the classifier is in its decision (0-100)."""
    return int(round(max(prob, 1 - prob) * 100))


# --------------------------------------------------------------------------- #
# Recommendation / explanation generation (clinical templates)
# --------------------------------------------------------------------------- #
def build_recommendations(prob: float, metrics: dict) -> dict:
    """Generate explanation + dietary/exercise/medical guidance for a patient."""
    risk = risk_level_for(prob)
    pct = round(prob * 100)
    is_obese = float(metrics.get("bmi", 0)) >= 30
    is_smoker = str(metrics.get("smoking_history", "")) == "current"
    has_hypertension = int(metrics.get("hypertension", 0)) == 1
    hba1c = metrics.get("HbA1c_level", "?")
    glucose = metrics.get("blood_glucose_level", "?")

    if risk == "High":
        return {
            "explanation": (
                f"Your calculated risk metric of {pct}% indicates a strong probability of metabolic "
                f"dysfunction or diabetes. Your clinical inputs (HbA1c of {hba1c}% and Glucose of "
                f"{glucose} mg/dL) are primary contributors. Immediate diagnostic screening and guidance "
                "from a healthcare provider are strongly indicated."
            ),
            "dietary": [
                "Strictly limit simple carbohydrates, added processed sugars, and refined cereals to prevent rapid blood glucose spikes.",
                "Include non-starchy leafy green vegetables, high-quality lean proteins (poultry, tofu, fish), and healthy lipid sources (avocado, raw seeds).",
                "Adopt consistent macronutrient distributions throughout meals to avoid sudden postprandial insulin surges.",
            ],
            "exercise": [
                "Perform at least 150-180 minutes of moderate-intensity cardiorespiratory exercise (such as brisk walking or swimming) weekly.",
                "Incorporate strength or resistance training twice a week to promote skeletal muscle glucose utilization.",
                "Implement short 10-15 minute walks immediately after primary meals to help smooth glycemic curves.",
            ],
            "medical": [
                "Promptly schedule an appointment with a primary care physician or board-certified endocrinologist for a diagnostic oral glucose tolerance or HbA1c test.",
                "Initiate routine home blood glucose monitoring as recommended by your physician to track baseline variations.",
                "Review full clinical metrics including lipid panels and renal function, because high risk correlates with elevated cardiovascular risk.",
            ],
        }

    if risk == "Medium":
        return {
            "explanation": (
                f"Based on your physiological inputs, you are at a moderate risk level ({pct}%). This is often "
                "characterized as the pre-diabetic stage, meaning early, active lifestyle interventions are highly "
                "effective at halting progression or reversing insulin resistance."
            ),
            "dietary": [
                "Shift refined white grains towards complex, fiber-rich alternatives (such as steel-cut oats, quinoa, and brown rice).",
                "Implement mindful portion sizes and practice reading nutrition labels to identify hidden industrial sugars.",
                (
                    "Aim for a moderate caloric deficit of 300-500 kcal per day to target gradual fat loss."
                    if is_obese
                    else "Opt for water and herbal teas instead of sodas, sports drinks, or sweetened beverages."
                ),
            ],
            "exercise": [
                "Aim for a minimum of 150 minutes of moderate physical activity every week.",
                "Reduce sedentary screen-time by introducing brief standing and mobility intervals every 60 minutes.",
                "Engage in moderately strenuous activities like hiking, cycling, or recreational sports that keep your heart rate moderately elevated.",
            ],
            "medical": [
                "Engage in clinical screening of HbA1c and metabolic markers every 6 to 12 months to monitor lifestyle progression.",
                "Work with a certified care specialist or nutritionist to establish a structured, sustainable dietary plan.",
                (
                    "Maintain meticulous blood pressure management, as hypertension is a multiplicative cardiovascular factor in pre-diabetes."
                    if has_hypertension
                    else "Discuss preventative strategies during your standard annual physical examination."
                ),
            ],
        }

    return {
        "explanation": (
            f"Your calculated risk of {pct}% indicates healthy glycemic and metabolic function. "
            "Keep maintaining your health-promoting habits."
        ),
        "dietary": [
            "Continue prioritizing real, unprocessed foods including vegetables, healthy fats, and high-quality fiber.",
            "Regulate intake of highly processed snack items and fast foods.",
            "Support robust metabolic pathways by consuming adequate micronutrients and minerals, and staying hydrated.",
        ],
        "exercise": [
            "Maintain your current level of physical fitness, blending aerobic cardiorespiratory conditioning and strength exercises.",
            "Keep daily step counts high (aiming for 8,000 to 10,000 steps on average).",
            "Use active recreation as a natural coping mechanism for psychological stress, which can trigger cortisol release.",
        ],
        "medical": [
            "Maintain routine annual wellness examinations consisting of standard blood chemistry profiles.",
            (
                "Consider tobacco cessation coaching, as smoking drastically increases systemic inflammation and peripheral arterial risk."
                if is_smoker
                else "Review standard metabolic panels occasionally to ensure your HbA1c levels remain safely below 5.7%."
            ),
            "Track key vitals over time to ensure ongoing cardio-metabolic efficiency.",
        ],
    }


def build_impact_summary(orig_prob: float, sim_prob: float, modifications: dict, improvement: int) -> str:
    summary = (
        f"By adjusting your key biometrics (BMI towards {modifications.get('bmi')} kg/m², "
        f"HbA1c to {modifications.get('HbA1c_level')}%, and Blood Glucose to "
        f"{modifications.get('blood_glucose_level')} mg/dL), your estimated diabetes risk probability "
        f"transforms from {round(orig_prob * 100)}% down to {round(sim_prob * 100)}%. "
    )
    if improvement > 0:
        summary += (
            f"This marks an impressive {improvement}% overall reduction in risk! Implementing these "
            "physiological optimizations relieves significant metabolic pressure off your pancreas, enhances "
            "insulin receptor sensitivity, and lowers the long-term risk of arterial, renal, or ocular "
            "complications associated with runaway blood sugars."
        )
    elif improvement == 0:
        summary += (
            "Your risk level remained steady. Achieving marginal metric improvement still fosters important "
            "base-level metabolic equilibrium."
        )
    else:
        summary += (
            "These biometric changes increase expected metabolic risk factors. Focus on consistent, downward "
            "adjustments of glycated hemoglobin and fasting sugars under professional clinical supervision."
        )
    return summary


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok" if MODEL is not None else "degraded",
            "model_loaded": MODEL is not None,
            "model_type": type(MODEL).__name__ if MODEL is not None else None,
            "error": MODEL_LOAD_ERROR,
        }
    )


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return jsonify({"success": False, "error": "Prediction model is not available on the server."}), 503

    metrics = request.get_json(silent=True) or {}
    required = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    if any(not isinstance(metrics.get(key), (int, float)) for key in required):
        return jsonify({"success": False, "error": "Invalid physiological metrics provided in payload."}), 400

    try:
        prob = predict_probability(metrics)
        risk = risk_level_for(prob)
        recs = build_recommendations(prob, metrics)
        return jsonify(
            {
                "success": True,
                "prediction": 1 if prob >= 0.5 else 0,
                "probability": prob,
                "risk_level": risk,
                "confidence": confidence_for(prob),
                "recommendations": {
                    "dietary": recs["dietary"],
                    "exercise": recs["exercise"],
                    "medical": recs["medical"],
                },
                "explanation": recs["explanation"],
            }
        )
    except Exception as exc:
        app.logger.exception("Error in /api/predict")
        return jsonify({"success": False, "error": f"Internal error during evaluation: {exc}"}), 500


@app.post("/api/simulate")
def simulate():
    if MODEL is None:
        return jsonify({"success": False, "error": "Prediction model is not available on the server."}), 503

    payload = request.get_json(silent=True) or {}
    original_data = payload.get("original_data")
    modifications = payload.get("modifications")
    if not original_data or not modifications:
        return jsonify({"success": False, "error": "Missing original_data or modifications values."}), 400

    try:
        orig_prob = predict_probability(original_data)
        simulated_data = {**original_data, **modifications}
        sim_prob = predict_probability(simulated_data)

        improvement = 0
        if orig_prob > 0:
            improvement = round(((orig_prob - sim_prob) / orig_prob) * 100)
        improvement = max(-100, improvement)

        return jsonify(
            {
                "original_prediction": orig_prob,
                "simulated_prediction": sim_prob,
                "improvement_percentage": improvement,
                "impact_summary": build_impact_summary(orig_prob, sim_prob, modifications, improvement),
            }
        )
    except Exception as exc:
        app.logger.exception("Error in /api/simulate")
        return jsonify({"success": False, "error": f"Internal error during simulation: {exc}"}), 500


@app.post("/api/recommendations")
def recommendations():
    payload = request.get_json(silent=True) or {}
    risk_level = payload.get("risk_level", "Low")
    metrics = payload.get("health_metrics") or {
        "gender": "Female",
        "age": 40,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 24,
        "HbA1c_level": 5.5,
        "blood_glucose_level": 100,
    }
    mock_prob = 0.75 if risk_level == "High" else 0.45 if risk_level == "Medium" else 0.15
    recs = build_recommendations(mock_prob, metrics)
    return jsonify(
        {
            "dietary_recommendations": recs["dietary"],
            "exercise_recommendations": recs["exercise"],
            "medical_recommendations": recs["medical"],
        }
    )


if __name__ == "__main__":
    print(f"[DiaFlux] ML backend running on http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
