"""
app.py  – Flask REST backend for House Rent Prediction
Endpoints:
  GET  /api/options   -> dropdown options for the UI
  POST /api/predict   -> accepts JSON, returns predicted rent
  GET  /api/health    -> simple health-check
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH    = os.path.join("models", "rent_model.pkl")
ENCODERS_PATH = os.path.join("models", "encoders.pkl")

app = Flask(__name__)
CORS(app)   # Allow Streamlit frontend on a different port

# ---------- load artefacts at startup ----------

def load_artefacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run  python train_model.py  first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        data = pickle.load(f)
    return model, data["encoders"], data["feature_cols"]

MODEL, ENCODERS, FEATURE_COLS = load_artefacts()

# ---------- static option lists (from training data) ----------

OPTIONS = {
    "area_type":        ["Super Area", "Carpet Area", "Built Area"],
    "city":             ["Kolkata", "Mumbai", "Bangalore", "Delhi", "Chennai", "Hyderabad"],
    "furnishing_status":["Unfurnished", "Semi-Furnished", "Furnished"],
    "tenant_preferred": ["Bachelors/Family", "Bachelors", "Family"],
    "floor_type":       ["Ground", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                         "Upper Basement", "Lower Basement"],
}

# ---------- helpers ----------

def parse_floor_input(floor_num_str: str, total_floors: int):
    """Convert user-supplied floor string + total to (floor_num, total_floors)."""
    fstr = str(floor_num_str).strip().lower()
    if fstr == "ground":
        num = 0
    elif "basement" in fstr:
        num = -1
    else:
        try:
            num = int(fstr)
        except ValueError:
            num = 0
    return num, int(total_floors)


def build_feature_row(payload: dict, encoders: dict) -> pd.DataFrame:
    floor_num, total_floors = parse_floor_input(
        payload.get("floor_num", "Ground"),
        payload.get("total_floors", 2),
    )
    row = {
        "BHK":               int(payload["bhk"]),
        "Size":              int(payload["size"]),
        "Floor_Num":         floor_num,
        "Total_Floors":      total_floors,
        "Area Type":         payload["area_type"],
        "City":              payload["city"],
        "Furnishing Status": payload["furnishing_status"],
        "Tenant Preferred":  payload["tenant_preferred"],
        "Bathroom":          int(payload["bathroom"]),
    }
    df = pd.DataFrame([row])
    cat_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
    for col in cat_cols:
        le = encoders[col]
        # Handle unseen labels gracefully
        val = df[col].astype(str).iloc[0]
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])
    return df[FEATURE_COLS]


# ---------- routes ----------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})


@app.route("/api/options", methods=["GET"])
def options():
    return jsonify(OPTIONS)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        required = ["bhk", "size", "area_type", "city",
                    "furnishing_status", "tenant_preferred", "bathroom"]
        missing = [k for k in required if k not in payload]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        X = build_feature_row(payload, ENCODERS)
        log_pred = MODEL.predict(X)[0]
        rent = float(np.expm1(log_pred))

        # Confidence band ±10 %
        low  = round(rent * 0.90, -2)
        mid  = round(rent, -2)
        high = round(rent * 1.10, -2)

        return jsonify({
            "predicted_rent": mid,
            "range_low":      low,
            "range_high":     high,
            "currency":       "INR",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
