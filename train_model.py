"""
train_model.py
Preprocesses House_Rent_Dataset.csv, trains a RandomForest regressor,
and saves the fitted pipeline + label encoders to models/.
Run once before starting the app:  python train_model.py
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH   = os.path.join("data", "House_Rent_Dataset.csv")
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "rent_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

# ---------- helpers ----------

def parse_floor(floor_str):
    """
    Converts e.g. '3 out of 5' -> floor_number=3, total_floors=5
                  'Ground out of 2' -> 0, 2
                  'Upper Basement out of 2' -> -1, 2
    """
    floor_str = str(floor_str).strip().lower()
    parts = floor_str.split("out of")
    total = int(parts[1].strip()) if len(parts) == 2 else 1
    raw = parts[0].strip()
    if raw == "ground":
        num = 0
    elif "basement" in raw:
        num = -1
    else:
        try:
            num = int(raw)
        except ValueError:
            num = 0
    return num, total


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop rows with missing critical values
    df.dropna(subset=["Rent", "Size", "BHK", "Bathroom"], inplace=True)

    # Parse Floor into numeric features
    df[["Floor_Num", "Total_Floors"]] = df["Floor"].apply(
        lambda x: pd.Series(parse_floor(x))
    )

    # Drop columns not needed for modelling
    df.drop(columns=["Posted On", "Floor", "Area Locality", "Point of Contact"], inplace=True)

    # Encode categorical columns
    cat_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
    return df, cat_cols


def encode_categoricals(df, cat_cols, encoders=None, fit=True):
    """Label-encode each categorical column. Returns (df, encoders_dict)."""
    if encoders is None:
        encoders = {}
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df, encoders


def main():
    print("Loading dataset …")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df_raw.shape}")

    df, cat_cols = preprocess(df_raw)
    df, encoders = encode_categoricals(df, cat_cols, fit=True)

    feature_cols = ["BHK", "Size", "Floor_Num", "Total_Floors",
                    "Area Type", "City", "Furnishing Status",
                    "Tenant Preferred", "Bathroom"]
    target_col = "Rent"

    X = df[feature_cols]
    y = np.log1p(df[target_col])          # log-transform target → better fit

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest …")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    print(f"\n  Test MAE  : Rs.{mae:,.0f}")
    print(f"  Test RMSE : Rs.{rmse:,.0f}")
    print(f"  Test R2   : {r2:.4f}")

    # Feature importances
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  Feature Importances:")
    print(fi.to_string())

    # Save artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump({"encoders": encoders, "feature_cols": feature_cols}, f)

    print(f"\nModel saved  -> {MODEL_PATH}")
    print(f"Encoders saved -> {ENCODERS_PATH}")


if __name__ == "__main__":
    main()
