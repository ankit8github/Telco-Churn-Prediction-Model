"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn
prediction model with STRICT training/serving consistency.

Key guarantees:
- Deterministic model loading (no silent fallbacks in Docker)
- Exact feature transformation parity with training
- Explicit failure if model or schema is missing
"""

import os
import pandas as pd
import mlflow

# ============================================================
# MODEL LOADING (FAIL FAST – PRODUCTION SAFE)
# ============================================================

# SINGLE source of truth for production
MODEL_DIR = "/app/model"

if not os.path.exists(MODEL_DIR):
    raise RuntimeError(
        f"❌ Model directory not found at {MODEL_DIR}.\n"
        "Docker image is invalid.\n"
        "Fix: Export model to ./model and COPY it into the image."
    )

try:
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    raise RuntimeError(
        f"❌ Failed to load MLflow model from {MODEL_DIR}: {e}"
    )

# ============================================================
# FEATURE SCHEMA LOADING (STRICT ORDER GUARANTEE)
# ============================================================

FEATURE_COLS = None

feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")

if os.path.exists(feature_file):
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns")
else:
    raise RuntimeError(
        f"❌ feature_columns.txt not found in {MODEL_DIR}.\n"
        "Model artifacts are incomplete."
    )

# ============================================================
# FEATURE TRANSFORMATION CONSTANTS
# ============================================================

BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ============================================================
# TRANSFORMATION PIPELINE (TRAIN/SERVE CONSISTENT)
# ============================================================

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Numeric coercion
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Binary encoding
    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # One-hot encode remaining categoricals
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Boolean → int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align with training schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

# ============================================================
# PREDICTION FUNCTION (SAFE & DETERMINISTIC)
# ============================================================

def predict(input_dict: dict) -> str:
    """
    Perform churn prediction on a single customer record.
    """

    if model is None:
        raise RuntimeError("Model is not loaded – inference aborted.")

    # Convert input → DataFrame
    df = pd.DataFrame([input_dict])

    # Transform features
    df_enc = _serve_transform(df)

    # Predict
    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0]
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"
