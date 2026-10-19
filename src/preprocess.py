"""
preprocess.py
-------------
Reusable preprocessing logic so that TRAIN and SERVE use the *exact* same
transformations (this is the key to reproducibility in MLOps).

Key dataset quirks handled here:
  * `ca` and `thal` sometimes contain '?' -> treat as missing.
  * target `num` is 0-4 -> binarise to 0 (no disease) / 1 (disease present).
  * numeric vs categorical columns are scaled / one-hot encoded separately.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_RAW = "num"
TARGET = "target"

# Clinically, these behave as continuous values.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
# These are categorical codes (even though stored as numbers).
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load raw CSV, coerce '?' to NaN, and binarise the target."""
    df = pd.read_csv(csv_path)
    df = df.replace("?", pd.NA)

    # ca / thal may arrive as strings because of '?'; force numeric.
    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarise: 0 -> 0 (absence), {1,2,3,4} -> 1 (presence).
    df[TARGET] = (pd.to_numeric(df[TARGET_RAW], errors="coerce") > 0).astype(int)
    df = df.drop(columns=[TARGET_RAW])
    return df


def build_preprocessor() -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer (fit only on training data)."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def split_X_y(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y
