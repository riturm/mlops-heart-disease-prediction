"""
tests/test_pipeline.py
-----------------------
Unit tests covering data processing + model behaviour (Task 5).
These run in CI; the pipeline must FAIL if any assertion fails.

Run:
    pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocess import (ALL_FEATURES, TARGET, build_preprocessor,  # noqa: E402
                        load_and_clean)


@pytest.fixture
def tiny_df(tmp_path):
    """Create a small synthetic CSV mimicking the raw schema, incl. a '?'."""
    data = {
        "age": [63, 67, 41], "sex": [1, 1, 0], "cp": [1, 4, 2],
        "trestbps": [145, 160, 130], "chol": [233, 286, 204],
        "fbs": [1, 0, 0], "restecg": [2, 2, 2], "thalach": [150, 108, 172],
        "exang": [0, 1, 0], "oldpeak": [2.3, 1.5, 1.4], "slope": [3, 2, 1],
        "ca": [0, 3, "?"], "thal": [6, 3, 3], "num": [0, 2, 0],
    }
    csv = tmp_path / "raw.csv"
    pd.DataFrame(data).to_csv(csv, index=False)
    return csv


def test_target_is_binary(tiny_df):
    df = load_and_clean(str(tiny_df))
    assert set(df[TARGET].unique()).issubset({0, 1})
    # num=2 must map to disease present (1)
    assert df[TARGET].tolist() == [0, 1, 0]


def test_question_marks_become_missing(tiny_df):
    df = load_and_clean(str(tiny_df))
    assert df["ca"].isna().sum() == 1


def test_preprocessor_outputs_numeric_matrix(tiny_df):
    df = load_and_clean(str(tiny_df))
    X = df[ALL_FEATURES]
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(df)
    assert np.isfinite(Xt.astype(float)).all()  # no NaNs after imputation
