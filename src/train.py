"""
train.py
--------
Train Logistic Regression and Random Forest inside a single sklearn Pipeline
(preprocessor + model), evaluate with cross-validation + a held-out test set,
log everything to MLflow, and persist the BEST model to models/model.joblib.

Run:
    python src/train.py
"""
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, load_and_clean, split_X_y

MODEL_DIR = Path("models")
RANDOM_STATE = 42


def evaluate(model, X_test, y_test) -> dict:
    """Compute the metrics required by the rubric on the held-out test set."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_one(name, estimator, param_grid, X_train, y_train, X_test, y_test):
    """Build pipeline, tune with GridSearchCV, log to MLflow, return fitted model + metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])

    with mlflow.start_run(run_name=name):
        search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # 5-fold CV accuracy on the training split (for the report).
        cv_acc = cross_val_score(best, X_train, y_train, cv=5, scoring="accuracy")

        metrics = evaluate(best, X_test, y_test)
        metrics["cv_accuracy_mean"] = float(np.mean(cv_acc))

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # Save the fitted pipeline and log it as an artifact (version-robust:
        # works across all MLflow versions, no flavor/serialization issues).
        model_file = MODEL_DIR / f"{name}.joblib"
        joblib.dump(best, model_file)
        mlflow.log_artifact(str(model_file), artifact_path="model")

        # Log EDA plots if generated (run src/eda.py first).
        fig_dir = Path("reports/figures")
        if fig_dir.exists():
            mlflow.log_artifacts(str(fig_dir), artifact_path="eda_figures")

        print(f"[train] {name}: {metrics}")
        return best, metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("heart-disease-classification")

    df = load_and_clean("data/heart_raw.csv")
    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # ---- Model 1: Logistic Regression ----
    lr_grid = {"model__C": [0.1, 1.0, 10.0], "model__max_iter": [1000]}
    lr_model, results["logreg"] = train_one(
        "logistic_regression",
        LogisticRegression(random_state=RANDOM_STATE),
        lr_grid, X_train, y_train, X_test, y_test,
    )

    # ---- Model 2: Random Forest ----
    rf_grid = {"model__n_estimators": [100, 300], "model__max_depth": [None, 5, 10]}
    rf_model, results["rf"] = train_one(
        "random_forest",
        RandomForestClassifier(random_state=RANDOM_STATE),
        rf_grid, X_train, y_train, X_test, y_test,
    )

    # ---- Select best model by ROC-AUC and persist ----
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = lr_model if best_name == "logreg" else rf_model
    joblib.dump(best_model, MODEL_DIR / "model.joblib")
    print(f"[train] Best model = {best_name} -> saved to {MODEL_DIR/'model.joblib'}")


if __name__ == "__main__":
    main()
