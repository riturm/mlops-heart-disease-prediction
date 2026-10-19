"""
eda.py
------
Generate the EDA visualisations required by Task 1:
  * histograms of numeric features
  * correlation heatmap
  * class-balance bar chart

Saves PNGs to reports/figures/ so they can be embedded in the report and
logged as MLflow artifacts.

NOTE: These are starter visuals. Personalise them (palette, extra plots like
boxplots by target, pairplots) so your EDA is your own, as the rubric asks.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import NUMERIC_FEATURES, TARGET, load_and_clean

FIG_DIR = Path("reports/figures")


def run_eda(csv_path: str = "data/heart_raw.csv") -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(csv_path)
    sns.set_theme(style="whitegrid")

    # 1) Histograms of numeric features
    df[NUMERIC_FEATURES].hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribution of Numeric Features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "histograms.png", dpi=150)
    plt.close()

    # 2) Correlation heatmap (numeric columns + target)
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 3) Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df)
    plt.title("Class Balance (0 = no disease, 1 = disease)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_balance.png", dpi=150)
    plt.close()

    print(f"[eda] Figures saved to {FIG_DIR}/")
    print("[eda] Class counts:\n", df[TARGET].value_counts())


if __name__ == "__main__":
    run_eda()
