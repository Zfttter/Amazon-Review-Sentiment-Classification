"""Ternary sentiment classification, ROC (LR), and error analysis."""

import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_data_ternary
from src.evaluation import (
    error_analysis,
    plot_class_distribution,
    plot_confusion_matrix_normalized,
    plot_roc_curves,
)
from src.models import get_models_balanced, get_models_ternary, train_and_evaluate
from src.preprocessing import build_pipeline

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
NEUTRAL_CLASS = 1
F1_LABELS = [0, 1, 2]


def _neutral_f1(y_true, y_pred):
    per_class = f1_score(
        y_true, y_pred, labels=F1_LABELS, average=None, zero_division=0
    )
    return float(per_class[NEUTRAL_CLASS])


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "results", "models")
    figures_dir = os.path.join(base, "results", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = load_data_ternary()
    print(
        "\nNote: Neutral class is imbalanced — consider this when "
        "interpreting accuracy.\n"
    )

    plot_class_distribution(
        df["label"],
        LABEL_NAMES,
        os.path.join(figures_dir, "ternary_class_distribution.png"),
        title="Ternary class distribution",
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    X_train = train_df["text"]
    X_test = test_df["text"]
    y_train = train_df["label"]
    y_test = test_df["label"]

    models = get_models_ternary()
    pipelines = {name: build_pipeline(clf) for name, clf in models.items()}

    results = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)

    print("\n--- Class-weight comparison (Neutral F1) ---")
    balanced_clfs = get_models_balanced()
    pipelines_bal = {
        name: build_pipeline(clf) for name, clf in balanced_clfs.items()
    }
    neutral_f1_unbal = {}
    neutral_f1_bal = {}
    for name in pipelines:
        neutral_f1_unbal[name] = _neutral_f1(y_test, results[name]["y_pred"])
        pipe_b = pipelines_bal[name]
        pipe_b.fit(X_train, y_train)
        y_pb = pipe_b.predict(X_test)
        neutral_f1_bal[name] = _neutral_f1(y_test, y_pb)
        print(
            f"{name}: Neutral F1 unbalanced={neutral_f1_unbal[name]:.4f} | "
            f"balanced={neutral_f1_bal[name]:.4f}"
        )

    model_names = list(pipelines.keys())
    x = np.arange(len(model_names))
    width = 0.35
    unbal_vals = [neutral_f1_unbal[n] for n in model_names]
    bal_vals = [neutral_f1_bal[n] for n in model_names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        unbal_vals,
        width,
        label="Default weights",
        color="steelblue",
    )
    ax.bar(
        x + width / 2,
        bal_vals,
        width,
        label="class_weight='balanced'",
        color="darkorange",
    )
    ax.set_ylabel("Neutral class F1")
    ax.set_title("Neutral F1: balanced vs unbalanced class weights")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=12, ha="right")
    ax.set_ylim(0, max(1.0, max(unbal_vals + bal_vals) * 1.15))
    ax.legend(loc="upper right")
    for i, n in enumerate(model_names):
        ax.text(
            i - width / 2,
            unbal_vals[i] + 0.02,
            f"{unbal_vals[i]:.3f}",
            ha="center",
            fontsize=8,
        )
        ax.text(
            i + width / 2,
            bal_vals[i] + 0.02,
            f"{bal_vals[i]:.3f}",
            ha="center",
            fontsize=8,
        )
    plt.tight_layout()
    cmp_path = os.path.join(
        figures_dir, "ternary_neutral_f1_balanced_vs_unbalanced.png"
    )
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print(f"Saved Neutral F1 comparison figure to {cmp_path}\n")

    for name, pipe in pipelines.items():
        safe = re.sub(r"[^\w\-]+", "_", name).strip("_").lower()
        joblib.dump(
            pipe,
            os.path.join(models_dir, f"ternary_{safe}_pipeline.joblib"),
        )

    for name in pipelines:
        safe = re.sub(r"[^\w\-]+", "_", name).strip("_").lower()
        plot_confusion_matrix_normalized(
            y_test,
            results[name]["y_pred"],
            name,
            LABEL_NAMES,
            os.path.join(figures_dir, f"ternary_cm_{safe}.png"),
        )

    lr_only = {"Logistic Regression": pipelines["Logistic Regression"]}
    plot_roc_curves(
        lr_only,
        X_test,
        y_test,
        LABEL_NAMES,
        os.path.join(figures_dir, "ternary_roc_lr.png"),
    )

    error_analysis(
        test_df,
        y_test,
        results["Logistic Regression"]["y_pred"],
        "Logistic Regression",
        n_samples=10,
        save_path=os.path.join(figures_dir, "ternary_error_breakdown_lr.png"),
    )

    print("\n" + "=" * 60)
    print("Final summary (model | accuracy | macro F1)")
    print("=" * 60)
    rows = []
    for name in pipelines:
        y_p = results[name]["y_pred"]
        macro_f1 = f1_score(y_test, y_p, average="macro")
        rows.append(
            {
                "model": name,
                "accuracy": results[name]["accuracy"],
                "macro_f1": macro_f1,
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
