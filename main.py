"""End-to-end Amazon review sentiment classification."""

import os
import re

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_and_clean_data
from src.evaluation import plot_confusion_matrix, plot_model_comparison
from src.models import get_models, train_and_evaluate
from src.preprocessing import build_pipeline


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "results", "models")
    figures_dir = os.path.join(base, "results", "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = load_and_clean_data()
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = get_models()
    pipelines = {name: build_pipeline(clf) for name, clf in models.items()}

    results = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)

    label_names = ["Negative", "Positive"]

    for name, pipe in pipelines.items():
        safe = re.sub(r"[^\w\-]+", "_", name).strip("_").lower()
        joblib.dump(pipe, os.path.join(models_dir, f"{safe}_pipeline.joblib"))

    for name in pipelines:
        safe = re.sub(r"[^\w\-]+", "_", name).strip("_").lower()
        plot_confusion_matrix(
            y_test,
            results[name]["y_pred"],
            name,
            label_names,
            os.path.join(figures_dir, f"{safe}_confusion_matrix.png"),
        )

    plot_model_comparison(
        results,
        os.path.join(figures_dir, "model_comparison.png"),
    )

    print("\n" + "=" * 50)
    print("Final summary (model | accuracy)")
    print("=" * 50)
    summary = pd.DataFrame(
        [{"model": n, "accuracy": results[n]["accuracy"]} for n in results]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
