"""Plotting helpers for confusion matrices and model comparison."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, model_name, label_names, save_path):
    """Seaborn heatmap of confusion matrix; saved at dpi=150."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_normalized(
    y_true, y_pred, model_name, label_names, save_path
):
    """Row-normalized confusion matrix (true-class percentages); dpi=150."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    annot = np.array([[f"{v:.1%}" for v in row] for row in cm])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"Normalized Confusion Matrix — {model_name}")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, save_path):
    """Horizontal bar chart of accuracies; x-axis 0.5–1.0; dpi=150."""
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, accs, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Accuracy")
    ax.set_title("Model comparison (accuracy)")
    ax.set_xlim(0.5, 1.0)
    for bar, acc in zip(bars, accs):
        ax.text(
            acc + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}",
            va="center",
            fontsize=10,
        )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(y, label_names, save_path, title="Class distribution"):
    """Bar chart of class counts; dpi=150."""
    s = pd.Series(y).value_counts().reindex(
        range(len(label_names)), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(label_names, s.values, color="steelblue")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate(s.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _multiclass_score_matrix(pipe, X):
    """Probability or decision scores, shape (n_samples, n_classes)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    return pipe.decision_function(X)


def plot_roc_curves(pipelines, X_test, y_test, label_names, save_path):
    """
    One-vs-rest ROC per class for each fitted pipeline; one subplot per model.
    """
    classes = list(range(len(label_names)))
    y_true = np.asarray(y_test)
    y_bin = label_binarize(y_true, classes=classes)
    n_models = len(pipelines)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), squeeze=False)
    axes = axes[0]

    for ax, (model_name, pipe) in zip(axes, pipelines.items()):
        scores = _multiclass_score_matrix(pipe, X_test)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        for k, cname in enumerate(label_names):
            fpr, tpr, _ = roc_curve(y_bin[:, k], scores[:, k])
            auc_k = roc_auc_score(y_bin[:, k], scores[:, k])
            ax.plot(fpr, tpr, lw=2, label=f"{cname} (AUC={auc_k:.3f})")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(model_name)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def error_analysis(
    df_test, y_true, y_pred, model_name, n_samples=10, save_path=None
):
    """
    Misclassification summary, neutral-focused examples, length stats, optional bar chart.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    df = pd.DataFrame(
        {
            "text": df_test["text"].values,
            "rating": df_test["rating"].values,
            "true_label": y_true,
            "pred_label": y_pred,
        }
    )

    n = len(df)
    wrong = df[df["true_label"] != df["pred_label"]]
    n_wrong = len(wrong)
    rate = n_wrong / n if n else 0.0

    print(f"\n=== Error analysis — {model_name} ===")
    print(f"Total misclassifications: {n_wrong} / {n} ({rate:.2%})")

    label_names = ["Negative", "Neutral", "Positive"]
    breakdown = pd.crosstab(
        df["true_label"],
        df["pred_label"],
        rownames=["True"],
        colnames=["Pred"],
        margins=False,
    )
    print("\nPrediction counts by true class (rows=true, cols=pred):")
    print(breakdown.to_string())

    neutral_pos = wrong[
        (wrong["true_label"] == 1) & (wrong["pred_label"] == 2)
    ]
    neutral_neg = wrong[
        (wrong["true_label"] == 1) & (wrong["pred_label"] == 0)
    ]

    def _truncate(s, max_len=200):
        s = str(s)
        return s if len(s) <= max_len else s[:200] + "..."

    print(
        f"\nNeutral → Positive errors (showing up to {n_samples} examples):"
    )
    for _, row in neutral_pos.head(n_samples).iterrows():
        print(f"  rating={row['rating']!r} | {_truncate(row['text'])}")

    print(
        f"\nNeutral → Negative errors (showing up to {n_samples} examples):"
    )
    for _, row in neutral_neg.head(n_samples).iterrows():
        print(f"  rating={row['rating']!r} | {_truncate(row['text'])}")

    neutral_correct = df[
        (df["true_label"] == 1) & (df["pred_label"] == 1)
    ]
    neutral_wrong = df[
        (df["true_label"] == 1) & (df["pred_label"] != 1)
    ]

    def _avg_word_count(frame):
        if frame.empty:
            return float("nan")
        lengths = frame["text"].astype(str).str.split().str.len()
        return float(lengths.mean())

    avg_ok = _avg_word_count(neutral_correct)
    avg_bad = _avg_word_count(neutral_wrong)
    print("\nAverage word count (Neutral reviews):")
    print(f"  Correctly classified: {avg_ok:.2f}")
    print(f"  Misclassified:        {avg_bad:.2f}")

    if save_path is not None:
        pairs = []
        counts = []
        for t in range(3):
            for p in range(3):
                if t == p:
                    continue
                c = int(((y_true == t) & (y_pred == p)).sum())
                pairs.append(f"{label_names[t]} → {label_names[p]}")
                counts.append(c)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(pairs)), counts, color="coral")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, rotation=35, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"Misclassification breakdown — {model_name}")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
