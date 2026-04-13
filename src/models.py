"""Model definitions and training / evaluation helpers."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC


def get_models():
    """Return three baseline classifiers keyed by display name."""
    return {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear SVM": LinearSVC(max_iter=2000, C=1.0),
    }


def get_models_ternary():
    """Same three models configured for multi-class (ternary) sentiment."""
    return {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, multi_class="auto"
        ),
        "Linear SVM": LinearSVC(max_iter=2000, C=1.0),
    }


def get_models_balanced():
    """
    Ternary models with ``class_weight='balanced'`` where sklearn allows it.

    ``ComplementNB`` has no ``class_weight`` argument; the instance is unchanged
    from the default (same as in :func:`get_models_ternary`).
    """
    return {
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            multi_class="auto",
            class_weight="balanced",
        ),
        "Linear SVM": LinearSVC(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
        ),
    }


def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test):
    """
    Fit each pipeline, predict on test set, print metrics.

    Returns
    -------
    dict
        model_name -> {"accuracy", "report", "y_pred"}
    """
    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(report)
        results[name] = {
            "accuracy": float(acc),
            "report": report,
            "y_pred": y_pred,
        }
    return results
