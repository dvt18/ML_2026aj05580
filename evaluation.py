"""Strict labeled-test CSV validation and prediction-only multiclass evaluation."""

from collections.abc import Sequence
import csv
import io
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET_COLUMN = "fault_type"
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_TEST_ROWS = 10_000


def read_test_csv(
    content: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_rows: int = MAX_TEST_ROWS,
) -> pd.DataFrame:
    """Read bounded UTF-8 CSV without silently mangling duplicate/broken headers."""
    if not content:
        raise ValueError("The CSV is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"CSV size exceeds the {max_bytes:,}-byte limit. Upload test data only.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV with UTF-8 encoding.") from error
    if "\x00" in text:
        raise ValueError("The CSV contains invalid null characters.")

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        if len(header) < 2 or any(not name or name != name.strip() for name in header):
            raise ValueError("CSV headers must be nonempty and have no surrounding whitespace.")
        if len(header) != len(set(header)):
            raise ValueError("Duplicate CSV column names are not allowed.")
        count = 0
        for row in reader:
            if not row:
                continue
            count += 1
            if len(row) != len(header):
                raise ValueError(f"CSV data row {count} has {len(row)} fields; expected {len(header)}.")
            if count > max_rows:
                raise ValueError(f"CSV has more than {max_rows:,} rows. Upload test data only.")
        if count == 0:
            raise ValueError("The CSV has headers but no test rows.")
    except (csv.Error, StopIteration) as error:
        raise ValueError("The CSV is malformed or has no header.") from error

    try:
        return pd.read_csv(io.StringIO(text), float_precision="round_trip")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ValueError("The CSV could not be parsed.") from error


def validate_test_data(
    data: pd.DataFrame,
    feature_names: Sequence[str],
    class_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric inputs in training order and known labels, without mutation."""
    if data.empty:
        raise ValueError("The test dataset contains no rows.")
    if len(data) > MAX_TEST_ROWS:
        raise ValueError(f"Upload no more than {MAX_TEST_ROWS:,} test rows.")
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    expected = list(feature_names) + [TARGET_COLUMN]
    missing = [name for name in expected if name not in data.columns]
    extra = [name for name in data.columns if name not in expected]
    if missing or extra:
        raise ValueError(f"CSV schema mismatch. Missing columns: {missing}; unexpected columns: {extra}.")

    X = data.loc[:, list(feature_names)].copy()
    if X.isna().any().any():
        raise ValueError("Input features contain missing values; supply complete test measurements.")
    for name in feature_names:
        if X[name].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Feature {name} must contain numbers, not Boolean values.")
        try:
            X[name] = pd.to_numeric(X[name], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"Feature {name} contains nonnumeric values.") from error
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Input features must contain finite numbers, not NaN or infinity.")

    y = data[TARGET_COLUMN].copy()
    if y.isna().any():
        raise ValueError("The target column contains missing labels.")
    unknown = sorted(set(map(str, y.unique())) - set(class_names))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}. Use the documented fault_type values.")
    y = y.astype(str)
    return X, y


def compute_metrics(
    y_true: Any,
    predictions: Any,
    probabilities: Any,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Compute six metrics using a fixed class universe and probability-based AUC.

    Precision/recall/F1 are macro averages over all trained classes, with zero
    assigned to undefined class terms. Seven-class macro OvR AUC is NaN if an
    uploaded test subset omits any class; it is never replaced with accuracy.
    """
    labels = list(class_names)
    actual = np.asarray(y_true)
    predicted = np.asarray(predictions)
    probability = np.asarray(probabilities, dtype=float)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("At least two distinct class labels are required.")
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) == 0 or len(predicted) != len(actual):
        raise ValueError("Targets and predictions must be equally sized, nonempty vectors.")
    if not set(actual).issubset(labels) or not set(predicted).issubset(labels):
        raise ValueError("Targets and predictions must belong to the trained classes.")
    if probability.shape != (len(actual), len(labels)):
        raise ValueError("Probability shape does not match test rows and class labels.")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1].")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")

    auc = float("nan")
    if set(actual) == set(labels):
        # Per-class binary AUC keeps probability-column ordering explicit.
        auc = float(np.mean([
            roc_auc_score(actual == label, probability[:, position])
            for position, label in enumerate(labels)
        ]))
    macro_options = {"labels": labels, "average": "macro", "zero_division": 0}
    mcc = 0.0 if len(set(actual)) == len(set(predicted)) == 1 else float(matthews_corrcoef(actual, predicted))
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": auc,
        "Precision": float(precision_score(actual, predicted, **macro_options)),
        "Recall": float(recall_score(actual, predicted, **macro_options)),
        "F1": float(f1_score(actual, predicted, **macro_options)),
        "MCC": mcc,
    }


def evaluate_model(model: Any, X: pd.DataFrame, y: Any, class_names: Sequence[str]) -> dict[str, Any]:
    """Use predict/predict_proba only; never train or modify an uploaded dataset."""
    labels = list(class_names)
    model_labels = list(model.classes_)
    if set(model_labels) != set(labels):
        raise ValueError("Saved model classes do not match the dataset schema.")
    predictions = np.asarray(model.predict(X))
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    probabilities = probabilities[:, [model_labels.index(label) for label in labels]]
    return {
        "metrics": compute_metrics(y, predictions, probabilities, labels),
        "predictions": predictions,
        "probabilities": probabilities,
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels),
        "classification_report": classification_report(
            y, predictions, labels=labels, target_names=labels, output_dict=True, zero_division=0
        ),
    }