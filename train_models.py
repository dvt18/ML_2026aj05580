"""Train, evaluate and save the five classifiers for UCI Steel Plates Faults.

Run from this folder:  python train_models.py
Outputs: model/*.joblib, model/schema.json, test_data.csv, results/metrics_comparison.csv
The Streamlit app only loads these saved pipelines; it never retrains.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import METRIC_NAMES, MODEL_FILES, TARGET_COLUMN, evaluate_model

SEED = 42
SOURCE_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
ARCHIVE_SHA256 = "cb8eb9859198b63f053e443513036b401746fa517ef58bd17c846c6741c93919"


def load_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    archive_path = data_dir / "steel_plates_faults.zip"
    if archive_path.is_file():
        archive_bytes = archive_path.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            archive_bytes = response.read()
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(archive_bytes)
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Dataset archive differs from the verified UCI source.")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        columns = archive.read("Faults27x7_var").decode("utf-8").splitlines()
        raw = pd.read_csv(io.BytesIO(archive.read("Faults.NNA")), sep=r"\s+", header=None, names=columns)
    features, indicators = columns[:27], columns[27:]
    if raw.shape != (1941, 34) or (raw[indicators].sum(axis=1) != 1).any():
        raise ValueError("Unexpected dataset structure.")
    X = raw[features].copy()
    y = raw[indicators].idxmax(axis=1).rename(TARGET_COLUMN)  # one-hot indicators -> one label
    return X, y, features, sorted(y.unique().tolist())


def model_specifications() -> dict:
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=10000, random_state=SEED))]),
            {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0], "classifier__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            Pipeline([("classifier", DecisionTreeClassifier(random_state=SEED))]),
            {"classifier__criterion": ["gini", "entropy"], "classifier__max_depth": [4, 8, 12, None],
             "classifier__min_samples_leaf": [1, 3, 5], "classifier__class_weight": [None, "balanced"]},
        ),
        "KNN": (
            Pipeline([("scale", StandardScaler()), ("classifier", KNeighborsClassifier())]),
            {"classifier__n_neighbors": [3, 5, 9, 15, 25], "classifier__weights": ["uniform", "distance"], "classifier__p": [1, 2]},
        ),
        "Naive Bayes": (
            Pipeline([("scale", StandardScaler()), ("classifier", GaussianNB())]),
            {"classifier__var_smoothing": np.logspace(-11, -5, 7).tolist()},
        ),
        "Support Vector Machine": (
            Pipeline([("scale", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=SEED, cache_size=256))]),
            {"classifier__C": [0.1, 1.0, 10.0, 100.0], "classifier__gamma": ["scale", 0.01, 0.1], "classifier__class_weight": [None, "balanced"]},
        ),
    }


def main(output: Path) -> pd.DataFrame:
    project = Path(__file__).resolve().parent
    X, y, features, classes = load_dataset(project / "data")

    # Split once; the test rows are never used for fitting or tuning.
    train_ids, test_ids = train_test_split(np.arange(len(X)), test_size=0.20, stratify=y, random_state=SEED)
    X_train, X_test, y_train, y_test = X.iloc[train_ids], X.iloc[test_ids], y.iloc[train_ids], y.iloc[test_ids]

    (output / "model").mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(parents=True, exist_ok=True)
    test_path = output / "test_data.csv"
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(test_path, index=False, float_format="%.17g")

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train))
    scorer = make_scorer(f1_score, labels=classes, average="macro", zero_division=0)
    metrics, cv_scores, hashes = {}, {}, {}
    for name, (pipeline, grid) in model_specifications().items():
        search = GridSearchCV(clone(pipeline), grid, scoring=scorer, cv=folds, refit=True, n_jobs=1, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.filterwarnings("ignore", message=r"The `probability` parameter was deprecated", category=FutureWarning)
            search.fit(X_train, y_train)
        model = search.best_estimator_
        result = evaluate_model(model, X_test, y_test, classes)
        metrics[name], cv_scores[name] = result["metrics"], float(search.best_score_)
        destination = output / "model" / MODEL_FILES[name]
        joblib.dump(model, destination, compress=3)
        hashes[MODEL_FILES[name]] = hashlib.sha256(destination.read_bytes()).hexdigest()
        print(f"{name}: {search.best_params_} | CV macro F1 {search.best_score_:.4f} | test accuracy {result['metrics']['Accuracy']:.4f}")

    comparison = pd.DataFrame(metrics).T[METRIC_NAMES]
    comparison.index.name = "ML Model Name"
    comparison.to_csv(output / "results" / "metrics_comparison.csv", float_format="%.12g")
    schema = {
        "dataset": "UCI Steel Plates Faults", "source_url": SOURCE_URL,
        "feature_names": features, "target_column": TARGET_COLUMN, "class_names": classes,
        "train_rows": len(train_ids), "test_rows": len(test_ids), "random_state": SEED,
        "test_data_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "versions": {"scikit-learn": sklearn.__version__},
        "model_files": MODEL_FILES, "model_sha256": hashes,
        "cv_selected_model": max(cv_scores, key=cv_scores.get),
        "observed_test_winner": comparison["F1"].idxmax(),
    }
    (output / "model" / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(comparison.round(4).to_string())
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent, help="Folder for models, test CSV and results.")
    main(parser.parse_args().output)
