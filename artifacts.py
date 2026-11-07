"""Load only the project's trusted, checksum-verified fitted pipelines."""

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import sklearn

from evaluation import MODEL_FILES


def load_artifacts(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject incompatible or changed files before using deployment artifacts.

    A checksum detects accidental changes, not malicious repository ownership.
    The repository itself must be trusted: joblib is not an untrusted-file format.
    """
    model_directory = Path(project_root) / "model"
    schema = json.loads((model_directory / "schema.json").read_text())
    if schema.get("model_files") != MODEL_FILES:
        raise ValueError("The saved model registry does not match the fixed application registry.")
    expected_version = schema["versions"]["scikit-learn"]
    if sklearn.__version__ != expected_version:
        raise ValueError(
            f"Saved models require scikit-learn {expected_version}; installed {sklearn.__version__}. "
            "Install the pinned dependencies or retrain in the intended environment."
        )
    if len(schema["feature_names"]) != 27 or len(schema["class_names"]) != 7:
        raise ValueError("Saved feature/class schema is inconsistent with this dataset.")
    if schema.get("cv_selected_model") not in MODEL_FILES:
        raise ValueError("The saved CV-selected model is missing or is not one of the five classifiers.")

    # Validate all hashes before deserializing even the first model.
    for filename in MODEL_FILES.values():
        actual = hashlib.sha256((model_directory / filename).read_bytes()).hexdigest()
        if actual != schema["model_sha256"].get(filename):
            raise ValueError(f"Model checksum mismatch for {filename}; regenerate or restore trusted artifacts.")

    models = {}
    for name, filename in MODEL_FILES.items():
        model = joblib.load(model_directory / filename)
        if list(model.feature_names_in_) != schema["feature_names"]:
            raise ValueError(f"Feature order mismatch in {name}.")
        if list(model.classes_) != schema["class_names"]:
            raise ValueError(f"Class order mismatch in {name}.")
        models[name] = model
    return schema, models