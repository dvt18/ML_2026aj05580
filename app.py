"""Streamlit: evaluate five saved steel-fault classifiers without retraining."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st

from artifacts import load_artifacts
from evaluation import (
    MAX_TEST_ROWS,
    METRIC_NAMES,
    MODEL_FILES,
    evaluate_model,
    read_test_csv,
    validate_test_data,
)


ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Loading verified model pipelines...")
def cached_artifacts(schema_fingerprint: str):
    """Schema contents invalidate the cache whenever trained artifacts change."""
    return load_artifacts(ROOT)


def main():
    st.set_page_config(page_title="Surface Fault Lab", layout="wide")
    st.caption("STEEL QUALITY / MACHINE LEARNING ASSIGNMENT 2")
    st.title("Surface Fault Lab")
    st.write("Compare five classifiers on labeled test observations. Every result below is computed from a saved model — never from retraining your upload.")

    try:
        schema_bytes = (ROOT / "model/schema.json").read_bytes()
        schema, models = cached_artifacts(hashlib.sha256(schema_bytes).hexdigest())
        sample_bytes = (ROOT / "test_data.csv").read_bytes()
        if hashlib.sha256(sample_bytes).hexdigest() != schema["test_data_sha256"]:
            raise ValueError("The bundled test CSV differs from the verified held-out dataset.")
    except Exception as error:
        st.error(f"The saved project artifacts could not be loaded: {error}")
        st.info("Run the training notebook and commit all five pipelines, their schema and the test CSV with the pinned dependencies.")
        st.stop()

    with st.sidebar:
        st.header("Evaluation controls")
        source = st.radio("Test data source", ["Bundled held-out test set", "Upload test CSV"], key="data_source")
        uploaded = st.file_uploader(
            "Upload a labeled test CSV", type=["csv"], accept_multiple_files=False,
            max_upload_size=5, disabled=source != "Upload test CSV", key="test_csv",
            help="Include all 27 numeric features and fault_type. Maximum 5 MB / 10,000 rows. Upload test data only.",
        )
        selected = st.selectbox(
            "Classifier", list(MODEL_FILES),
            index=list(MODEL_FILES).index(schema["cv_selected_model"]), key="model_selection",
            help="The default was selected using training-only cross-validation, not the held-out test set.",
        )
        st.download_button("Download the labeled test CSV", sample_bytes, file_name="test_data.csv", mime="text/csv")
        st.divider()
        st.caption("27 inputs · 7 fault classes · 5 saved models")
        st.caption(f"Training rows: {schema['train_rows']:,} · original held-out rows: {schema['test_rows']:,}")
        st.caption(f"Upload limit: {MAX_TEST_ROWS:,} rows. Files are processed in memory; the application does not write uploads to disk.")

    if source == "Upload test CSV" and uploaded is None:
        st.info("Choose a test CSV in the sidebar. No sample-data results are substituted in upload mode.")
        st.stop()
    csv_bytes = uploaded.getvalue() if source == "Upload test CSV" else sample_bytes
    try:
        frame = read_test_csv(csv_bytes)
        X, y = validate_test_data(frame, schema["feature_names"], schema["class_names"])
    except ValueError as error:
        st.error(str(error))
        st.info("Download the bundled test CSV to see the required headers and exact target labels. No rows were silently removed or used for training.")
        st.stop()

    class_names = schema["class_names"]
    absent_classes = sorted(set(class_names) - set(y))
    st.caption(f"Active data: {source} | {len(frame):,} rows | {len(X.columns)} inputs | {y.nunique()}/7 classes present")
    if absent_classes:
        st.warning(
            "AUC is N/A because the test subset omits: " + ", ".join(absent_classes)
            + ". Precision, recall and F1 still average over all seven trained classes; absent-class terms are zero."
        )

    with st.spinner("Calculating predictions for all five saved models..."):
        try:
            results = {name: evaluate_model(model, X, y, class_names) for name, model in models.items()}
        except Exception as error:
            st.error(f"Prediction failed; no model was retrained: {error}")
            st.stop()

    chosen = results[selected]
    st.header(selected)
    for offset in (0, 3):
        for column, metric in zip(st.columns(3), METRIC_NAMES[offset:offset + 3]):
            value = chosen["metrics"][metric]
            column.metric(metric, "N/A" if not np.isfinite(value) else f"{value:.4f}")
    st.caption("Precision / Recall / F1: macro average over seven classes. AUC: macro one-vs-rest ROC AUC from probabilities. MCC: multiclass correlation.")

    compare_tab, diagnostics_tab, predictions_tab, schema_tab = st.tabs(
        ["Compare all models", "Class diagnostics", "Predictions", "Data contract"]
    )
    with compare_tab:
        st.subheader("Same test rows. All five models.")
        comparison = pd.DataFrame({name: result["metrics"] for name, result in results.items()}).T[METRIC_NAMES]
        comparison.index.name = "ML Model Name"
        st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch", alt="All five classifiers and six test metrics")
        st.bar_chart(comparison[["Accuracy", "F1", "MCC"]])
        st.caption("The highest score on an uploaded file describes that file only; it does not change model parameters or the training-CV selection.")
        st.download_button("Download metric comparison", comparison.to_csv().encode("utf-8"), "metric_comparison.csv", "text/csv")

    with diagnostics_tab:
        st.subheader("Where predictions succeed and fail")
        plot_column, report_column = st.columns([1.1, 1])
        with plot_column:
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(chosen["confusion_matrix"], display_labels=class_names).plot(
                ax=ax, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=45,
            )
            ax.set_title(selected)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Confusion matrix: actual classes are rows; predictions are columns. A text table is provided below.")
            matrix_frame = pd.DataFrame(chosen["confusion_matrix"], index=class_names, columns=class_names)
            st.dataframe(matrix_frame, width="stretch", alt="Confusion matrix counts, actual rows and predicted columns")
        with report_column:
            st.subheader("Classification report")
            report = pd.DataFrame(chosen["classification_report"]).T.loc[class_names, ["precision", "recall", "f1-score", "support"]]
            st.dataframe(report.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:.0f}"}),
                         width="stretch", alt="Per-class precision, recall, F1 and true support")

    with predictions_tab:
        predictions = pd.DataFrame({"Test row": np.arange(1, len(y) + 1), "Actual": y.to_numpy(), "Predicted": chosen["predictions"]})
        predictions["Correct"] = predictions["Actual"] == predictions["Predicted"]
        label_positions = {label: index for index, label in enumerate(class_names)}
        predicted_positions = [label_positions[label] for label in chosen["predictions"]]
        predictions["Probability of predicted label"] = chosen["probabilities"][np.arange(len(y)), predicted_positions]
        errors_only = st.checkbox("Show misclassified rows only", key="errors_only")
        visible = predictions.loc[~predictions["Correct"]] if errors_only else predictions
        st.dataframe(visible, hide_index=True, width="stretch", alt="Actual labels, model predictions and predicted-label probabilities")
        st.caption("Probabilities are model estimates, not guarantees. For SVM, predict and probability argmax can differ.")
        st.download_button("Download selected-model predictions", predictions.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

    with schema_tab:
        st.subheader("Required CSV format")
        st.write("Exactly the 27 numeric feature columns below plus the target column fault_type are required. Feature order is normalized to the training order; missing, extra or duplicated columns are rejected.")
        st.dataframe(pd.DataFrame({"Required input feature": schema["feature_names"]}), hide_index=True, width="stretch", alt="Required numeric input feature names")
        st.write("Allowed fault_type labels:", ", ".join(class_names))
        st.caption("K_Scatch is the original UCI spelling, not a renamed target. Nulls, infinities, nonnumeric inputs and unknown targets are rejected.")
        st.caption("The application cannot establish that an arbitrary upload is genuinely unseen. Use only a held-out test set, never training data, when interpreting generalization metrics.")
        st.markdown("Dataset: [UCI Steel Plates Faults](https://doi.org/10.24432/C5J88N), Buscema, Terzi & Tastle (2010), CC BY 4.0.")

    st.divider()
    st.caption("Learning reference. Review and independently adapt this work under the assignment's academic-integrity rules. A local app run is not proof of BITS Virtual Lab execution.")


if __name__ == "__main__":
    main()