# Machine Learning Assignment 2: Steel Plate Fault Classification

## a. Problem Statement

Given 27 measured characteristics of a steel-plate surface fault, predict which of seven fault categories it belongs to. Compare Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes and Support Vector Machine on exactly the same held-out observations. Provide a Streamlit interface that evaluates the saved models on an uploaded labeled test CSV without retraining.

## b. Dataset Description

- **Dataset:** [UCI Steel Plates Faults, dataset 198](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).
- **Citation:** Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults [Dataset]. UCI Machine Learning Repository. [DOI: 10.24432/C5J88N](https://doi.org/10.24432/C5J88N).
- **License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- **Size:** 1,941 instances and 27 numerical input features (requirement: at least 500 instances and 12 features).
- **Target:** `fault_type`, derived from seven one-hot source indicators (exactly one is active per row). The indicators are excluded from the inputs. Classes: `Bumps` 402, `Dirtiness` 55, `K_Scatch` 391 (original source spelling), `Other_Faults` 673, `Pastry` 158, `Stains` 72, `Z_Scratch` 190.
- **Quality checks:** 0 missing values, 0 non-finite values and 0 duplicate rows. All rows are kept.
- **Split:** one stratified 80/20 split, `random_state=42`: 1,552 training rows and 389 test rows. The test rows are saved as [test_data.csv](test_data.csv) (27 inputs plus `fault_type`).

**Input features:** `X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`, `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Length_of_Conveyer`, `TypeOfSteel_A300`, `TypeOfSteel_A400`, `Steel_Plate_Thickness`, `Edges_Index`, `Empty_Index`, `Square_Index`, `Outside_X_Index`, `Edges_X_Index`, `Edges_Y_Index`, `Outside_Global_Index`, `LogOfAreas`, `Log_X_Index`, `Log_Y_Index`, `Orientation_Index`, `Luminosity_Index`, `SigmoidOfAreas`.

## c. GitHub Repository Link

- **GitHub repository:** _add your repository URL here_
- **Live Streamlit app:** _add your Streamlit Community Cloud URL here_

## d. Models Used

All five models use the same training/test rows and the same five stratified cross-validation folds on the training data. Hyperparameters are chosen by grid search maximizing training-only macro F1. `StandardScaler` is inside the Logistic Regression, KNN, Gaussian Naive Bayes and SVM pipelines; Decision Tree uses unscaled features.

Precision, Recall and F1 are macro averages over the seven classes. AUC is the macro one-vs-rest ROC AUC from predicted probabilities. MCC uses the multiclass formula.

### Comparison Table (389 held-out test rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7172 | 0.8608 | 0.7076 | 0.7466 | 0.7227 | 0.6374 |
| KNN | 0.7686 | 0.9220 | 0.7653 | 0.7866 | 0.7741 | 0.7044 |
| Naive Bayes | 0.6195 | 0.9257 | 0.6472 | 0.7212 | 0.6429 | 0.5604 |
| Support Vector Machine (SVM) | 0.7532 | 0.9498 | 0.7971 | 0.7509 | 0.7679 | 0.6814 |

### Selected Hyperparameters

| Model | Selected settings | CV macro F1 |
| --- | --- | --- |
| Logistic Regression | C=1.0, class_weight=None | 0.7220 |
| Decision Tree | criterion=entropy, max_depth=12, min_samples_leaf=1, class_weight=None | 0.7562 |
| KNN | n_neighbors=5, p=2, weights=distance | 0.7648 |
| Naive Bayes (Gaussian) | var_smoothing=1e-05 | 0.6199 |
| Support Vector Machine | RBF kernel, C=10.0, gamma=0.01, class_weight=None | 0.7837 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | The regularized linear baseline reaches accuracy 0.7275, macro F1 0.7418 and AUC 0.9379. Its training F1 (0.7413) is close to its test F1, so it does not overfit. It predicts Other_Faults for 30 of 81 actual Bumps cases. |
| Decision Tree | Training macro F1 is 0.9687, but CV F1 is 0.7562 and test F1 is 0.7227, showing a large generalization gap. Its AUC (0.8608) is the lowest because leaf-frequency probabilities rank classes less well. |
| KNN | Distance-weighted KNN with 5 neighbors gives the highest test accuracy (0.7686), macro F1 (0.7741) and MCC (0.7044). It correctly predicts 10/11 Dirtiness and 75/78 K_Scatch cases, but only 14/32 Pastry cases. |
| Naive Bayes | Gaussian Naive Bayes gives the lowest accuracy (0.6195) and macro F1 (0.6429), although its AUC is 0.9257. It recognizes only 38/135 Other_Faults cases. Several features are almost perfectly correlated (for example, Y_Minimum and Y_Maximum), which violates its independence assumption. |
| Support Vector Machine | RBF SVM has the best training-CV macro F1 (0.7837), the best test AUC (0.9498) and the best macro precision (0.7971). Its test macro F1 (0.7679) and recall (0.7509) are slightly below KNN. |
| **Overall winner** | **KNN** performs best on the held-out test set (accuracy 0.7686, F1 0.7741, MCC 0.7044). SVM is a close second and is the model chosen by training-only cross-validation, so it is the app's default selection. |

## Repository Structure

| Path | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit app: CSV upload, model dropdown, six metrics, confusion matrix and classification report |
| [evaluation.py](evaluation.py) | CSV validation and metric calculations used by the app and training script |
| [artifacts.py](artifacts.py) | Loads the saved models after checksum and version checks |
| [train_models.py](train_models.py) | Source code: loads the dataset, splits, tunes, evaluates and saves all five models |
| [model/](model/) | Five trained pipelines (`.joblib`) and `schema.json` (features, classes, checksums) |
| [test_data.csv](test_data.csv) | Labeled held-out test data (389 rows) for uploading to the app |
| [data/steel_plates_faults.zip](data/steel_plates_faults.zip) | Original UCI dataset archive |
| [results/metrics_comparison.csv](results/metrics_comparison.csv) | Full-precision metric table |
| [requirements.txt](requirements.txt) | Python dependencies |

## How to Run

Use Python 3.14 (the saved models were created with scikit-learn 1.9.0).

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

To retrain and regenerate the models, test data and metrics:

```bash
python train_models.py
```
