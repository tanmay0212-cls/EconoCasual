# EconoCausal Model Artifacts

This directory houses model metadata and serialized estimator bundles for the EconoCausal Double Machine Learning pipeline.

## Model Files

* **`dml_metadata.json`** (Tracked in Git): Comprehensive configuration and causal benchmark metadata for the trained `LinearDML` estimator, including nuisance model parameters, feature names, cross-fitting folds, and validation metrics against synthetic ground truth.
* **`dml_model.joblib`** (Excluded from Git / Auto-generated): Serialized model bundle containing the fitted EconML `LinearDML` estimator and fitted scikit-learn `ColumnTransformer` preprocessor.

## How to Generate or Retrain Model Weights

To train the Double Machine Learning model and generate `dml_model.joblib`:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Run training pipeline
python scripts/train_dml.py
```

This script:
1. Loads the synthetic retail campaign dataset (`sample_data/synthetic_retail_campaign.csv`).
2. Performs a stratified 80/20 train/evaluation split ($N_{\text{train}}=2,000$, $N_{\text{eval}}=500$).
3. Fits the feature preprocessor on training data only.
4. Trains the `LinearDML` model with 5-fold cross-fitting using `RandomForestRegressor` and `RandomForestClassifier` nuisance estimators.
5. Evaluates out-of-sample causal effects.
6. Serializes `dml_model.joblib` and `dml_metadata.json` to this directory.
