# Healthcare Analytics: Patient Length-of-Stay Prediction

An exploratory machine-learning project that studies hospital admission data and predicts a patient's **length-of-stay (LOS) category**. The refactored workflow provides a reproducible preprocessing pipeline, a held-out evaluation step, and a Streamlit demonstration app.

> **Educational project only.** This prototype is not medical advice or a clinical decision-support system. It must not be used to make admission, treatment, or discharge decisions.

## Project overview

Patient length of stay is an important operational measure. This project explores whether admission and hospital attributes can help estimate LOS categories for educational analysis of hospital capacity and resource planning.

### Models and historical results

The earlier README recorded these results:

| Historical model | Previously reported accuracy |
| --- | ---: |
| Naive Bayes | 34.55% |
| XGBoost | 43.05% |
| Dense Neural Network | 42.50% |

These values are retained only as historical project notes. They have not been reproduced by the refactored pipeline and should not be compared as validated benchmarks. Train the new pipeline on the labeled dataset to generate fresh metrics.

## Technology stack
- Python 3.10+
- Pandas and NumPy
- Scikit-learn and Joblib
- Streamlit
- Pytest and Ruff

## Repository structure

```text
healthcare-analytics/
 ├── app.py                      # Streamlit prediction interface
 ├─  train.py                    # Model training entry point
 ├─  pyproject.toml               # Project metadata and dependencies
 ┸─ src/healthcare_los/
 │   ├─ data.py                 # Dataset validation and feature preparation
│   └─ modeling.py             # Preprocessing, training, and evaluation
│  └─ cli.py                  # Training CLI
蔜── data/raw/                   # Local training CSV (not committed)
┸─ artifacts/                  # Generated model and evaluation outputs (not committed)
├─ tests/                      # Automated tests
├─ docs/MODELING_NOTES.md      # Modeling assumptions and cautions
┸─ [github/workflows/ci.yml]   # Automated test and lint workflow
┙─ Healthcare-Analytics-main/ # Original notebook and scripts retained for reference
```

## Dataset setup
The historical notebook expects a labeled training file with a `Stay` target. The current repository does **not** contain that labeled training file; it contains an older `Health50/test.csv` file, which is not suitable for supervised training because it does not include the target label.

1. Obtain the authorized labeled training CSV from the dataset provider or original project source.
2. Place it at `data/raw/train.csv`.
3. Confirm that it includes the `Stay` target column.
4. Do not commit patient-identifiable, confidential, or otherwise sensitive records. Local CSV files in `data/raw/` are ignored by Git.

See [`data/raw/README.md`](data/raw/README.md) for details.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Bhavani-0417/healthcare-analytics.git
cd healthcare-analytics
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the application and development dependencies:

```bash
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

## Train and evaluate

After placing the labeled CSV at `data/raw/train.csv`:

```bash
python train.py
```

Optional arguments:

```bash
python train.py --data data/raw/train.csv --target Stay --test-size 0.2 --random-state 42 --output artifacts
```

The training command fits preprocessing only on the training partition, evaluates against a stratified holdout partition, and writes:

- `artifacts/los_pipeline.joblib` — preprocessing and trained model in one artifact.
- `artifacts/metrics.json` — accuracy, macro-F1, weighted-F1, and per-class metrics.
- `artifacts/confusion_matrix.csv` — holdout confusion matrix.

Generated model files and metrics are excluded from Git by default.

## Run the Streamlit app

Train the model first, then run:

```bash
streamlit run app.py
```

The interface reads feature names, category options, and numeric ranges from the saved training artifact. It supports unseen categorical values through the fitted one-hot encoder's `handle_unknown="ignore"` behavior.

## Tests and code quality

```bash
pytest -q
ruff check .
```

A GitHub Actions workflow runs tests and lint checks on pushes to the refactor branch and on pull requests to `main`.

## Modeling notes and limitations

- `case_id` and `patientid` are excluded as identifiers; review all other features for leakage and relevance before drawing conclusions.
- Missing numeric values are median-imputed. Missing categorical values are most-frequent-imputed and one-hot encoded.
- Evaluation uses a stratified holdout split. For serious model assessment, add cross-validation, class-level error analysis, calibration review, and external validation.
- The earlier notebook separately fitted label encoders to train and test data and later generated synthetic classification data. Those notebook results are not used by the refactored training workflow.
- Any real-world healthcare use would require appropriate privacy, security, fairness, clinical validation, and governance review.

See [`docs/MODELING_NOTES.md`](docs/MODELING_NOTES.md).

## Future improvements

- Re-run model comparison using the authorized labeled dataset and a consistent evaluation protocol.
- Add feature-importance analysis and class-level error review.
- Add model versioning and reproducible experiment tracking.
- Add app screenshots only after the end-to-end workflow is verified.

## Author

**Bhavani Gujjari** · [GitHub](https://github.com/Bhavani-0417)

Feedback and suggestions are welcome.
