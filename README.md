# Healthcare Analytics — Patient Length-of-Stay Prediction

An exploratory machine-learning project for classifying a hospital patient's **length-of-stay (LOS) category** from structured admission data. The maintained workflow includes reusable preprocessing, hold-out evaluation, a Streamlit demonstration app, and a command-line training entry point.

> **Educational use only.** This prototype is not medical advice or a clinical decision-support system. It must not be used to make admission, triage, treatment, or discharge decisions.

## Project overview

Patient length of stay is an important operational measure for hospitals. This project explores whether selected admission and hospital attributes can help estimate LOS categories for learning and experimentation.

### What the maintained workflow does

- Accepts a labeled CSV upload in the Streamlit app.
- Uses the selected target column (the original project used `Stay`).
- Excludes known row/patient identifiers `case_id` and `patientid` when present.
- Splits data into training and hold-out partitions.
- Fits imputation and categorical encoding on the training partition only.
- Trains a class-weighted Random Forest baseline.
- Reports accuracy, balanced accuracy, macro-F1, weighted-F1, a classification report, and a confusion matrix.
- Downloads the fitted model bundle and supports batch predictions from another CSV.

The pipeline uses median imputation for numeric features and most-frequent imputation plus one-hot encoding for categorical features. Unknown categories at prediction time are handled by the fitted encoder.

## Technology stack

- Python 3.10+
- Pandas and NumPy
- Scikit-learn and Joblib
- Streamlit
- Pytest and Ruff
- Jupyter Notebook for the historical analysis

## Repository structure

```text
healthcare-analytics/
|-- app.py
|-- train.py
|-- pyproject.toml
|-- requirements.txt
|-- requirements-notebook.txt
|-- src/healthcare_los/
|   |-- __init__.py
|   |-- data.py
|   |-- modeling.py
|   `-- cli.py
|-- tests/
|-- data/
|   |-- raw/README.md
|   `-- synthetic_demo.csv
|-- artifacts/                 # Generated model/evaluation files; git-ignored
|-- docs/
|   |-- MODELING_NOTES.md
|   `-- MODEL_CARD.md
|-- notebooks/                 # Original exploratory notebook
|-- archive/legacy-project/    # Superseded app scripts, encoders, and old test CSV
|-- .github/workflows/ci.yml
`-- CONTRIBUTING.md
```

## Quick start

### 1. Clone and create an environment

```bash
git clone https://github.com/Bhavani-0417/healthcare-analytics.git
cd healthcare-analytics
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[app,dev]"
```

For the original notebook experiments, install `requirements-notebook.txt` as well.

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Upload a labeled CSV, select the target column, and choose **Train and evaluate model**. To try the interface without a real dataset, upload `data/synthetic_demo.csv`. It contains artificial demonstration rows only and must not be described as real healthcare data.

After training, the app displays hold-out metrics and the confusion matrix. You can download the fitted model bundle. To make batch predictions, upload a second CSV with the same predictor columns (the target column is not required).

### 4. Train from the command line

For repeatable local training, put an authorized labeled CSV at `data/raw/train.csv` (with a `Stay` column), then run:

```bash
python train.py
```

Or provide paths and settings explicitly:

```bash
python train.py --data data/raw/train.csv --target Stay --test-size 0.2 --random-state 42 --output artifacts
```

The command writes:

- `artifacts/los_pipeline.joblib` — fitted preprocessing and model bundle.
- `artifacts/metrics.json` — metrics and per-class report.
- `artifacts/confusion_matrix.csv` — hold-out confusion matrix.

Generated artifacts and local raw CSV files are excluded from Git.

## Dataset and privacy

The repository does not include the original labeled training dataset. The historical `archive/legacy-project/Health50/test.csv` file is an unlabeled test file and cannot be substituted for a labeled training set.

Before training on external data:
1. Verify dataset provenance, license, and permitted use.
2. Confirm the target definition and meaning of every feature.
3. Do not upload or commit identifiable, confidential, or otherwise sensitive patient information.
4. Keep private datasets and generated model files outside Git.

See [data/raw/README.md](data/raw/README.md).

## Tests and code quality

```bash
pytest -q
ruff check app.py train.py src tests
```

GitHub Actions runs tests and lint checks on pushes to `main` and pull requests to `main`. Legacy files are retained for traceability and excluded from the maintained-code lint target.

## Historical experiments

The original README recorded these accuracy values:

| Historical model | Previously reported accuracy |
| --- | ---: |
| Naive Bayes | 34.55% |
| XGBoost | 43.05% |
| Dense Neural Network | 42.50% |

These values are retained only as historical notes. They have not been independently reproduced by the maintained pipeline and must not be presented as validated benchmarks. The original notebook separately fits label encoders on training and test data and includes synthetic-data experiments; its results should be re-evaluated before comparison.

## Modeling notes and limitations

- Excluding `case_id` and `patientid` reduces identifier memorization, but other feature leakage may remain. Review feature definitions carefully.
- A stratified random hold-out split is a baseline, not external validation.
- Class imbalance can make accuracy misleading; inspect balanced accuracy and per-class metrics.
- Correlated records by patient, hospital, or time can make a random split optimistic.
- Any real-world healthcare use would require privacy, security, bias, clinical validation, and governance review.

See [docs/MODELING_NOTES.md](docs/MODELING_NOTES.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Author

**Bhavani Gujjari** · [GitHub](https://github.com/Bhavani-0417)

Feedback and constructive suggestions are welcome.
