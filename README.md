# Healthcare Analytics — Patient Length-of-Stay Prediction

A reproducible machine-learning project for exploring hospital admission data and predicting a patient's **length-of-stay (LOS) category**.

> **Educational use only.** This project is not a clinical decision-support system. Model outputs must not be used to make admission, treatment, triage, or discharge decisions.

## Project overview

Patient length of stay is an important operational measure for hospitals. This project explores whether structured admission data can be used to classify a patient's expected stay category.

### Maintained workflow

- Upload a labeled CSV through the Streamlit application.
- Select the target column (the original project used `Stay`).
- Split the data into training and hold-out test sets.
- Impute missing values and encode categorical features inside a scikit-learn pipeline.
- Train a Random Forest baseline.
- Review accuracy, balanced accuracy, macro-F1, weighted-F1, a classification report, and a confusion matrix.
- Download the fitted model bundle.
- Upload a second CSV containing the same input features to generate batch predictions and download the results.

Preprocessing is fitted on the training split only. The preprocessing and model are saved together in one pipeline to avoid inconsistent train/test transformations.

## Repository structure

```text
healthcare-analytics/
├── app.py
├── src/healthcare_los/
│   ├── __init__.py
│   ├── modeling.py
│   └── cli.py
├── tests/test_modeling.py
├── notebooks/
│   ├── README.md
│   └── legacy_healthcare_analysis.ipynb
├── archive/legacy-project/
│   ├── Health50/test.csv
│   ├── app.py
│   ├── app1.py
│   ├── encoder.pkl
│   ├── encoder1.pkl
│   └── health.py
├── data/README.md
├── .github/workflows/tests.yml
├── .gitignore
├── CONTRIBUTING.md
├── pyproject.toml
├── requirements.txt
└── requirements-notebook.txt
```

The original notebook, Streamlit experiments, encoder files, and historical test dataset are preserved for reference. The maintained application and model pipeline are at the repository root and under `src/healthcare_los/`.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Bhavani-0417/healthcare-analytics.git
cd healthcare-analytics
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Upload a labeled CSV, select its target column (usually `Stay`), and train the model. For a quick UI walkthrough, try `data/synthetic_demo.csv`; it contains artificial demonstration rows only. After evaluation, upload a second CSV with the same feature columns but without the target to produce batch predictions.

### 5. Train from the command line

```bash
python -m healthcare_los.cli --input path/to/train.csv --target Stay --output models/los_model.joblib
```

The output directory is created automatically. Generated model files are excluded from Git; train locally using a dataset you are authorized to use.

### 6. Run tests

```bash
pytest
```

## Model and evaluation

The maintained workflow currently uses a Random Forest classifier with:
- Median imputation for numeric features.
- Most-frequent imputation and one-hot encoding for categorical features.
- A reproducible train/hold-out split (default: 80/20, random seed 42).
- Class-balanced sampling weights.

The application reports:

| Metric | Meaning |
| --- | --- |
| Accuracy | Overall fraction of correct predictions |
| Balanced accuracy | Average recall across classes |
| Macro-F1 | F1 averaged equally across classes |
| Weighted-F1 | F1 averaged by class support |
| Confusion matrix | Correct and incorrect predictions by class |

### Historical experiments

The original academic README recorded these values:

| Historical experiment | Previously reported accuracy |
| --- | ---: |
| Naive Bayes | 34.55% |
| XGBoost | 43.05% |
| Dense Neural Network | 42.50% |

These figures have **not** been independently reproduced in the maintained pipeline. Do not compare them with new results until the original dataset, split strategy, preprocessing, and model configurations are recovered and documented. The current app uses a separate Random Forest baseline.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for the model's intended use, evaluation limitations, and data requirements.

## Data and privacy

The maintained training workflow expects a labeled CSV uploaded by the user or supplied to the CLI. The repository includes `data/synthetic_demo.csv`, a small artificial dataset for trying the app workflow. It is not real patient data and must not be used to draw healthcare conclusions. The repository does not contain a verified real training dataset for this pipeline.

The historical `test.csv` is preserved at `archive/legacy-project/Health50/test.csv`. Its provenance and license should be verified before redistribution or reuse. Do not upload identifiable patient information, confidential records, or data you are not authorized to share.

See [data/README.md](data/README.md) for data-handling guidance.

## Engineering and limitations

- Categorical encoders are fitted on training data only and reused for hold-out and prediction data.
- Unseen inference categories are handled by the fitted one-hot encoder.
- The model artifact contains the complete preprocessing and classifier pipeline.
- Automated tests use synthetic data and do not require a healthcare dataset.
- A random hold-out split is not a substitute for temporal or external validation.
- Dataset leakage, class imbalance, distribution shift, and inconsistent labels can materially affect results.

This is an educational project, not medical advice. Any real-world healthcare application would require appropriate clinical validation, privacy and security review, bias assessment, regulatory review, and qualified human oversight.

## Roadmap

- [x] Separate maintained code from original experiments.
- [x] Add a reusable preprocessing and training pipeline.
- [x] Add hold-out evaluation and model export.
- [x] Add batch prediction from a CSV upload.
- [x] Add automated tests and continuous integration.
- [ ] Reproduce historical experiments on a verified dataset.
- [ ] Add screenshots after testing with an authorized dataset.
- [ ] Add external or temporal validation.

## Author

**Bhavani Gujjari**  
[GitHub: @Bhavani-0417](https://github.com/Bhavani-0417)

Feedback and suggestions are welcome.
