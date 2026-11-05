# Healthcare Analytics: Patient Length-of-Stay Prediction

A machine-learning project exploring hospital admission data to predict a patient's **length-of-stay (LOS) category**. The goal is to investigate how patient and admission characteristics can support hospital capacity planning and operational decision-making.

> **Project status:** Academic / exploratory project. The notebook and Streamlit scripts are historical implementations and should be reviewed and tested before being used as a production application. This project is not a clinical decision-support system.

## Problem Statement

Patient length of stay is an important operational measure for hospitals. Estimating the likely stay category at admission may help teams study bed demand, resource allocation, and care planning.

This repository explores data preparation, exploratory analysis, feature encoding, and classification approaches for LOS prediction.

## Models Explored

The original project report records the following accuracy values:

| Model | Reported accuracy |
| --- | ---: |
| Naive Bayes | 34.55% |
| XGBoost | 43.05% |
| Dense Neural Network | 42.50% |

**Interpretation:** These are values reported in the original project documentation. They have not been independently reproduced in this repository review. Re-run the notebook with a documented train/validation/test split before treating them as comparable model benchmarks.

## Repository Contents

```text
healthcare-analytics/
├── Healthcare-Analytics-main/
│   ├── Health50/
│   │   └── test.csv
│   ├── app.py
│   ├── app1.py
│   ├── health.py
│   ├── health.ipynb
│   ├── encoder.pkl
│   └── encoder1.pkl
└── README.md
```

- **health.ipynb** — exploratory data analysis, preprocessing, and model experiments.
- **app.py / app1.py / health.py** — Streamlit application experiments.
- **Health50/test.csv** — dataset file currently visible in the repository.
- **encoder*.pkl** — serialized encoders used by earlier experiments.

Some scripts refer to files such as `train.csv` and `model.pkl` that are not present at the paths currently referenced. The app therefore may not run as-is.

## Technology Stack

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- Streamlit

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Bhavani-0417/healthcare-analytics.git
cd healthcare-analytics
```

### 2. Set up Python

Use Python 3.10 or a compatible version, then create and activate a virtual environment.

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The repository does not currently include a verified dependency lockfile or requirements file. Install the packages used by the notebook and app scripts:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost streamlit jupyter
```

### 4. Explore the notebook

Open `Healthcare-Analytics-main/health.ipynb` in Jupyter. The notebook expects training and test CSV files in its working directory; provide the appropriate dataset files before running all cells.

### 5. Streamlit application

The Streamlit scripts are experimental and have file/model dependencies that need to be reconciled before launch. In particular, `health.py` loads `model.pkl` and `train.csv`; those files must exist at the expected paths and be generated using a compatible preprocessing pipeline.

## Important Reproducibility Notes

- Fit categorical encoders on training data only, then use the same fitted encoders to transform validation and test data.
- Keep preprocessing, feature ordering, and model artifacts together in a reproducible pipeline.
- Evaluate on a held-out dataset and report metrics such as accuracy, macro-F1, and a confusion matrix.
- Document the exact dataset source, target definition, split strategy, and random seed.
- Do not commit private, identifiable, or otherwise sensitive patient data.

## Responsible Use

This project is for educational experimentation with healthcare data. Predictions are not medical advice and must not be used to make decisions about patient treatment, discharge, or admission. Any real-world use would require appropriate clinical validation, privacy review, bias assessment, and governance.

## Future Improvements

- Consolidate the duplicate Streamlit scripts into one maintainable application.
- Add a reproducible preprocessing and model-training pipeline.
- Add a requirements file with tested package versions.
- Add model evaluation outputs and visualizations.
- Add input validation, error handling, and automated tests.
- Provide screenshots or a short demo after the application is verified.

## Author

**Bhavani Gujjari**  
GitHub: [@Bhavani-0417](https://github.com/Bhavani-0417)

---

Contributions, suggestions, and constructive feedback are welcome.
