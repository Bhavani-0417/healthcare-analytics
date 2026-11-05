# Model Card — Healthcare Length-of-Stay Baseline

## Model identity

- Project: Healthcare Analytics — Patient Length-of-Stay Prediction
- Maintained estimator: scikit-learn RandomForestClassifier
- Model version: 0.2.0 baseline
- Intended development stage: educational / exploratory

## Intended use

Explore a tabular classification workflow for a dataset whose target represents a length-of-stay category. The project is intended for learning, experimentation, and demonstration of a reproducible machine-learning pipeline.

## Out-of-scope use

This model is not intended for diagnosis, admission, triage, treatment planning, discharge decisions, bed assignment, or other decisions affecting an individual patient. It has not been clinically validated and must not be used as a substitute for qualified professional judgment.

## Training and evaluation

The maintained workflow accepts a labeled CSV supplied by the user. It removes rows with missing target values, removes all-empty feature columns, splits the data into training and hold-out partitions, and fits preprocessing only on the training partition.

Numeric columns use median imputation. Categorical columns use most-frequent imputation and one-hot encoding with unknown categories ignored at inference time. A class-weighted Random Forest is used as the baseline estimator.

Metrics reported by the app are accuracy, balanced accuracy, macro-F1, weighted-F1, a classification report, and a confusion matrix.

## Dataset status

No verified training dataset is currently included in the maintained workflow. The archived historical test CSV is retained only for traceability. Dataset provenance, usage permissions, label semantics, and privacy status must be confirmed before use.

## Performance

No independently reproduced performance claim is made for this model version. The original project README included historical results for Naive Bayes, XGBoost, and a dense neural network. Those numbers were not re-created by the new pipeline and are not evidence of this baseline's performance.

When publishing a new experiment, record the dataset version, row counts, target definition, feature set, train/test split, random seed, per-class support, metrics, and any known leakage risks.

## Limitations and risks

- A random hold-out split may overestimate performance when records are correlated by patient, hospital, or time.
- Class imbalance can make accuracy misleading; inspect balanced accuracy and per-class metrics.
- Dataset shift across hospitals and populations may substantially reduce performance.
- Missing, miscoded, or inconsistently defined target labels can invalidate evaluation.
- Feature leakage may occur if inputs encode information only known after admission or after the outcome.
- The preprocessing and classifier pipeline is a technical baseline, not a clinically approved system.

## Ethical, privacy, and safety considerations

Only use datasets that are lawfully obtained and authorized for the intended use. Do not commit identifiable patient data, confidential records, credentials, or private model artifacts. Any real-world use requires qualified clinical governance, privacy and security controls, bias assessment, external validation, regulatory review, and ongoing monitoring.

## Maintenance

Reassess this model card whenever the dataset, feature engineering, training pipeline, evaluation method, or intended use changes.
