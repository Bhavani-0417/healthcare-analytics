# Model Card — Healthcare Length-of-Stay Baseline

## Model identity

- Project: Healthcare Analytics — Patient Length-of-Stay Prediction
- Maintained estimator: scikit-learn RandomForestClassifier
- Model version: 1.0.0 baseline
- Development stage: educational / exploratory

## Intended use

Demonstrate a tabular classification workflow using a dataset whose target represents a length-of-stay category. The project is intended for learning, experimentation, and evaluation practice.

## Out-of-scope use

This model is not intended for diagnosis, admission, triage, treatment planning, discharge decisions, bed assignment, or other decisions affecting an individual patient. It has not been clinically validated and must not be used as a substitute for qualified professional judgment.

## Training and evaluation

The maintained workflow accepts a labeled CSV supplied by the user. It removes rows with missing target values, excludes the known identifier columns `case_id` and `patientid` when present, removes all-empty predictor columns, and splits the data into training and hold-out partitions.

Preprocessing is fitted only on the training partition. Numeric values are median-imputed. Categorical values are most-frequent-imputed and one-hot encoded; unseen categories are ignored at inference. A class-weighted Random Forest is used as the baseline classifier.

Reported metrics are accuracy, balanced accuracy, macro-F1, weighted-F1, a per-class classification report, and a confusion matrix.

## Dataset status

No verified original training dataset is included in the repository. The synthetic demo data exists only to test the app interface and is not real patient data. Dataset provenance, usage permissions, target semantics, and privacy status must be confirmed before any external dataset is used.

## Performance

No validated performance claim is made for this model version. Previously reported Naive Bayes, XGBoost, and neural-network results from the original project have not been independently reproduced by the maintained workflow.

When publishing an experiment, record the dataset version, row counts, target definition, feature set, split strategy, random seed, per-class support, metrics, and known leakage risks.

## Limitations and risks

- A random hold-out split may overestimate performance when records are correlated by patient, hospital, or time.
- Class imbalance can make accuracy misleading; inspect balanced accuracy and per-class metrics.
- Dataset shift across hospitals and populations may substantially reduce performance.
- Missing, miscoded, or inconsistently defined target labels can invalidate evaluation.
- Features that encode information only known after the prediction point can cause target leakage.
- Excluding row/patient identifiers reduces one source of memorization risk but does not remove all potential leakage.
- The model is a technical baseline, not a clinically approved system.

## Ethical, privacy, and safety considerations

Only use datasets that are lawfully obtained and authorized for the intended use. Do not commit identifiable patient data, confidential records, credentials, or private model artifacts. Any real-world use requires qualified clinical governance, privacy and security controls, bias assessment, external validation, regulatory review, and ongoing monitoring.

## Maintenance

Reassess this model card whenever the dataset, feature engineering, training pipeline, evaluation method, or intended use changes.
