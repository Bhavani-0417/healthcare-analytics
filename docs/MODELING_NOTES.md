# Modeling and data notes

## Target and predictors

The original notebook uses `Stay` as the target label. The maintained workflow removes rows with a missing target, excludes `case_id` and `patientid` from predictors when present, and drops columns that are entirely missing. Remaining columns are candidate predictors; review them for relevance and leakage before interpreting results.

## Preprocessing

The preprocessing pipeline is fitted only on the training partition. Numeric features use median imputation. Categorical features use most-frequent imputation and one-hot encoding with unseen categories ignored at inference time.

## Evaluation

The baseline uses a fixed-seed holdout split and reports accuracy, balanced accuracy, macro-F1, weighted-F1, a per-class classification report, and a confusion matrix. These metrics describe only the supplied dataset and split; they are not evidence of clinical validity or generalization to another hospital.

For more serious assessment, consider patient-grouped or time-based splits where appropriate, cross-validation, calibration review, per-class error analysis, subgroup checks, and external validation.

## Historical notebook caution

The original notebook separately fits label encoders to train and test data and later creates synthetic classification data. Separate encoding can map categories inconsistently, and synthetic-data experiments do not measure generalization on the original hospital dataset. The maintained workflow does not use the historical notebook's reported results.

## Data and privacy

Use datasets only when their provenance and terms allow it. Do not upload or commit identifiable patient data, confidential records, or other sensitive information.
