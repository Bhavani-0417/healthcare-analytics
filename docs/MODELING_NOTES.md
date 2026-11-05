# Modeling and data notes

## Target and predictors

The original notebook uses `Stay` as the target label. The refactored baseline excludes `case_id` and `patientid` from predictors because they are identifiers rather than generalizable patient attributes. All remaining columns are treated as candidate predictors; review them for leakage and appropriateness before interpreting results.

## Preprocessing

The training pipeline learns imputation and categorical encoding from the training split only. It uses median imputation for numeric columns, most-frequent imputation plus one-hot encoding for categorical columns, and ignores unseen categories at inference time.

## Evaluation

A stratified holdout split is used with a fixed random seed. Accuracy, macro-F1, weighted-F1, a per-class classification report, and a confusion matrix are saved under `artifacts/` after training. These metrics are descriptive and should not be interpreted as evidence of clinical validity.

## Historical notebook caution

The original notebook contains separate label-encoding of training and test frames and later creates synthetic classification data. Those operations can lead to inconsistent encodings or results that do not measure performance on the original hospital dataset. The refactored training command does not reuse those notebook metrics.
