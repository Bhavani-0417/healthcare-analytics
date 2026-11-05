# Local training data

Place an authorized **labeled training CSV** here as `train.csv`. It must contain the target column `Stay` (or pass another target column with `--target` when using the CLI).

The original labeled training file is not included in this repository. The historical `archive/legacy-project/Health50/test.csv` file is an unlabeled test dataset and cannot be used as a supervised training set.

Obtain data only from a source whose terms permit your intended use. Do not commit patient-identifiable, confidential, or otherwise sensitive records. CSV files in this directory are ignored by Git.

For a small artificial example to try the Streamlit interface, use `data/synthetic_demo.csv`. It is synthetic demonstration data, not real patient data.
