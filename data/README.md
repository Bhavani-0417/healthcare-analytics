# Data

The maintained workflow accepts a labeled CSV uploaded through Streamlit or supplied to the CLI. The original project used a target column named `Stay`. There is no verified training dataset in the maintained workflow.

Before using or publishing a dataset, verify its source, license, permitted use, and absence of identifying or confidential patient information. Document target meaning, feature definitions, and filtering decisions.

Keep large or private files outside Git. The root `.gitignore` excludes `data/raw/` and `data/processed/`.

Historical test data is preserved under `archive/legacy-project/Health50/` and is not automatically used by the new pipeline.
