# Contributing

Thank you for considering an improvement.

## Development setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -e ".[app,dev]"`.
4. Run tests: `pytest -q`.
5. Run lint: `ruff check app.py src tests`.

## Pull request checklist

- Keep changes focused and explain the problem solved.
- Add or update tests when behavior changes.
- Do not commit patient-identifiable, confidential, or unauthorized datasets.
- Do not commit credentials, local environment files, or generated model artifacts.
- Document dataset provenance, target definition, split strategy, and evaluation method.
- Do not present exploratory metrics as validated clinical performance.

Prefer small functions, descriptive names, readable error messages, and reproducible experiments.
