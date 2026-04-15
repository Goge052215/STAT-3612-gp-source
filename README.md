# Brain Tumor Classifier

*STAT 3612 Group Project*

This repository contains our `STAT3612: Statistical Machine Learning` group project on presurgical brain tumor classification using the course-provided multimodal dataset.

The task is a five-class classification problem from the Kaggle competition, using structured clinical information and radiology reports. The current production approach in this repo is the LightGBM pipeline in `src/lgbm.py`.

## What This Project Does

The pipeline combines:

- tabular clinical and derived report features;
- TF-IDF features from radiology reports using both word and character n-grams;
- a multiclass LightGBM classifier;
- minority-class handling through class weighting, selective upsampling, and class-scale tuning.

In short, the workflow is designed to perform well on the Kaggle macro-F1 style setting while staying fast enough for repeated experimentation.

## Current Workflow

The main workflow in `src/lgbm.py` is:

1. Load `train`, `val`, and `test` JSON splits from `data/`.
2. Optionally merge patient-level clinical CSV files from `data/clinical_information/`.
3. Clean reports and engineer extra report indicators such as hydrocephalus, pineal, sellar, and ventricular keywords.
4. Encode structured features and build TF-IDF text features.
5. Fuse tabular and text features into one sparse matrix.
6. Upsample the hardest minority classes.
7. Tune LightGBM hyperparameters with the custom Bayesian optimization helper in `src/utils.py`.
8. Tune class-specific probability scales on the validation split.
9. Retrain on `train + val` and generate final test predictions for Kaggle submission.

## Repo Layout

- `src/lgbm.py`: main end-to-end training and prediction pipeline.
- `src/utils.py`: shared helpers, including the manual Bayesian optimization routine.
- `docs/summary.md`: project brief and dataset description from the course.
- `docs/writeups.md`: detailed method writeup and explanation of why the final LightGBM system works.
- `requirements.txt`: Python dependencies for the project.

## How I Use This Repo

My practical workflow is:

1. Read `docs/summary.md` to stay aligned with the project requirements and allowed data usage.
2. Use `docs/writeups.md` to document modeling decisions and keep the final report consistent with the code.
3. Iterate on `src/lgbm.py` for feature engineering, imbalance handling, and model tuning.
4. Run experiments locally, compare validation metrics, and keep the strongest pipeline as the Kaggle submission path.
5. Push cleaned project code and documentation to GitHub for version control and reproducibility.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python src/lgbm.py
```

Note: this script expects the course dataset files to exist under a local `data/` directory, which is not included in this repository.
