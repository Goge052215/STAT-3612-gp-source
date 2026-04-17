# Light Brain Tumor Classifier

*STAT 3612 Group Project*

This repository contains our `STAT3612: Statistical Machine Learning` group project on presurgical brain tumor classification using the course-provided multimodal dataset.

The task is a five-class classification problem from the Kaggle competition, using structured clinical information and radiology reports. The current production approach in this repo is the LightGBM pipeline in `src/lgbm.py`.

- Kaggle competition link [here](https://www.kaggle.com/competitions/2026-spring-sdst-stat-3612-group-project/overview)
- Dataset download link [here](https://www.kaggle.com/competitions/2026-spring-sdst-stat-3612-group-project/data)

## Project Overview

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

For a more **technical** explanation of the workflow, please refer to the [writeups.md](https://github.com/Goge052215/Brain-Tumor-Classifier-Light/blob/main/docs/writeups.md).

## Repo Layout

- `src/`: source code directory.
  - `src/lgbm.py`: main end-to-end training and prediction pipeline.
  - `src/utils.py`: shared helpers, including the manual Bayesian optimization routine.
- `docs/`: documentation directory.
  - `docs/summary.md`: project brief and dataset description from the course.
  - `docs/writeups.md`: detailed method writeup and explanation of why the final LightGBM system works.
- `requirements.txt`: Python dependencies for the project.
- `data.zip`: dataset zip file, needs to be extracted to `data/` directory.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python src/lgbm.py
```
