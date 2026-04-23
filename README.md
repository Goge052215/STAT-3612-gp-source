# Light Brain Tumor Classifier

*STAT 3612 Group Project*

This repository contains our `STAT3612: Statistical Machine Learning` group project on presurgical brain tumor classification using the course-provided multimodal dataset.

The task is a five-class classification problem from the Kaggle competition, using structured clinical information and radiology reports. The current production approach in this repo is the LightGBM pipeline in `src/lgbm.py`.

- Kaggle competition link [here](https://www.kaggle.com/competitions/2026-spring-sdst-stat-3612-group-project/overview)
- Dataset download link [here](https://www.kaggle.com/competitions/2026-spring-sdst-stat-3612-group-project/data)
- **Final Report (PDF) link [here](https://github.com/Goge052215/Brain-Tumor-Classifier-Light/blob/main/report.pdf)**

## Project Overview

The pipeline combines:

- tabular clinical and derived report features;
- optional modality-level radiomics features with ANOVA filtering and missingness handling;
- TF-IDF features from radiology reports using both word and character n-grams;
- a multiclass LightGBM classifier;
- minority-class handling through class weighting, selective upsampling, Bayesian HPO, and OOF-driven class-scale tuning.

In short, the workflow is designed to perform well on the Kaggle macro-F1 style setting while staying fast enough for repeated experimentation.

## Current Workflow

The main workflow in `src/lgbm.py` is:

1. Load `train`, `val`, and `test` JSON splits from `data/`.
2. Optionally merge patient-level clinical CSV files from `data/clinical_information/`.
3. Optionally merge radiomics CSV features (`data/radiomics_info/{train,val,test}`), then filter radiomics columns with ANOVA (`p <= 0.05`) and missingness rules.
4. Clean reports and engineer extra report indicators such as hydrocephalus, pineal, sellar, and ventricular keywords.
5. Encode structured features and build TF-IDF text features.
6. Fuse tabular and text features into one sparse matrix.
7. Run 5-fold stratified CV diagnostics on `train + val`.
8. Upsample focus minority classes and run Bayesian HPO (via `src/utils.py`) on the main train/val split.
9. Tune class scales on validation probabilities, then re-tune scales with OOF probabilities on `train + val` (default behavior).
10. Retrain one final model on balanced `train + val` and generate final test predictions for Kaggle submission.

For a more **technical** explanation of the workflow, please refer to the [writeups.md](https://github.com/Goge052215/Brain-Tumor-Classifier-Light/blob/main/docs/writeups.md). Alternatively, check out our [Final Report (PDF)](https://github.com/Goge052215/Brain-Tumor-Classifier-Light/blob/main/report.pdf).

## Repo Layout

- `src/`: source code directory.
  - `src/lgbm.py`: main end-to-end training and prediction pipeline.
  - `src/utils.py`: shared helpers, including Bayesian optimization and submission builders.
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
