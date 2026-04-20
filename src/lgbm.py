import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from utils import bayes_optimize, build_submission

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The optimal value found for dimension .* of parameter k1__k2__length_scale is close to the specified upper bound 100.0.*",
    category=ConvergenceWarning,
    module="sklearn.gaussian_process.kernels",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLINICAL_DIR = DATA_DIR / "clinical_information"
TRAIN_RADIOMICS_DIR = DATA_DIR / "radiomics_info" / "train"
VAL_RADIOMICS_DIR = DATA_DIR / "radiomics_info" / "val"
TEST_RADIOMICS_DIR = DATA_DIR / "radiomics_info" / "test"

FOCUS_CLASSES = [
    "Brain Metastase Tumour",
    "Pineal tumour and Choroid plexus tumour",
    "Tumors of the sellar region",
]

KFOLD_N_SPLITS = 5
KFOLD_RANDOM_STATE = 42
OOF_SCALE_N_SPLITS = 5


def load_json_split(name: str):
    with (DATA_DIR / f"{name}.json").open("r") as f:
        return json.load(f)


def load_optional_clinical_csv(name: str):
    csv_path = CLINICAL_DIR / f"{name}_patient_info.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_optional_radiomics_split(split_dir: str | Path, split_name: str):
    p = Path(split_dir)
    if not p.exists():
        return None
    files = sorted(p.glob(f"*_radiomics_{split_name}.csv"))
    if not files:
        return None

    merged = None
    for fp in files:
        modality = fp.name.replace(f"_radiomics_{split_name}.csv", "")
        df = pd.read_csv(fp, encoding="utf-8-sig")
        if "case_id" not in df.columns:
            continue
        df = df.drop(columns=["sex", "age", "modality"], errors="ignore")
        rad_cols = [c for c in df.columns if c.startswith("rad_")]
        if not rad_cols:
            continue
        keep = df[["case_id"] + rad_cols].copy()
        keep["case_id"] = pd.to_numeric(keep["case_id"], errors="coerce")
        keep = keep.dropna(subset=["case_id"])
        keep["case_id"] = keep["case_id"].astype(int)
        keep = keep.rename(columns={c: f"{modality}__{c}" for c in rad_cols})
        merged = keep if merged is None else pd.merge(merged, keep, on="case_id", how="outer")
    return merged


def select_radiomics_by_anova(train_df: pd.DataFrame, p_threshold: float = 0.05):
    radiomics_cols = [c for c in train_df.columns if "__rad_" in c or c.startswith("rad_")]
    if not radiomics_cols:
        return []

    y = train_df["Overall_class"]
    valid_mask = y.notna()
    if int(valid_mask.sum()) == 0 or int(y[valid_mask].nunique()) < 2:
        return radiomics_cols

    X = train_df.loc[valid_mask, radiomics_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    y = y.loc[valid_mask]
    _, pvals = f_classif(X, y)
    return [
        c for c, p in zip(radiomics_cols, pvals)
        if np.isfinite(p) and float(p) <= float(p_threshold)
    ]


def json_to_df(data):
    rows = []
    for case_id, info in data.items():
        report = info.get("report", "")
        modalities = info.get("available_modalities", [])
        label = info.get("Overall_class", None)
        rows.append(
            {
                "case_id": int(case_id),
                "Overall_class": label,
                "report": report,
                "n_modalities": len(modalities),
            }
        )
    return pd.DataFrame(rows)


def combine_split(clinical_df: pd.DataFrame | None, json_df: pd.DataFrame):
    if clinical_df is None:
        return json_df.copy()
    return pd.merge(clinical_df, json_df, on="case_id", how="left")


def align_columns_from_train(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    for c in train_df.columns:
        if c not in val_df.columns:
            val_df[c] = np.nan
        if c not in test_df.columns:
            test_df[c] = np.nan
    return train_df, val_df, test_df


def add_report_features(df: pd.DataFrame):
    df = df.copy()
    df["report"] = df["report"].fillna("").astype(str).str.lower()
    report_tokens = df["report"].str.findall(r"\b\w+\b")

    df["report_len"] = df["report"].apply(len)
    df["report_word_count"] = report_tokens.apply(len)
    df["report_unique_word_count"] = report_tokens.apply(lambda toks: len(set(toks)))
    df["report_unique_ratio"] = (
        df["report_unique_word_count"] / df["report_word_count"].clip(lower=1)
    )
    df["report_digit_count"] = df["report"].str.count(r"\d")
    df["report_punct_count"] = df["report"].str.count(r"[^\w\s]")
    df["has_enhancement"] = df["report"].str.contains("enhancement").astype(int)
    df["has_edema"] = df["report"].str.contains("edema").astype(int)
    df["has_midline_shift"] = df["report"].str.contains("midline").astype(int)
    df["has_hydrocephalus"] = df["report"].str.contains("hydrocephalus").astype(int)
    df["has_multiple"] = df["report"].str.contains("multiple").astype(int)
    df["has_diffusion_restrict"] = df["report"].str.contains(r"diffusion|restrict").astype(int)
    df["has_cystic"] = df["report"].str.contains(r"cyst|cystic").astype(int)
    df["has_extra_axial_meningioma_keywords"] = df["report"].str.contains(
        r"dural tail|extra-axial|parasagittal|convexity|falx|meningioma"
    ).astype(int)
    df["has_pineal"] = df["report"].str.contains(r"pineal").astype(int)
    df["has_sellar"] = df["report"].str.contains(r"sellar|suprasellar|pituitary").astype(int)
    df["has_ventricular"] = df["report"].str.contains(r"ventricle|ventricular").astype(int)
    df["focus_keyword_hits"] = (
        df["has_pineal"] + df["has_sellar"] + df["has_hydrocephalus"] + df["has_ventricular"]
    )
    df["enhancement_x_edema"] = df["has_enhancement"] * df["has_edema"]
    df["hydrocephalus_x_ventricular"] = df["has_hydrocephalus"] * df["has_ventricular"]
    return df


def fill_missing(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    for col in train.columns:
        if col == "Overall_class":
            continue
        if col == "report":
            train[col] = train[col].fillna("")
            val[col] = val[col].fillna("")
            test[col] = test[col].fillna("")
            continue
        if train[col].dtype == "object":
            train[col] = train[col].fillna("Unknown")
            val[col] = val[col].fillna("Unknown")
            test[col] = test[col].fillna("Unknown")
        else:
            median_value = train[col].median()
            train[col] = train[col].fillna(median_value)
            val[col] = val[col].fillna(median_value)
            test[col] = test[col].fillna(median_value)
    return train, val, test


def build_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    tabular_cols = [c for c in X_train.columns if c not in ["report", "case_id"]]
    X_train_tab = X_train[tabular_cols].copy()
    X_val_tab = X_val[tabular_cols].copy()
    X_test_tab = X_test[tabular_cols].copy()

    all_tab = pd.concat([X_train_tab, X_val_tab, X_test_tab], axis=0)
    all_tab = pd.get_dummies(all_tab, drop_first=False)
    X_train_tab_encoded = all_tab.iloc[: len(X_train_tab), :]
    X_val_tab_encoded = all_tab.iloc[len(X_train_tab): len(X_train_tab) + len(X_val_tab), :]
    X_test_tab_encoded = all_tab.iloc[len(X_train_tab) + len(X_val_tab):, :]

    tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500)
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=1200,
    )

    X_train_text_word = tfidf_word.fit_transform(X_train["report"])
    X_val_text_word = tfidf_word.transform(X_val["report"])
    X_test_text_word = tfidf_word.transform(X_test["report"])
    X_train_text_char = tfidf_char.fit_transform(X_train["report"])
    X_val_text_char = tfidf_char.transform(X_val["report"])
    X_test_text_char = tfidf_char.transform(X_test["report"])
    X_train_text = hstack([X_train_text_word, X_train_text_char], format="csr")
    X_val_text = hstack([X_val_text_word, X_val_text_char], format="csr")
    X_test_text = hstack([X_test_text_word, X_test_text_char], format="csr")

    X_train_encoded = hstack(
        [csr_matrix(X_train_tab_encoded.to_numpy(dtype="float32")), X_train_text],
        format="csr",
    )
    X_val_encoded = hstack(
        [csr_matrix(X_val_tab_encoded.to_numpy(dtype="float32")), X_val_text],
        format="csr",
    )
    X_test_encoded = hstack(
        [csr_matrix(X_test_tab_encoded.to_numpy(dtype="float32")), X_test_text],
        format="csr",
    )
    return X_train_encoded, X_val_encoded, X_test_encoded


def to_lgbm_input(X):
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    if hasattr(X, "astype"):
        return X.astype(np.float32, copy=False)
    return X


def blend_score(y_true, y_pred, focus_ids):
    micro = float(f1_score(y_true, y_pred, average="micro"))
    weighted = float(f1_score(y_true, y_pred, average="weighted"))
    if len(focus_ids) == 0:
        focus_recall = weighted
    else:
        recalls = recall_score(y_true, y_pred, labels=focus_ids, average=None, zero_division=0)
        focus_recall = float(np.mean(recalls))
    return 0.45 * micro + 0.35 * weighted + 0.20 * focus_recall


def normalize_lgbm_params(params):
    return {
        "n_estimators": int(np.clip(int(round(params["n_estimators"])), 120, 800)),
        "learning_rate": float(10 ** np.clip(float(params["lr_log10"]), -2.5, -0.7)),
        "num_leaves": int(np.clip(int(round(params["num_leaves"])), 16, 128)),
        "max_depth": int(np.clip(int(round(params["max_depth"])), -1, 20)),
        "min_child_samples": int(np.clip(int(round(params["min_child_samples"])), 5, 80)),
        "subsample": float(np.clip(float(params["subsample"]), 0.6, 1.0)),
        "colsample_bytree": float(np.clip(float(params["colsample_bytree"]), 0.6, 1.0)),
        "reg_alpha": float(10 ** np.clip(float(params["reg_alpha_log10"]), -6.0, 1.0)),
        "reg_lambda": float(10 ** np.clip(float(params["reg_lambda_log10"]), -6.0, 1.0)),
        "objective": "multiclass",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }


def upsample_focus_classes(X, y, focus_ids, seed=42):
    if len(focus_ids) == 0:
        return X, y
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(y)
    median_count = int(np.median(class_counts[class_counts > 0]))
    x_blocks = [X]
    y_blocks = [y]
    for cls_id in focus_ids:
        cls_idx = np.where(y == cls_id)[0]
        if len(cls_idx) == 0:
            continue
        target_count = max(len(cls_idx) * 3, median_count)
        extra = target_count - len(cls_idx)
        if extra <= 0:
            continue
        sampled = rng.choice(cls_idx, size=extra, replace=True)
        x_blocks.append(X[sampled])
        y_blocks.append(y[sampled])
    X_aug = vstack(x_blocks, format="csr")
    y_aug = np.concatenate(y_blocks)
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], y_aug[perm]


def apply_class_scales(proba, scales):
    adjusted = proba * scales.reshape(1, -1)
    denom = adjusted.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return adjusted / denom


def tune_class_scales(y_true, proba, focus_ids, rounds=4):
    n_classes = proba.shape[1]
    if len(focus_ids) == 0:
        return np.ones(n_classes, dtype=np.float64), blend_score(y_true, np.argmax(proba, axis=1), focus_ids)

    scales = np.ones(n_classes, dtype=np.float64)
    best_score = blend_score(y_true, np.argmax(proba, axis=1), focus_ids)
    grid = np.unique(np.concatenate([np.linspace(0.5, 3.0, 26), np.array([1.15, 1.35, 1.65, 2.25, 2.75])]))

    for _ in range(rounds):
        improved = False
        for cls_id in focus_ids:
            best_local = scales[cls_id]
            for g in grid:
                trial = scales.copy()
                trial[cls_id] = g
                pred = np.argmax(apply_class_scales(proba, trial), axis=1)
                score = blend_score(y_true, pred, focus_ids)
                if score > best_score:
                    best_score = score
                    best_local = g
                    improved = True
            scales[cls_id] = best_local
        if not improved:
            break
    return scales, best_score


def tune_lgbm_bayes(X_tr, y_tr, X_va, y_va, focus_ids, n_trials=40, n_init=10, seed=42):
    bounds = {
        "n_estimators": (120, 700),
        "lr_log10": (-2.3, -0.9),
        "num_leaves": (20, 128),
        "max_depth": (-1, 18),
        "min_child_samples": (5, 80),
        "subsample": (0.65, 1.0),
        "colsample_bytree": (0.65, 1.0),
        "reg_alpha_log10": (-6.0, 0.7),
        "reg_lambda_log10": (-6.0, 0.7),
    }
    best_score = -1.0
    best_params = None

    def _objective(
        n_estimators,
        lr_log10,
        num_leaves,
        max_depth,
        min_child_samples,
        subsample,
        colsample_bytree,
        reg_alpha_log10,
        reg_lambda_log10,
    ):
        nonlocal best_score, best_params
        params = normalize_lgbm_params(
            {
                "n_estimators": n_estimators,
                "lr_log10": lr_log10,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child_samples,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_alpha_log10": reg_alpha_log10,
                "reg_lambda_log10": reg_lambda_log10,
            }
        )
        model = LGBMClassifier(**params)
        model.fit(to_lgbm_input(X_tr), y_tr)
        pred = model.predict(to_lgbm_input(X_va))
        score = blend_score(y_va, pred, focus_ids)
        if score > best_score:
            best_score = score
            best_params = params
        return score

    bayes_optimize(
        objective=_objective,
        bounds=bounds,
        n_trials=n_trials,
        n_init=n_init,
        seed=seed,
    )

    if best_params is None:
        raise RuntimeError("Bayes HPO did not produce valid parameters.")
    return best_params, best_score


def run_kfold_cv(X_all: pd.DataFrame, y_all: pd.Series, X_test_ref: pd.DataFrame, n_splits=5, seed=42):
    y_all = y_all.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    label_encoder_cv = LabelEncoder()
    label_encoder_cv.fit(y_all)
    y_all_enc = label_encoder_cv.transform(y_all)
    focus_ids_cv = [
        int(label_encoder_cv.transform([c])[0])
        for c in FOCUS_CLASSES if c in label_encoder_cv.classes_
    ]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all_enc), start=1):
        X_tr_fold = X_all.iloc[tr_idx].copy()
        X_va_fold = X_all.iloc[va_idx].copy()
        y_tr_fold = y_all.iloc[tr_idx]
        y_va_fold = y_all.iloc[va_idx]

        X_tr_enc, X_va_enc, _ = build_features(X_tr_fold, X_va_fold, X_test_ref)
        y_tr_enc = label_encoder_cv.transform(y_tr_fold)
        y_va_enc = label_encoder_cv.transform(y_va_fold)

        X_tr_bal, y_tr_bal = upsample_focus_classes(X_tr_enc, y_tr_enc, focus_ids_cv, seed=seed + fold_idx)

        fold_params, _ = tune_lgbm_bayes(
            X_tr_bal, y_tr_bal, X_va_enc, y_va_enc,
            focus_ids=focus_ids_cv, n_trials=20, n_init=6, seed=seed + fold_idx
        )

        model = LGBMClassifier(**fold_params)
        model.fit(to_lgbm_input(X_tr_bal), y_tr_bal)

        va_proba = model.predict_proba(to_lgbm_input(X_va_enc))
        fold_scales, _ = tune_class_scales(y_va_enc, va_proba, focus_ids_cv, rounds=4)
        va_pred = np.argmax(apply_class_scales(va_proba, fold_scales), axis=1)
        fold_blended = blend_score(y_va_enc, va_pred, focus_ids_cv)
        fold_scores.append(float(fold_blended))

        print(f"[CV] Fold {fold_idx}/{n_splits} blended score: {fold_blended:.6f}")

    if fold_scores:
        print(f"[CV] Mean blended score: {float(np.mean(fold_scores)):.6f}")
        print(f"[CV] Std blended score: {float(np.std(fold_scores)):.6f}")

    return fold_scores


def tune_class_scales_from_oof(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_test_ref: pd.DataFrame,
    best_params: dict,
    focus_ids: list[int],
    n_splits=5,
    seed=42,
):
    y_all = y_all.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)
    y_all_enc = LabelEncoder().fit_transform(y_all)
    n_classes = int(np.max(y_all_enc)) + 1
    oof_proba = np.zeros((len(X_all), n_classes), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all_enc), start=1):
        X_tr_fold = X_all.iloc[tr_idx].copy()
        X_va_fold = X_all.iloc[va_idx].copy()
        y_tr_fold_enc = y_all_enc[tr_idx]

        X_tr_enc, X_va_enc, _ = build_features(X_tr_fold, X_va_fold, X_test_ref)
        X_tr_bal, y_tr_bal = upsample_focus_classes(X_tr_enc, y_tr_fold_enc, focus_ids, seed=seed + fold_idx)

        model = LGBMClassifier(**best_params)
        model.fit(to_lgbm_input(X_tr_bal), y_tr_bal)
        oof_proba[va_idx] = model.predict_proba(to_lgbm_input(X_va_enc))

    scales, score = tune_class_scales(y_all_enc, oof_proba, focus_ids, rounds=4)
    return scales, float(score)


def main():
    start_time = time.time()

    train_csv = load_optional_clinical_csv("train")
    val_csv = load_optional_clinical_csv("val")
    test_csv = load_optional_clinical_csv("test")

    train_json = load_json_split("train")
    val_json = load_json_split("val")
    test_json = load_json_split("test")

    train_df = combine_split(train_csv, json_to_df(train_json))
    val_df = combine_split(val_csv, json_to_df(val_json))
    test_df = combine_split(test_csv, json_to_df(test_json))

    # Optional radiomics merge + ANOVA filtering (drop p > 0.05).
    train_rad = load_optional_radiomics_split(TRAIN_RADIOMICS_DIR, "train")
    val_rad = load_optional_radiomics_split(VAL_RADIOMICS_DIR, "val")
    test_rad = load_optional_radiomics_split(TEST_RADIOMICS_DIR, "test")
    if train_rad is not None:
        train_df = pd.merge(train_df, train_rad, on="case_id", how="left")
    if val_rad is not None:
        val_df = pd.merge(val_df, val_rad, on="case_id", how="left")
    if test_rad is not None:
        test_df = pd.merge(test_df, test_rad, on="case_id", how="left")

    selected_radiomics = select_radiomics_by_anova(train_df, p_threshold=0.05)
    all_radiomics = [c for c in train_df.columns if "__rad_" in c or c.startswith("rad_")]
    drop_radiomics = [c for c in all_radiomics if c not in set(selected_radiomics)]
    if drop_radiomics:
        train_df = train_df.drop(columns=drop_radiomics, errors="ignore")
        val_df = val_df.drop(columns=drop_radiomics, errors="ignore")
        test_df = test_df.drop(columns=drop_radiomics, errors="ignore")
    if all_radiomics:
        print(
            f"Radiomics ANOVA filter: kept {len(selected_radiomics)}/{len(all_radiomics)} features "
            f"(p <= 0.05)."
        )

    # Missingness policy for retained radiomics.
    RADIOMICS_MAX_MISSING_RATE = 0.40
    if selected_radiomics:
        too_sparse = [
            c for c in selected_radiomics
            if float(train_df[c].isna().mean()) > RADIOMICS_MAX_MISSING_RATE
        ]
        if too_sparse:
            train_df = train_df.drop(columns=too_sparse, errors="ignore")
            val_df = val_df.drop(columns=too_sparse, errors="ignore")
            test_df = test_df.drop(columns=too_sparse, errors="ignore")
            selected_radiomics = [c for c in selected_radiomics if c not in set(too_sparse)]
            print(
                f"Radiomics missingness filter: dropped {len(too_sparse)} columns "
                f"(missing > {RADIOMICS_MAX_MISSING_RATE:.0%})."
            )

        for c in selected_radiomics:
            if float(train_df[c].isna().mean()) > 0.10:
                if c not in val_df.columns:
                    val_df[c] = np.nan
                if c not in test_df.columns:
                    test_df[c] = np.nan
                ind = f"{c}__is_missing"
                train_df[ind] = train_df[c].isna().astype(int)
                val_df[ind] = val_df[c].isna().astype(int)
                test_df[ind] = test_df[c].isna().astype(int)

    train_df, val_df, test_df = align_columns_from_train(train_df, val_df, test_df)

    train = add_report_features(train_df)
    val = add_report_features(val_df)
    test = add_report_features(test_df)

    train = train.drop(columns=["Sex"], errors="ignore")
    val = val.drop(columns=["Sex"], errors="ignore")
    test = test.drop(columns=["Sex"], errors="ignore")
    train, val, test = fill_missing(train, val, test)

    X_train = train.drop(columns=["Overall_class"])
    y_train = train["Overall_class"]
    X_val = val.drop(columns=["Overall_class"])
    y_val = val["Overall_class"]
    X_test = test.copy()

    X_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_cv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    run_kfold_cv(
        X_cv,
        y_cv,
        X_test,
        n_splits=KFOLD_N_SPLITS,
        seed=KFOLD_RANDOM_STATE,
    )

    X_train_encoded, X_val_encoded, _ = build_features(X_train, X_val, X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train, y_val], axis=0))
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    focus_ids = [int(label_encoder.transform([c])[0]) for c in FOCUS_CLASSES if c in label_encoder.classes_]

    X_train_balanced, y_train_balanced = upsample_focus_classes(
        X_train_encoded, y_train_enc, focus_ids, seed=42
    )

    best_params, best_val_blended_hpo = tune_lgbm_bayes(
        X_train_balanced,
        y_train_balanced,
        X_val_encoded,
        y_val_enc,
        focus_ids=focus_ids,
        n_trials=40,
        n_init=10,
        seed=42,
    )
    print("Best Params From Bayes HPO:", best_params)

    lgbm = LGBMClassifier(**best_params)
    lgbm.fit(to_lgbm_input(X_train_balanced), y_train_balanced)

    val_proba = lgbm.predict_proba(to_lgbm_input(X_val_encoded))
    val_class_scales, val_tuned_blended_score = tune_class_scales(y_val_enc, val_proba, focus_ids, rounds=4)
    class_scales = val_class_scales
    best_val_blended_score = float(val_tuned_blended_score)
    X_oof = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_oof = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    oof_scales, oof_blended_score = tune_class_scales_from_oof(
        X_oof,
        y_oof,
        X_test,
        best_params=best_params,
        focus_ids=focus_ids,
        n_splits=OOF_SCALE_N_SPLITS,
        seed=KFOLD_RANDOM_STATE,
    )
    class_scales = oof_scales
    best_val_blended_score = float(oof_blended_score)
    print(
        f"OOF Scale Tuning Enabled: using OOF scales (score={oof_blended_score:.6f}, "
        f"val-tuned={val_tuned_blended_score:.6f})"
    )
    val_proba_used = val_proba

    val_pred = np.argmax(apply_class_scales(val_proba_used, class_scales), axis=1)
    val_pred_label = label_encoder.inverse_transform(val_pred)

    print("Validation Accuracy:", accuracy_score(y_val, val_pred_label))
    print("Validation Micro F1:", f1_score(y_val, val_pred_label, average="micro"))
    print("Validation Macro F1:", f1_score(y_val, val_pred_label, average="macro"))
    print("Validation Weighted F1:", f1_score(y_val, val_pred_label, average="weighted"))
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred_label))
    print("Best Validation Blended Score During HPO:", best_val_blended_hpo)
    print("Best Validation Blended Score After HPO + Scale Tuning:", best_val_blended_score)
    print("Best Validation Blended Score (Val-Tuned Scales):", float(val_tuned_blended_score))
    print("Best Validation Blended Score After Threshold Tuning:", best_val_blended_score)
    print("Final Validation Strategy: LGBM Only")
    best_class_scales = {
        label_encoder.inverse_transform([i])[0]: float(class_scales[i]) for i in focus_ids
    }
    print("Best Class Scales:", best_class_scales)

    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    X_full_tab = X_full.drop(columns=["report", "case_id"], errors="ignore")
    X_test_tab_final = X_test.drop(columns=["report", "case_id"], errors="ignore")
    full_tab_all = pd.concat([X_full_tab, X_test_tab_final], axis=0)
    full_tab_all = pd.get_dummies(full_tab_all, drop_first=False)
    X_full_tab_encoded = full_tab_all.iloc[: len(X_full_tab), :]
    X_test_tab_final_encoded = full_tab_all.iloc[len(X_full_tab):, :]

    tfidf_final_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2500)
    tfidf_final_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=1200)

    X_full_text_word = tfidf_final_word.fit_transform(X_full["report"])
    X_test_text_word_final = tfidf_final_word.transform(X_test["report"])
    X_full_text_char = tfidf_final_char.fit_transform(X_full["report"])
    X_test_text_char_final = tfidf_final_char.transform(X_test["report"])
    X_full_text = hstack([X_full_text_word, X_full_text_char], format="csr")
    X_test_text_final = hstack([X_test_text_word_final, X_test_text_char_final], format="csr")

    X_full_encoded = hstack([csr_matrix(X_full_tab_encoded.to_numpy(dtype="float32")), X_full_text], format="csr")
    X_test_final = hstack(
        [csr_matrix(X_test_tab_final_encoded.to_numpy(dtype="float32")), X_test_text_final],
        format="csr",
    )

    y_full_enc = label_encoder.transform(y_full)
    X_full_balanced, y_full_balanced = upsample_focus_classes(X_full_encoded, y_full_enc, focus_ids, seed=42)

    lgbm_final = LGBMClassifier(**best_params)
    lgbm_final.fit(to_lgbm_input(X_full_balanced), y_full_balanced)

    test_proba = lgbm_final.predict_proba(to_lgbm_input(X_test_final))

    test_pred = np.argmax(apply_class_scales(test_proba, class_scales), axis=1)
    test_pred_label = label_encoder.inverse_transform(test_pred)

    submission = build_submission(test["case_id"], test_pred_label, target_col="Overall_class")
    output_dir = ROOT / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print("\nGenerated predictions:", len(submission))
    print("Submission saved to:", submission_path)
    print(f"Overall running time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
