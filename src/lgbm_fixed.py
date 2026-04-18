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
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

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
start_time = time.time()

# path config
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CLINICAL_DIR = DATA_DIR / "clinical_information"
FOCUS_CLASSES = [
    "Brain Metastase Tumour",
    "Pineal tumour and Choroid plexus tumour",
    "Tumors of the sellar region",
]
FIXED_LGBM_PARAMS = {
    'n_estimators': 695, 
    'learning_rate': 0.09088841083294796, 
    'num_leaves': 33, 
    'max_depth': 12, 
    'min_child_samples': 30, 
    'subsample': 0.7950636550563394, 
    'colsample_bytree': 0.8758706053844969, 
    'reg_alpha': 9.381155722619392e-06, 
    'reg_lambda': 0.04374699982986275, 
    'objective': 'multiclass', 
    'class_weight': 'balanced', 
    'random_state': 42, 
    'n_jobs': -1, 
    'verbosity': -1
}
FIXED_CLASS_SCALES = {
    'Brain Metastase Tumour': 0.70 , 
    'Pineal tumour and Choroid plexus tumour': 1.0, 
    'Tumors of the sellar region': 1.0
}

# load json reports and split them
def load_json_split(name: str):
    with (DATA_DIR / f"{name}.json").open("r") as f:
        return json.load(f)

# load optional clinical information
def load_optional_clinical_csv(name: str):
    csv_path = CLINICAL_DIR / f"{name}_patient_info.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None

# define datasets
train_csv = load_optional_clinical_csv("train")
val_csv = load_optional_clinical_csv("val")
test_csv = load_optional_clinical_csv("test")

train_json = load_json_split("train")
val_json = load_json_split("val")
test_json = load_json_split("test")

# convert json to dataframe
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

# json-format dataframe
train_df_json = json_to_df(train_json)
val_df_json = json_to_df(val_json)
test_df_json = json_to_df(test_json)

# combine them and split
def combine_split(clinical_df: pd.DataFrame | None, json_df: pd.DataFrame):
    if clinical_df is None:
        return json_df.copy()
    return pd.merge(clinical_df, json_df, on="case_id", how="left")

# csv+json dataframe
train_df = combine_split(train_csv, train_df_json)
val_df = combine_split(val_csv, val_df_json)
test_df = combine_split(test_csv, test_df_json)

# feature augmentation
def add_report_features(df):
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

# feature augmented dfs
train = add_report_features(train_df)
val = add_report_features(val_df)
test = add_report_features(test_df)

# drop the Sex features
train = train.drop(columns=["Sex"], errors="ignore")
val = val.drop(columns=["Sex"], errors="ignore")
test = test.drop(columns=["Sex"], errors="ignore")

# fill missing values
for col in train.columns:
    if col == "Overall_class":
        continue
    # fill empty chars for report
    if col == "report":
        train[col] = train[col].fillna("")
        val[col] = val[col].fillna("")
        test[col] = test[col].fillna("")
        continue
    # fill 'unknown' for categorical features
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Unknown")
        val[col] = val[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")
    # take medians for the rest
    else:
        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        val[col] = val[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

# set the train/val/test sets
X_train = train.drop(columns=["Overall_class"])
y_train = train["Overall_class"]
X_val = val.drop(columns=["Overall_class"])
y_val = val["Overall_class"]
X_test = test.copy()

# encode the tabular features
tabular_cols = [c for c in X_train.columns if c not in ["report", "case_id"]]
X_train_tab = X_train[tabular_cols].copy()
X_val_tab = X_val[tabular_cols].copy()
X_test_tab = X_test[tabular_cols].copy()

# combine the tabular features in one dataframe
all_tab = pd.concat([X_train_tab, X_val_tab, X_test_tab], axis=0)
all_tab = pd.get_dummies(all_tab, drop_first=False)

X_train_tab_encoded = all_tab.iloc[: len(X_train_tab), :]
X_val_tab_encoded = all_tab.iloc[len(X_train_tab) : len(X_train_tab) + len(X_val_tab), :]
X_test_tab_encoded = all_tab.iloc[len(X_train_tab) + len(X_val_tab) :, :]

# tf-idf vectorization - 'word' layer
tfidf_word = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=2500,
)
# tf-idf vectorization - 'char' layer
tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=1200,
)

# vectorize the reports
X_train_text_word = tfidf_word.fit_transform(X_train["report"])
X_val_text_word = tfidf_word.transform(X_val["report"])
X_test_text_word = tfidf_word.transform(X_test["report"])
X_train_text_char = tfidf_char.fit_transform(X_train["report"])
X_val_text_char = tfidf_char.transform(X_val["report"])
X_test_text_char = tfidf_char.transform(X_test["report"])
X_train_text = hstack([X_train_text_word, X_train_text_char], format="csr")
X_val_text = hstack([X_val_text_word, X_val_text_char], format="csr")
X_test_text = hstack([X_test_text_word, X_test_text_char], format="csr")

# combine the tabular features and the vectorized reports
X_train_encoded = hstack(
    [csr_matrix(X_train_tab_encoded.to_numpy(dtype="float32")), X_train_text], 
    format="csr"
)
X_val_encoded = hstack(
    [csr_matrix(X_val_tab_encoded.to_numpy(dtype="float32")), X_val_text], 
    format="csr"
)
X_test_encoded = hstack(
    [csr_matrix(X_test_tab_encoded.to_numpy(dtype="float32")), X_test_text], 
    format="csr"
)

# final encoding of the labels
label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([y_train, y_val], axis=0))
y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
focus_ids = [
    int(label_encoder.transform([c])[0]) 
    for c in FOCUS_CLASSES if c in label_encoder.classes_
]


# convert the input to a consistent format
def to_lgbm_input(X):
    # Keep a consistent no-feature-name representation across fit/predict.
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    if hasattr(X, "astype"):
        return X.astype(np.float32, copy=False)
    return X

# score blending: 0.45 weighted F1, 0.35 macro F1, 0.20 focus recall F1
def blend_score(y_true, y_pred, focus_ids):
    micro = float(f1_score(y_true, y_pred, average="micro"))
    weighted = float(f1_score(y_true, y_pred, average="weighted"))
    if len(focus_ids) == 0:
        focus_recall = weighted
    else:
        recalls = recall_score(
            y_true, y_pred, 
            labels=focus_ids, 
            average=None, 
            zero_division=0
        )
        focus_recall = float(np.mean(recalls))
    return 0.45 * micro + 0.35 * weighted + 0.20 * focus_recall

# upsample the focus classes - mainly the minority classes
def upsample_focus_classes(X, y, focus_ids, seed=42):
    if len(focus_ids) == 0:
        return X, y

    # rng random sampling
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

# scale the class probabilities
def apply_class_scales(proba, scales):
    adjusted = proba * scales.reshape(1, -1)
    denom = adjusted.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return adjusted / denom

# tuning based on the minority classes
def tune_class_scales(y_true, proba, focus_ids, rounds=4):
    n_classes = proba.shape[1]
    if len(focus_ids) == 0:
        return (
            np.ones(n_classes, dtype=np.float64), 
            blend_score(y_true, np.argmax(proba, axis=1), focus_ids)
        )
    scales = np.ones(n_classes, dtype=np.float64)
    best_score = blend_score(y_true, np.argmax(proba, axis=1), focus_ids)

    # grid search
    grid = np.unique(
        np.concatenate([
            np.linspace(0.5, 3.0, 26), 
            np.array([1.15, 1.35, 1.65, 2.25, 2.75])
        ])
    )

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


# call the upsampling for imbalanced dataset
X_train_balanced, y_train_balanced = upsample_focus_classes(
    X_train_encoded, y_train_enc, focus_ids, seed=42
)

# now LGBM fit (fixed params, no Bayesian HPO)
best_params = FIXED_LGBM_PARAMS.copy()

lgbm = LGBMClassifier(**best_params)
lgbm.fit(to_lgbm_input(X_train_balanced), y_train_balanced)

# probability scaling
val_proba = lgbm.predict_proba(to_lgbm_input(X_val_encoded))
class_scales, best_val_blended_score = tune_class_scales(y_val_enc, val_proba, focus_ids, rounds=4)
val_pred = np.argmax(apply_class_scales(val_proba, class_scales), axis=1)
val_pred_label = label_encoder.inverse_transform(val_pred)

# print results
print("Validation Accuracy:", accuracy_score(y_val, val_pred_label))
print("Validation Micro F1:", f1_score(y_val, val_pred_label, average="micro"))
print("Validation Macro F1:", f1_score(y_val, val_pred_label, average="macro"))
print("Validation Weighted F1:", f1_score(y_val, val_pred_label, average="weighted"))
print("\nClassification Report:")
print(classification_report(y_val, val_pred_label))
print("Best Validation Blended Score After Threshold Tuning:", best_val_blended_score)
print("Class Scales:", {
    label_encoder.inverse_transform([i])[0]: float(class_scales[i]) for i in focus_ids
})

# train again on train + val and infer test
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)
X_full_tab = X_full.drop(columns=["report", "case_id"], errors="ignore")
X_test_tab_final = X_test.drop(columns=["report", "case_id"], errors="ignore")
full_tab_all = pd.concat([X_full_tab, X_test_tab_final], axis=0)
full_tab_all = pd.get_dummies(full_tab_all, drop_first=False)
X_full_tab_encoded = full_tab_all.iloc[: len(X_full_tab), :]
X_test_tab_final_encoded = full_tab_all.iloc[len(X_full_tab):, :]

tfidf_final_word = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=2500,
)
tfidf_final_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    max_features=1200,
)

X_full_text_word = tfidf_final_word.fit_transform(X_full["report"])
X_test_text_word_final = tfidf_final_word.transform(X_test["report"])
X_full_text_char = tfidf_final_char.fit_transform(X_full["report"])
X_test_text_char_final = tfidf_final_char.transform(X_test["report"])
X_full_text = hstack([X_full_text_word, X_full_text_char], format="csr")
X_test_text_final = hstack([X_test_text_word_final, X_test_text_char_final], format="csr")

X_full_encoded = hstack(
    [csr_matrix(X_full_tab_encoded.to_numpy(dtype="float32")), X_full_text],
    format="csr",
)
X_test_final = hstack(
    [csr_matrix(X_test_tab_final_encoded.to_numpy(dtype="float32")), X_test_text_final],
    format="csr",
)
y_full_enc = label_encoder.transform(y_full)
X_full_balanced, y_full_balanced = upsample_focus_classes(
    X_full_encoded, y_full_enc, focus_ids, seed=42
)

lgbm_final = LGBMClassifier(**best_params)
lgbm_final.fit(to_lgbm_input(X_full_balanced), y_full_balanced)

test_proba = lgbm_final.predict_proba(to_lgbm_input(X_test_final))
test_pred = np.argmax(apply_class_scales(test_proba, class_scales), axis=1)
test_pred_label = label_encoder.inverse_transform(test_pred)

# final submission frame
submission = pd.DataFrame(
    {
        "case_id": test["case_id"],
        "Overall_class": test_pred_label,
    }
)

output_dir = ROOT / "submissions"
output_dir.mkdir(parents=True, exist_ok=True)
submission_path = output_dir / "submission.csv"
submission.to_csv(submission_path, index=False)

print("\nGenerated predictions:", len(submission))
print(f"Saved: {submission_path}")
end_time = time.time()
print(f"Overall running time: {end_time - start_time:.2f} seconds")
