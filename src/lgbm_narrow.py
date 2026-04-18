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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

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
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
LATE_FUSION_TFIDF_WEIGHT = 0.60

# ============ 窄范围贝叶斯优化 ============
def bayes_optimize_narrow(objective, n_trials=30, n_init=8, seed=42):
    """窄范围贝叶斯优化 - 基于第一次运行的最佳参数"""
    
    # 窄范围 bounds（基于原最佳参数 376, 0.0906, 118, 17, 64, 0.896, 0.841, 9.38e-6, 0.0437）
    bounds = {
        "n_estimators": (300, 450),        # 原 376
        "lr_log10": (-1.2, -0.8),          # 原 lr=0.0906 → log10≈-1.04
        "num_leaves": (90, 125),           # 原 118
        "max_depth": (12, 18),             # 原 17
        "min_child_samples": (50, 70),     # 原 64
        "subsample": (0.80, 0.95),         # 原 0.896
        "colsample_bytree": (0.78, 0.92),  # 原 0.841
        "reg_alpha_log10": (-5.5, -3.5),   # 原 9.38e-6 → log10≈-5.03
        "reg_lambda_log10": (-1.8, -0.5),  # 原 0.0437 → log10≈-1.36
    }
    
    keys = list(bounds.keys())
    low = np.array([bounds[k][0] for k in keys])
    high = np.array([bounds[k][1] for k in keys])
    dim = len(keys)
    
    rng = np.random.default_rng(seed)
    x_hist = []
    y_hist = []
    best_score = -1.0
    best_params = None
    
    def _eval_point(x):
        nonlocal best_score, best_params
        params_dict = {k: float(v) for k, v in zip(keys, x)}
        score = objective(**params_dict)
        x_hist.append(np.asarray(x, dtype=np.float64))
        y_hist.append(score)
        if score > best_score:
            best_score = score
            best_params = params_dict.copy()
            print(f"    ✨ New best: {score:.4f} | n_est={params_dict['n_estimators']:.0f}, lr={10**params_dict['lr_log10']:.5f}, leaves={params_dict['num_leaves']:.0f}")
        return score
    
    # 初始随机采样
    print(f"  Initial random sampling ({n_init} trials)...")
    for _ in range(n_init):
        x0 = rng.uniform(low, high)
        _eval_point(x0)
    
    # 贝叶斯优化迭代
    print(f"  Bayesian optimization ({n_trials - n_init} iterations)...")
    for step in range(n_trials - n_init):
        x_arr = np.vstack(x_hist)
        y_arr = np.asarray(y_hist, dtype=np.float64)
        y_mean, y_std = np.mean(y_arr), np.std(y_arr)
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (y_arr - y_mean) / y_std
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, random_state=seed+step)
        
        try:
            gp.fit(x_arr, y_scaled)
            x_cand = rng.uniform(low, high, size=(256, dim))
            mu, sigma = gp.predict(x_cand, return_std=True)
            acq = mu + 2.0 * sigma
            x_next = x_cand[np.argmax(acq)]
        except:
            x_next = rng.uniform(low, high)
        
        _eval_point(x_next)
    
    return best_params, best_score


def normalize_lgbm_params(params):
    """将原始参数转换为 LGBM 可用的格式"""
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

# 默认参数（用于 fallback）
DEFAULT_LGBM_PARAMS = {
    'n_estimators': 376,
    'learning_rate': 0.0906,
    'num_leaves': 118,
    'max_depth': 17,
    'min_child_samples': 64,
    'subsample': 0.896,
    'colsample_bytree': 0.841,
    'reg_alpha': 9.38e-6,
    'reg_lambda': 0.0437,
    'objective': 'multiclass',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1,
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
    if col == "report":
        train[col] = train[col].fillna("")
        val[col] = val[col].fillna("")
        test[col] = test[col].fillna("")
        continue
    if train[col].dtype == "object":
        train[col] = train[col].fillna("Unknown")
        val[col] = val[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")
    elif pd.api.types.is_numeric_dtype(train[col]):
        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        val[col] = val[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)
    else:
        train[col] = train[col].fillna("Unknown")
        val[col] = val[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")

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
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=np.float32, copy=False)
    if hasattr(X, "astype"):
        return X.astype(np.float32, copy=False)
    return X

# score blending: 0.45 micro F1, 0.35 weighted F1, 0.20 focus recall F1
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

def _select_torch_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def compute_frozen_bert_embeddings(texts, model_name=BERT_MODEL_NAME, max_length=256, batch_size=16):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = _select_torch_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    texts = list(texts)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(all_embeddings)

def blend_probabilities(proba_tfidf, proba_bert, tfidf_weight=LATE_FUSION_TFIDF_WEIGHT):
    bert_weight = 1.0 - float(tfidf_weight)
    return (float(tfidf_weight) * proba_tfidf) + (bert_weight * proba_bert)

print("="*60)
print("实验3: 窄范围贝叶斯优化")
print("="*60)

# frozen ClinicalBERT embeddings
print("\n计算 BERT embeddings...")
X_train_bert = compute_frozen_bert_embeddings(X_train["report"])
X_val_bert = compute_frozen_bert_embeddings(X_val["report"])
X_test_bert = compute_frozen_bert_embeddings(X_test["report"])

X_train_bert_encoded = hstack(
    [csr_matrix(X_train_tab_encoded.to_numpy(dtype="float32")), csr_matrix(X_train_bert)],
    format="csr",
)
X_val_bert_encoded = hstack(
    [csr_matrix(X_val_tab_encoded.to_numpy(dtype="float32")), csr_matrix(X_val_bert)],
    format="csr",
)
X_test_bert_encoded = hstack(
    [csr_matrix(X_test_tab_encoded.to_numpy(dtype="float32")), csr_matrix(X_test_bert)],
    format="csr",
)

# call the upsampling for imbalanced dataset
print("\n上采样少数类...")
X_train_balanced, y_train_balanced = upsample_focus_classes(
    X_train_encoded, y_train_enc, focus_ids, seed=42
)
X_train_bert_balanced, y_train_bert_balanced = upsample_focus_classes(
    X_train_bert_encoded, y_train_enc, focus_ids, seed=42
)

# ============ 窄范围贝叶斯优化找最佳参数 ============
print("\n开始窄范围贝叶斯优化...")

def objective_for_opt(
    n_estimators, lr_log10, num_leaves, max_depth, min_child_samples,
    subsample, colsample_bytree, reg_alpha_log10, reg_lambda_log10
):
    params = normalize_lgbm_params({
        "n_estimators": n_estimators,
        "lr_log10": lr_log10,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha_log10": reg_alpha_log10,
        "reg_lambda_log10": reg_lambda_log10,
    })
    model = LGBMClassifier(**params)
    model.fit(to_lgbm_input(X_train_balanced), y_train_balanced)
    pred = model.predict(to_lgbm_input(X_val_encoded))
    return blend_score(y_val_enc, pred, focus_ids)

best_raw_params, best_score = bayes_optimize_narrow(objective_for_opt, n_trials=30, n_init=8, seed=42)
best_params = normalize_lgbm_params(best_raw_params)

print(f"\n✅ 优化完成！")
print(f"  最佳分数: {best_score:.4f}")
print(f"  最佳参数:")
for k, v in best_params.items():
    if k not in ['objective', 'class_weight', 'random_state', 'n_jobs', 'verbosity']:
        print(f"    {k}: {v}")

# now LGBM fit with optimized params
print("\n训练最终模型...")
lgbm = LGBMClassifier(**best_params)
lgbm.fit(to_lgbm_input(X_train_balanced), y_train_balanced)
lgbm_bert = LGBMClassifier(**best_params)
lgbm_bert.fit(to_lgbm_input(X_train_bert_balanced), y_train_bert_balanced)

# probability scaling
val_proba_tfidf = lgbm.predict_proba(to_lgbm_input(X_val_encoded))
val_proba_bert = lgbm_bert.predict_proba(to_lgbm_input(X_val_bert_encoded))
val_proba = blend_probabilities(val_proba_tfidf, val_proba_bert)
class_scales, best_val_blended_score = tune_class_scales(y_val_enc, val_proba, focus_ids, rounds=4)
val_pred = np.argmax(apply_class_scales(val_proba, class_scales), axis=1)
val_pred_label = label_encoder.inverse_transform(val_pred)

# print results
print("\n" + "="*60)
print("验证集结果:")
print("="*60)
print(f"Validation Accuracy: {accuracy_score(y_val, val_pred_label):.4f}")
print(f"Validation Micro F1: {f1_score(y_val, val_pred_label, average='micro'):.4f}")
print(f"Validation Macro F1: {f1_score(y_val, val_pred_label, average='macro'):.4f}")
print(f"Validation Weighted F1: {f1_score(y_val, val_pred_label, average='weighted'):.4f}")
print(f"Best Validation Blended Score: {best_val_blended_score:.4f}")
print("Class Scales:", {
    label_encoder.inverse_transform([i])[0]: float(class_scales[i]) for i in focus_ids
})

# train again on train + val with 5-fold OOF calibration
print("\n" + "="*60)
print("5-Fold CV Stacking...")
print("="*60)

X_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
y_full_enc = label_encoder.transform(y_full)

def build_fold_matrices(train_df, infer_df):
    train_tab = train_df.drop(columns=["report", "case_id"], errors="ignore").copy()
    infer_tab = infer_df.drop(columns=["report", "case_id"], errors="ignore").copy()

    train_tab = pd.get_dummies(train_tab, drop_first=False)
    infer_tab = pd.get_dummies(infer_tab, drop_first=False)
    infer_tab = infer_tab.reindex(columns=train_tab.columns, fill_value=0)

    tfidf_word_local = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=2500,
    )
    tfidf_char_local = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=1200,
    )

    X_train_text_word_local = tfidf_word_local.fit_transform(train_df["report"])
    X_infer_text_word_local = tfidf_word_local.transform(infer_df["report"])
    X_train_text_char_local = tfidf_char_local.fit_transform(train_df["report"])
    X_infer_text_char_local = tfidf_char_local.transform(infer_df["report"])
    X_train_text_local = hstack([X_train_text_word_local, X_train_text_char_local], format="csr")
    X_infer_text_local = hstack([X_infer_text_word_local, X_infer_text_char_local], format="csr")

    X_train_tab_local = csr_matrix(train_tab.to_numpy(dtype="float32"))
    X_infer_tab_local = csr_matrix(infer_tab.to_numpy(dtype="float32"))
    X_train_tfidf_final = hstack([X_train_tab_local, X_train_text_local], format="csr")
    X_infer_tfidf_final = hstack([X_infer_tab_local, X_infer_text_local], format="csr")
    return X_train_tfidf_final, X_infer_tfidf_final, X_train_tab_local, X_infer_tab_local

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(label_encoder.classes_)
oof_proba_tfidf = np.zeros((len(y_full_enc), n_classes), dtype=np.float32)
oof_proba_bert = np.zeros((len(y_full_enc), n_classes), dtype=np.float32)
test_proba_tfidf_folds = []
test_proba_bert_folds = []

print("计算完整数据集 BERT embeddings...")
X_full_bert = compute_frozen_bert_embeddings(X_full["report"])
X_test_bert_full = compute_frozen_bert_embeddings(X_test["report"])

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_full_enc)), y_full_enc), start=1):
    print(f"\n  Fold {fold}/5...")
    X_tr_df = X_full.iloc[tr_idx].reset_index(drop=True)
    X_va_df = X_full.iloc[va_idx].reset_index(drop=True)
    y_tr_fold = y_full_enc[tr_idx]
    y_va_fold = y_full_enc[va_idx]

    X_tr_fold_enc, X_va_fold_enc, X_tr_tab_local, X_va_tab_local = build_fold_matrices(X_tr_df, X_va_df)
    _, X_test_fold_enc, _, X_test_tab_local = build_fold_matrices(X_tr_df, X_test.reset_index(drop=True))
    X_tr_fold_bert_enc = hstack([X_tr_tab_local, csr_matrix(X_full_bert[tr_idx])], format="csr")
    X_va_fold_bert_enc = hstack([X_va_tab_local, csr_matrix(X_full_bert[va_idx])], format="csr")
    X_test_fold_bert_enc = hstack([X_test_tab_local, csr_matrix(X_test_bert_full)], format="csr")

    X_tr_fold_bal, y_tr_fold_bal = upsample_focus_classes(
        X_tr_fold_enc, y_tr_fold, focus_ids, seed=42 + fold
    )
    X_tr_fold_bert_bal, y_tr_fold_bert_bal = upsample_focus_classes(
        X_tr_fold_bert_enc, y_tr_fold, focus_ids, seed=42 + fold
    )

    lgbm_fold = LGBMClassifier(**best_params)
    lgbm_fold.fit(to_lgbm_input(X_tr_fold_bal), y_tr_fold_bal)
    lgbm_bert_fold = LGBMClassifier(**best_params)
    lgbm_bert_fold.fit(to_lgbm_input(X_tr_fold_bert_bal), y_tr_fold_bert_bal)

    va_proba_tfidf_fold = lgbm_fold.predict_proba(to_lgbm_input(X_va_fold_enc))
    va_proba_bert_fold = lgbm_bert_fold.predict_proba(to_lgbm_input(X_va_fold_bert_enc))
    oof_proba_tfidf[va_idx] = va_proba_tfidf_fold
    oof_proba_bert[va_idx] = va_proba_bert_fold

    test_proba_tfidf_folds.append(lgbm_fold.predict_proba(to_lgbm_input(X_test_fold_enc)))
    test_proba_bert_folds.append(lgbm_bert_fold.predict_proba(to_lgbm_input(X_test_fold_bert_enc)))

# Stacking
oof_stack_features = np.hstack([oof_proba_tfidf, oof_proba_bert]).astype(np.float32)
stacker = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=42,
)
stacker.fit(oof_stack_features, y_full_enc)
oof_proba = stacker.predict_proba(oof_stack_features)

class_scales_oof, best_oof_blended_score = tune_class_scales(
    y_full_enc, oof_proba, focus_ids, rounds=4
)
print(f"\nBest OOF Blended Score: {best_oof_blended_score:.4f}")

# Final test predictions
test_proba_tfidf = np.mean(np.stack(test_proba_tfidf_folds, axis=0), axis=0)
test_proba_bert = np.mean(np.stack(test_proba_bert_folds, axis=0), axis=0)
test_stack_features = np.hstack([test_proba_tfidf, test_proba_bert]).astype(np.float32)
test_proba = stacker.predict_proba(test_stack_features)
test_pred = np.argmax(apply_class_scales(test_proba, class_scales_oof), axis=1)
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
submission_path = output_dir / "submission_narrow.csv"
submission.to_csv(submission_path, index=False)

print("\n" + "="*60)
print(f"✅ 完成！")
print(f"  生成预测: {len(submission)} 行")
print(f"  保存到: {submission_path}")
print(f"  总运行时间: {time.time() - start_time:.2f} 秒")
print("="*60)