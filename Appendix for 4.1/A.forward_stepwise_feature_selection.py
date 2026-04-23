import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "/kaggle/input/competitions/2026-spring-sdst-stat-3612-group-project"
TRAIN_DIR = f"{DATA_DIR}/kaggle-dataset"

def load_json(name):
    with open(f"{TRAIN_DIR}/{name}.json") as f:
        return json.load(f)

def json_to_df(data):
    rows = []
    for case_id, info in data.items():
        report = info.get("report", "")
        if isinstance(report, dict):
            report = report.get("finding", "")
        rows.append({"case_id": int(case_id), "Overall_class": info.get("Overall_class"), "report": report})
    return pd.DataFrame(rows)

def add_features(df):
    df["report"] = df["report"].fillna("").astype(str).str.lower()
    df["has_enhancement"] = df["report"].str.contains("enhancement").astype(int)
    df["has_edema"] = df["report"].str.contains("edema").astype(int)
    df["has_hydrocephalus"] = df["report"].str.contains("hydrocephalus").astype(int)
    df["has_pineal"] = df["report"].str.contains("pineal").astype(int)
    df["has_sellar"] = df["report"].str.contains("sellar|suprasellar|pituitary").astype(int)
    df["has_ventricular"] = df["report"].str.contains("ventricle|ventricular").astype(int)
    return df

def build_features(train, val):
    tab_cols = [c for c in train.columns if c not in ["report", "case_id"]]
    train_tab, val_tab = train[tab_cols], val[tab_cols]
    all_tab = pd.concat([train_tab, val_tab])
    all_tab = pd.get_dummies(all_tab, drop_first=False)
    train_tab_enc = all_tab.iloc[:len(train_tab)]
    val_tab_enc = all_tab.iloc[len(train_tab):]
    
    tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=2500)
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=1200)
    
    train_word = tfidf_word.fit_transform(train["report"])
    val_word = tfidf_word.transform(val["report"])
    train_char = tfidf_char.fit_transform(train["report"])
    val_char = tfidf_char.transform(val["report"])
    
    train_text = hstack([train_word, train_char])
    val_text = hstack([val_word, val_char])
    
    train_enc = hstack([csr_matrix(train_tab_enc.to_numpy(dtype="float32")), train_text])
    val_enc = hstack([csr_matrix(val_tab_enc.to_numpy(dtype="float32")), val_text])
    
    return train_enc, val_enc

def evaluate_features(X, y, indices, cv_folds=3):
    if not indices:
        return 1.0 / len(np.unique(y))
    X_sub = X[:, indices]
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_sub, y):
        model = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbosity=-1)
        model.fit(X_sub[train_idx], y[train_idx])
        pred = model.predict(X_sub[val_idx])
        scores.append(f1_score(y[val_idx], pred, average='macro'))
    return np.mean(scores)

train_df = json_to_df(load_json("train"))
val_df = json_to_df(load_json("val"))

train = add_features(train_df)
val = add_features(val_df)

for col in train.columns:
    if col == "Overall_class":
        continue
    if col == "report":
        train[col] = train[col].fillna("")
        val[col] = val[col].fillna("")
    elif train[col].dtype == "object":
        train[col] = train[col].fillna("Unknown")
        val[col] = val[col].fillna("Unknown")
    else:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        val[col] = val[col].fillna(med)

X_train_raw = train.drop(columns=["Overall_class"])
y_train = train["Overall_class"]
X_val_raw = val.drop(columns=["Overall_class"])
y_val = val["Overall_class"]

X_train_enc, X_val_enc = build_features(X_train_raw, X_val_raw)

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([y_train, y_val]))
y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)

X_full = hstack([X_train_enc, X_val_enc])
y_full = np.concatenate([y_train_enc, y_val_enc])

n_features = X_full.shape[1]
selected = []
remaining = list(range(n_features))
best_score = 1.0 / len(np.unique(y_full))

print("Forward stepwise selection:")
for step in range(30):
    best_imp = 0
    best_idx = None
    for idx in remaining[:200]:
        candidate = selected + [idx]
        score = evaluate_features(X_full, y_full, candidate)
        imp = score - best_score
        if imp > best_imp:
            best_imp = imp
            best_idx = idx
            best_candidate_score = score
    if best_imp <= 0.001:
        break
    selected.append(best_idx)
    remaining.remove(best_idx)
    best_score = best_candidate_score
    print(f"Step {step+1}: added feature {best_idx}, score={best_score:.4f}")

print(f"Selected {len(selected)} features, final score={best_score:.4f}")