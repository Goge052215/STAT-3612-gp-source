import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

DATA_DIR = "/kaggle/input/competitions/2026-spring-sdst-stat-3612-group-project"
TRAIN_DIR = f"{DATA_DIR}/kaggle-dataset"
TEST_DIR = f"{DATA_DIR}/new_test"

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

def load_json(name):
    with open(f"{TRAIN_DIR if name != 'test' else TEST_DIR}/{name}.json") as f:
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
    return df

def build_features(train, val, test):
    tab_cols = [c for c in train.columns if c not in ["report", "case_id"]]
    train_tab, val_tab, test_tab = train[tab_cols], val[tab_cols], test[tab_cols]
    all_tab = pd.concat([train_tab, val_tab, test_tab])
    all_tab = pd.get_dummies(all_tab, drop_first=False)
    train_tab_enc = all_tab.iloc[:len(train_tab)]
    val_tab_enc = all_tab.iloc[len(train_tab):len(train_tab)+len(val_tab)]
    test_tab_enc = all_tab.iloc[len(train_tab)+len(val_tab):]
    
    tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=2500)
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=1200)
    
    train_word = tfidf_word.fit_transform(train["report"])
    val_word = tfidf_word.transform(val["report"])
    test_word = tfidf_word.transform(test["report"])
    train_char = tfidf_char.fit_transform(train["report"])
    val_char = tfidf_char.transform(val["report"])
    test_char = tfidf_char.transform(test["report"])
    
    train_text = hstack([train_word, train_char])
    val_text = hstack([val_word, val_char])
    test_text = hstack([test_word, test_char])
    
    train_enc = hstack([csr_matrix(train_tab_enc.to_numpy(dtype="float32")), train_text])
    val_enc = hstack([csr_matrix(val_tab_enc.to_numpy(dtype="float32")), val_text])
    test_enc = hstack([csr_matrix(test_tab_enc.to_numpy(dtype="float32")), test_text])
    
    return train_enc, val_enc, test_enc

def select_top_features(X, y, n=200):
    temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    temp.fit(X, y)
    importances = temp.feature_importances_
    return np.argsort(importances)[-n:]

train_df = json_to_df(load_json("train"))
val_df = json_to_df(load_json("val"))
test_df = json_to_df(load_json("test"))

train = add_features(train_df)
val = add_features(val_df)
test = add_features(test_df)

for col in train.columns:
    if col == "Overall_class":
        continue
    if col == "report":
        train[col] = train[col].fillna("")
        val[col] = val[col].fillna("")
        test[col] = test[col].fillna("")
    elif train[col].dtype == "object":
        train[col] = train[col].fillna("Unknown")
        val[col] = val[col].fillna("Unknown")
        test[col] = test[col].fillna("Unknown")
    else:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        val[col] = val[col].fillna(med)
        test[col] = test[col].fillna(med)

X_train_raw = train.drop(columns=["Overall_class"])
y_train = train["Overall_class"]
X_val_raw = val.drop(columns=["Overall_class"])
y_val = val["Overall_class"]
X_test_raw = test.copy()

X_train_enc, X_val_enc, X_test_enc = build_features(X_train_raw, X_val_raw, X_test_raw)

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([y_train, y_val]))
y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)

top_idx = select_top_features(X_train_enc, y_train_enc, n=200)

X_train_sel = X_train_enc[:, top_idx]
X_val_sel = X_val_enc[:, top_idx]
X_test_sel = X_test_enc[:, top_idx]

model = RandomForestClassifier(**RF_PARAMS)
model.fit(X_train_sel, y_train_enc)

val_pred = model.predict(X_val_sel)
print(f"Validation Macro F1: {f1_score(y_val_enc, val_pred, average='macro'):.4f}")

test_pred = model.predict(X_test_sel)
test_pred_label = label_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({"case_id": test["case_id"].astype(str), "Overall_class": test_pred_label})
submission.to_csv("submission_rf.csv", index=False)
print("Saved: submission_rf.csv")