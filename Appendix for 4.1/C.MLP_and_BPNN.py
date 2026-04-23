import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

DATA_DIR = "/kaggle/input/competitions/2026-spring-sdst-stat-3612-group-project"
TRAIN_DIR = f"{DATA_DIR}/kaggle-dataset"

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 64,
    "learning_rate": "adaptive",
    "max_iter": 200,
    "random_state": 42,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

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

def build_features(train, val):
    tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=2500)
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=1200)
    
    train_word = tfidf_word.fit_transform(train["report"])
    val_word = tfidf_word.transform(val["report"])
    train_char = tfidf_char.fit_transform(train["report"])
    val_char = tfidf_char.transform(val["report"])
    
    train_text = hstack([train_word, train_char])
    val_text = hstack([val_word, val_char])
    
    return train_text, val_text

train_df = json_to_df(load_json("train"))
val_df = json_to_df(load_json("val"))

train_df["report"] = train_df["report"].fillna("").astype(str).str.lower()
val_df["report"] = val_df["report"].fillna("").astype(str).str.lower()

X_train, X_val = build_features(train_df, val_df)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["Overall_class"])
y_val = label_encoder.transform(val_df["Overall_class"])

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = MLPClassifier(**MLP_PARAMS)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print(f"Validation Macro F1: {f1_score(y_val, val_pred, average='macro'):.4f}")