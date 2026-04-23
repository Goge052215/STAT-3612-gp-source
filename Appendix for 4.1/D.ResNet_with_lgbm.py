import json
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import os

DATA_DIR = "/kaggle/input/competitions/2026-spring-sdst-stat-3612-group-project"
TRAIN_DIR = f"{DATA_DIR}/kaggle-dataset"
TEST_DIR = f"{DATA_DIR}/new_test"

LGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.08,
    "num_leaves": 45,
    "max_depth": 12,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.001,
    "reg_lambda": 0.1,
    "objective": "multiclass",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

MODALITY_MAP = {"ax t1": "ax_t1", "ax t1c+": "ax_t1c", "ax t2": "ax_t2", "ax t2f": "ax_t2f"}
MODALITIES = ["ax t1", "ax t1c+", "ax t2", "ax t2f"]

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

def load_resnet(data_dir, case_id, modality):
    mapped = MODALITY_MAP.get(modality, modality.replace(" ", "_"))
    path = os.path.join(data_dir, "image_features", "image_features", str(case_id), mapped, "image.npy")
    if os.path.exists(path):
        return np.load(path)
    return np.zeros(2048)

def build_resnet_matrix(df, data_dir):
    features = []
    for _, row in df.iterrows():
        case_features = []
        for mod in MODALITIES:
            case_features.append(load_resnet(data_dir, row["case_id"], mod))
        features.append(np.concatenate(case_features))
    return np.vstack(features)

def build_text_features(train, val, test):
    tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=2500)
    tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2, max_features=1200)
    
    train_word = tfidf_word.fit_transform(train["report"])
    val_word = tfidf_word.transform(val["report"])
    test_word = tfidf_word.transform(test["report"])
    train_char = tfidf_char.fit_transform(train["report"])
    val_char = tfidf_char.transform(val["report"])
    test_char = tfidf_char.transform(test["report"])
    
    return hstack([train_word, train_char]), hstack([val_word, val_char]), hstack([test_word, test_char])

train_df = json_to_df(load_json("train"))
val_df = json_to_df(load_json("val"))
test_df = json_to_df(load_json("test"))

train_df["report"] = train_df["report"].fillna("").astype(str).str.lower()
val_df["report"] = val_df["report"].fillna("").astype(str).str.lower()
test_df["report"] = test_df["report"].fillna("").astype(str).str.lower()

X_text_train, X_text_val, X_text_test = build_text_features(train_df, val_df, test_df)

resnet_train = build_resnet_matrix(train_df, TRAIN_DIR)
resnet_val = build_resnet_matrix(val_df, TRAIN_DIR)
resnet_test = build_resnet_matrix(test_df, TEST_DIR)

X_train = hstack([X_text_train, csr_matrix(resnet_train)])
X_val = hstack([X_text_val, csr_matrix(resnet_val)])
X_test = hstack([X_text_test, csr_matrix(resnet_test)])

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_df["Overall_class"], val_df["Overall_class"]]))
y_train = label_encoder.transform(train_df["Overall_class"])
y_val = label_encoder.transform(val_df["Overall_class"])
y_test = label_encoder.transform(test_df["Overall_class"]) if "Overall_class" in test_df else None

model = LGBMClassifier(**LGBM_PARAMS)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
print(f"Validation Macro F1: {f1_score(y_val, val_pred, average='macro'):.4f}")

test_pred = model.predict(X_test)
test_pred_label = label_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({"case_id": test_df["case_id"].astype(str), "Overall_class": test_pred_label})
submission.to_csv("submission_resnet.csv", index=False)
print("Saved: submission_resnet.csv")