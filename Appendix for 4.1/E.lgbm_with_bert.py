import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

DATA_DIR = "/kaggle/input/competitions/2026-spring-sdst-stat-3612-group-project"
TRAIN_DIR = f"{DATA_DIR}/kaggle-dataset"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 256
BATCH_SIZE = 16

LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "objective": "multiclass",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
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

def compute_bert_embeddings(texts, model, tokenizer, device):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            pooled = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(pooled)
    return np.vstack(embeddings)

train_df = json_to_df(load_json("train"))
val_df = json_to_df(load_json("val"))

train_df["report"] = train_df["report"].fillna("").astype(str)
val_df["report"] = val_df["report"].fillna("").astype(str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Computing BERT embeddings...")
X_train = compute_bert_embeddings(train_df["report"].tolist(), model, tokenizer, device)
X_val = compute_bert_embeddings(val_df["report"].tolist(), model, tokenizer, device)

label_encoder = LabelEncoder()
label_encoder.fit(pd.concat([train_df["Overall_class"], val_df["Overall_class"]]))
y_train = label_encoder.transform(train_df["Overall_class"])
y_val = label_encoder.transform(val_df["Overall_class"])

model_lgbm = LGBMClassifier(**LGBM_PARAMS)
model_lgbm.fit(X_train, y_train)

val_pred = model_lgbm.predict(X_val)
print(f"Validation Macro F1: {f1_score(y_val, val_pred, average='macro'):.4f}")

test_df = json_to_df(load_json("test"))
test_df["report"] = test_df["report"].fillna("").astype(str)
X_test = compute_bert_embeddings(test_df["report"].tolist(), model, tokenizer, device)

test_pred = model_lgbm.predict(X_test)
test_pred_label = label_encoder.inverse_transform(test_pred)

submission = pd.DataFrame({"case_id": test_df["case_id"].astype(str), "Overall_class": test_pred_label})
submission.to_csv("submission_bert.csv", index=False)
print("Saved: submission_bert.csv")