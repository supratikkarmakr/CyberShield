"""
Train and export the best-performing cyberbullying detection model.

Uses Random Forest with TF-IDF vectorization -- the most reliable and
optimal model from the notebook experiments:
  - Accuracy: ~84%
  - F1 Score: ~0.84

Artifacts saved:
  - model/rf_model.joblib       (trained Random Forest classifier)
  - model/tfidf_vectorizer.joblib (fitted TF-IDF vectorizer)
  - model/label_map.json        (label-to-name mapping)
"""

import json
import os
import re

import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words("english"))
KEEP_WORDS = {
    "not", "until", "against", "up", "down", "no", "nor",
    "aren't", "couldn", "couldn't", "didn", "didn't",
    "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn",
    "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't", "don't",
}
stop_words = [w for w in stop_words if w not in KEEP_WORDS]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset.csv")


def preprocess_text(content: str) -> str:
    content = content.lower()
    content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)
    content = re.sub(r"@\w+", "", content)
    content = re.sub(r"[^a-zA-Z\s]", "", content)
    tokens = [
        token.lemma_
        for token in nlp(content)
        if token.text not in stop_words and len(token) > 1
    ]
    return " ".join(tokens)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Rows: {len(df)}")

    print("Preprocessing text...")
    df["clean_content"] = df["content"].apply(preprocess_text)

    X = df["clean_content"]
    y = df["label"]

    print("Vectorizing with TF-IDF (max_features=5000)...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest (n_estimators=40, random_state=42)...")
    rf = RandomForestClassifier(n_estimators=40, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"  F1 Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Bullying", "Bullying"]))

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    label_path = os.path.join(MODEL_DIR, "label_map.json")

    joblib.dump(rf, model_path)
    joblib.dump(tfidf, vec_path)

    label_map = {0: "Non-Bullying", 1: "Bullying"}
    with open(label_path, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved to        : {model_path}")
    print(f"Vectorizer saved to   : {vec_path}")
    print(f"Label map saved to    : {label_path}")
    print("\nDone! You can now run `python server.py` to start the inference API.")


if __name__ == "__main__":
    main()
