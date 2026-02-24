"""
Flask inference server for cyberbullying detection.

Endpoints:
  POST /predict
    Body (JSON):
      { "text": "some sentence" }
      or
      { "texts": ["sentence 1", "sentence 2", ...] }

    Response (JSON):
      {
        "results": [
          {
            "text": "some sentence",
            "label": "Bullying",
            "confidence": 0.87
          }
        ]
      }

Usage:
  python server.py          # starts on port 5000
  python server.py --port 8080
"""

import argparse
import json
import os
import re
import sys

import joblib
import nltk
import spacy
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

nlp = spacy.load("en_core_web_sm")

stop_words_set = set(stopwords.words("english"))
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
stop_words_list = [w for w in stop_words_set if w not in KEEP_WORDS]


def preprocess_text(content: str) -> str:
    content = content.lower()
    content = re.sub(r"http\S+|www\S+|https\S+", "", content, flags=re.MULTILINE)
    content = re.sub(r"@\w+", "", content)
    content = re.sub(r"[^a-zA-Z\s]", "", content)
    tokens = [
        token.lemma_
        for token in nlp(content)
        if token.text not in stop_words_list and len(token) > 1
    ]
    return " ".join(tokens)


def load_model():
    model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    label_path = os.path.join(MODEL_DIR, "label_map.json")

    for p in [model_path, vec_path, label_path]:
        if not os.path.exists(p):
            print(f"ERROR: Missing artifact: {p}")
            print("Run `python train_and_export.py` first.")
            sys.exit(1)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    with open(label_path) as f:
        label_map = json.load(f)

    return model, vectorizer, label_map


model, vectorizer, label_map = load_model()


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    texts = data.get("texts") or []
    single = data.get("text")
    if single:
        texts = [single]

    if not texts:
        return jsonify({"error": "Provide 'text' or 'texts' in the request body."}), 400

    cleaned = [preprocess_text(t) for t in texts]
    features = vectorizer.transform(cleaned)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []
    for i, text in enumerate(texts):
        pred_label = int(predictions[i])
        confidence = float(probabilities[i][pred_label])
        results.append({
            "text": text,
            "label": label_map.get(str(pred_label), str(pred_label)),
            "confidence": round(confidence, 4),
        })

    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForest-TfIdf"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting cyberbullying detection API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
