# Cyberbullying Detection Model Service

A Python-based inference API for detecting cyberbullying in text, designed to be consumed by a Node.js application.

## Model Details

- **Algorithm**: Random Forest Classifier (n_estimators=40)
- **Features**: TF-IDF vectorization (max_features=5000) on preprocessed text
- **Performance**: ~84% accuracy, ~0.84 F1 score (weighted)
- **Classes**: `Non-Bullying` (0), `Bullying` (1)

This was selected as the most optimal and reliable model from multiple ML/DL models evaluated in the notebook (MLP, Random Forest, LSTM, GRU, Bi-LSTM, CNN). Random Forest was chosen over the deep learning models because:
1. Nearly identical top-tier performance to the best DL model
2. Deterministic and reproducible results
3. No GPU/TensorFlow dependency in production
4. Fast inference, easy serialization

## Setup

```bash
cd cyberbully-model

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Train and export model artifacts
python train_and_export.py

# Start the inference server
python server.py --port 5000
```

## API Usage

### Health Check
```
GET http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are so ugly and stupid"}'
```

Response:
```json
{
  "results": [
    {
      "text": "you are so ugly and stupid",
      "label": "Bullying",
      "confidence": 0.92
    }
  ]
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["you are so ugly", "have a great day!"]}'
```

Response:
```json
{
  "results": [
    {
      "text": "you are so ugly",
      "label": "Bullying",
      "confidence": 0.89
    },
    {
      "text": "have a great day!",
      "label": "Non-Bullying",
      "confidence": 0.95
    }
  ]
}
```

## Using from Node.js

```javascript
const axios = require('axios');

async function detectBullying(text) {
  const response = await axios.post('http://localhost:5000/predict', { text });
  return response.data.results[0];
}

// Single sentence
const result = await detectBullying("you are worthless");
console.log(result);
// { text: "you are worthless", label: "Bullying", confidence: 0.87 }

// Multiple sentences
const response = await axios.post('http://localhost:5000/predict', {
  texts: ["I hate you", "Nice work on the project!"]
});
console.log(response.data.results);
```

## Deploy to Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo (`supratikkarmakr/CyberShield`)
4. Configure:
   - **Build Command**: `bash build.sh`
   - **Start Command**: `gunicorn server:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click **Create Web Service** — Render installs deps, downloads spaCy model, and starts gunicorn
6. Visit your `https://cybershield-xxxx.onrender.com` URL

Alternatively, Render auto-detects settings from the included `render.yaml` Blueprint.

## File Structure

```
cyberbully-model/
├── server.py             # Flask inference API + serves frontend
├── train_and_export.py   # Train model & save artifacts
├── requirements.txt      # Pinned Python dependencies
├── build.sh              # Render build script (deps + spaCy + NLTK)
├── render.yaml           # Render Blueprint config
├── Procfile              # Railway/Heroku start command
├── railway.toml          # Railway deploy config
├── .gitignore
├── README.md
├── static/
│   └── index.html        # CyberShield frontend (served at /)
└── model/                # Created by train_and_export.py
    ├── rf_model.joblib
    ├── tfidf_vectorizer.joblib
    └── label_map.json
```
