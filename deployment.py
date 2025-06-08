from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
from huggingface_hub import login
import os

app = Flask(__name__)

# === Global var untuk model & tokenizer ===
sentiment_model = None
sentiment_tokenizer = None

# === DETEKSI DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = 0 if torch.cuda.is_available() else -1

# === Load model saat server mulai jalan ===
@app.before_first_request
def load_model():
    global sentiment_model, sentiment_tokenizer
    print("🔄 Loading IndoBERT model...")

    # Login ke HF pakai ENV VAR (jangan hardcode token!)
    login(token=os.environ.get("HF_TOKEN"))

    sentiment_tokenizer = AutoTokenizer.from_pretrained("siRendy/indobert-analisis-sentimen-review-produk-p3")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("siRendy/indobert-analisis-sentimen-review-produk-p3")
    sentiment_model.to(device)

    print("✅ Model loaded")

# === Fungsi prediksi ===
def predict_sentiment(text):
    classifier = pipeline(
        "text-classification",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=device_index
    )
    result = classifier(text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }

# === ROUTE API ===
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    result = predict_sentiment(text)
    return jsonify({
        "status": "success",
        "original_text": text,
        "label": result["sentiment"],
        "confidence": result["confidence"]
    })

@app.route("/")
def home():
    return "✅ Sentiment API using IndoBERT is ready!"

if __name__ == '__main__':
    app.run(port=5000)
