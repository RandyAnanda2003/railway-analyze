from flask import Flask, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    pipeline
)
import torch
from huggingface_hub import login
import pandas as pd
import numpy as np
import warnings
import os

app = Flask(__name__)

# === Login HF ===
login(token=os.environ.get("HF_TOKEN"))

# === DETEKSI DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = 0 if torch.cuda.is_available() else -1  # Untuk pipeline()

# === Load model untuk analisis sentimen ===
sentiment_tokenizer = AutoTokenizer.from_pretrained("siRendy/indobert-analisis-sentimen-review-produk-p3")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("siRendy/indobert-analisis-sentimen-review-produk-p3")
sentiment_model.to(device)  # <- Pindahkan model ke device

# === Fungsi untuk prediksi sentimen ===
def predict_sentiment(text):
    classifier = pipeline(
        "text-classification",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        device=device_index  # HARUS int: 0 (GPU) atau -1 (CPU)
    )
    result = classifier(text)[0]
    return {
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }

    
# === ROUTES =======================================================================================

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

if __name__ == '__main__':
    app.run(port=5000)
