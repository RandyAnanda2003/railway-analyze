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
from datetime import timedelta
from xgboost import XGBRegressor
from datetime import timedelta
import google.generativeai as genai
import warnings
import math
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

# === Load model untuk summarization ===
summarization_tokenizer = AutoTokenizer.from_pretrained("siRendy/indoT5-sumarisasi-review-produk-p2")
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("siRendy/indoT5-sumarisasi-review-produk-p2")
summarization_model.to(device)  # <- Pindahkan model ke device

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

# === Fungsi untuk ringkasan teks ===
def summarize_text(text, model, tokenizer, num_beams=25, no_repeat_ngram_size=2):
    inputs = tokenizer.encode(
        f"ringkaslah: {text}", return_tensors="pt", max_length=1400, truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs,
        max_length=270,
        min_length=100,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=2.0,
        length_penalty=1.5,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# === Fungsi untuk sales forecasting ===

warnings.filterwarnings("ignore")
WINDOW = 30
FORECAST_STEP = 7  # Predict next 7 days from each window

def create_windows(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - FORECAST_STEP + 1):
        X.append(data[i:i + window_size, 0])
        y.append(data[i + window_size:i + window_size + FORECAST_STEP, 0])
    return np.array(X), np.array(y)

def forecast_future(model, last_window, steps):
    preds = []
    window = last_window.copy()
    # We need to predict in chunks of FORECAST_STEP days
    for _ in range(0, steps, FORECAST_STEP):
        current_steps = min(FORECAST_STEP, steps - len(preds))
        # Predict next FORECAST_STEP days
        pred = model.predict(window.reshape(1, -1))[0][:current_steps]
        preds.extend(pred)
        # Update window with new predictions
        window = np.roll(window, -current_steps)
        window[-current_steps:] = pred
    return np.array(preds)

def format_rupiah(value):
    return "Rp. {:,.0f}".format(value).replace(",", ".")

def get_gemini_analysis(historical_df, predictions_json):
    # Format historical data
    historical_md = historical_df.reset_index()[['date', 'total_sales']] \
                                  .rename(columns={'total_sales': 'actual_sales'}) \
                                  .to_markdown(index=False)
    
    # Format future predictions
    future_dates = list(predictions_json.keys())
    future_sales = [item['total_sales'] for item in predictions_json.values()]
    future_df = pd.DataFrame({'date': future_dates, 'predicted_sales': future_sales})
    future_md = future_df.to_markdown(index=False)

    data_str = f"=== Data Historis ===\n{historical_md}\n\n=== Prediksi 30 Hari ===\n{future_md}"

    prompt = f"""Anda adalah analis bisnis e-commerce profesional. **JANGAN BANYAK BASA BASI. tolong buat paragraf singkat/ringkas saja, buat dalam satu paragraf seperti contoh output. ingat anda berbicara kepada pedagang bukan developer model !**

    Tugas Anda:
    - Jelaskan tren penjualan (naik/turun/stabil) dalam 30 hari ke depan.
    - Berikan alasan prediksi tersebut (misalnya: libur Lebaran/libur sekolah, akhir pekan, hari kerja, tanggal cantik seperti 11.11, dll).
    - Berikan saran strategi bisnis kepada pedagang e-commerce agar bisa memanfaatkan momen atau mengatasi penurunan penjualan.
    - Gunakan Bahasa Indonesia yang profesional dan mudah dimengerti.

    ini adalah tanggal penting :
    # tanggal cantik promo :
    "2025-01-01", "2025-01-11", "2025-01-25",
    "2025-02-02", "2025-02-14", "2025-02-25",
    "2025-03-03", "2025-03-08", "2025-03-25",
    "2025-04-04", "2025-04-25",
    "2025-05-05", "2025-05-25",
    "2025-06-06", "2025-06-25",
    "2025-07-07", "2025-07-25",
    "2025-08-08", "2025-08-17", "2025-08-25",
    "2025-09-09", "2025-09-25",
    "2025-10-10", "2025-10-25",
    "2025-11-11", "2025-11-25",
    "2025-12-12", "2025-12-25", "2025-12-31"
    # 2025 Holidays : 
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-03-31",
    "2025-04-18", "2025-04-20", "2025-05-01", "2025-05-12", "2025-05-29",
    "2025-06-01", "2025-06-06", "2025-06-27", "2025-08-17", "2025-09-05",
    "2025-12-25"
    # Ramadan periods (1 month before Lebaran) : ("2025-03-01", "2025-03-30")
    # Lebaran periods (1 week after Idul Fitri) : ("2025-03-31", "2025-04-07")
    # School holiday periods : ("2025-06-15", "2025-07-15"),("2025-12-15", "2026-01-05")

    **Contoh output:**
    Berdasarkan tren penjualan sebelumnya, diprediksi akan terjadi penurunan penjualan setelah libur Lebaran karena masyarakat kembali fokus bekerja. Disarankan untuk membuat promo pasca-Lebaran agar menarik perhatian pembeli. Perlu diingat bahwa kondisi pasar bisa berubah sewaktu-waktu, maka strategi ini hanya sebagai acuan awal.
    
    Berikut data penjualan (30 hari terakhir dan 30 hari ke depan):
    {data_str}"""

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gagal mendapatkan analisis Gemini: {str(e)}"
    
    


# === ROUTES =======================================================================================
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    summary = summarize_text(text, summarization_model, summarization_tokenizer)
    return jsonify({
        "status": "success",
        "original_text": text,
        "summary": summary
    })

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


@app.route('/forecast', methods=['POST'])
def forecast():
    if 'file' not in request.files:
        return jsonify({'error': 'File CSV tidak ditemukan'}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    try:
        df = df[['date', 'total_sales']]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Check if data is sufficient
        if len(df) < WINDOW + FORECAST_STEP - 1:
            return jsonify({
                'error': f'Data historis kurang dari {WINDOW + FORECAST_STEP - 1} hari',
                'analysis': f'Tidak dapat membuat prediksi karena data historis kurang dari {WINDOW + FORECAST_STEP - 1} hari'
            }), 400

        # Apply log transformation to handle outliers
        df['total_sales'] = df['total_sales'].apply(lambda x: math.log(x) if x > 0 else 0)

        # Prepare data for model
        values = df[['total_sales']].values

        # Training model with specified parameters
        X, y = create_windows(values, WINDOW)
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=120,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            min_child_weight=10,
            gamma=0
        )
        model.fit(X, y)

        # Forecast 30 days ahead
        last_window = values[-WINDOW:, 0]
        future_preds_log = forecast_future(model, last_window, 30)
        
        # Apply inverse log transformation
        future_preds = np.exp(future_preds_log)

        # Generate future dates and format predictions
        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=30)
        predictions_dict = {
            str(date.date()): {"total_sales": format_rupiah(float(sales))}
            for date, sales in zip(future_dates, future_preds)
        }

        # Get last 30 days of actual data (inverse log transform)
        historical_data = df.tail(30).copy()
        historical_data['total_sales'] = historical_data['total_sales'].apply(lambda x: math.exp(x))

        # Get analysis from Gemini
        gemini_recommendation = get_gemini_analysis(historical_data, predictions_dict)

        # Handle Gemini analysis error
        if "Gagal" in gemini_recommendation or "error" in gemini_recommendation.lower():
            return jsonify({
                "predictions": predictions_dict,
                "analysis": "Terjadi kesalahan saat memproses analisis. Silakan coba lagi.",
                "error": gemini_recommendation
            }), 500

        return jsonify({
            "predictions": predictions_dict,
            "analysis": gemini_recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
