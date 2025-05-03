from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import uuid
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import hashlib
import base64
import glob

# Ganti ini jika kamu punya fungsi khusus
def preprocess_text(text):
    # Minimal: lower, hapus angka, simbol, split, hapus stopword
    import re, string
    STOPWORDS = set(['yang', 'untuk', 'dengan', 'pada', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'atau', 'juga', 'karena', 'ada', 'tidak', 'sudah', 'saja', 'sangat', 'lebih', 'agar', 'sebagai', 'jadi', 'oleh', 'kalau', 'dalam', 'bisa', 'akan', 'mereka', 'kami', 'kita', 'anda', 'saya', 'dia', 'aku', 'kamu', 'pun', 'apa', 'siapa', 'berapa', 'bagaimana', 'mengapa', 'semua', 'hanya', 'masih', 'setelah', 'sebelum', 'harus', 'dapat'])
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

def img_to_base64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def get_latest_wordcloud(sentiment):
    pattern = os.path.join(STATIC_FOLDER, f'wordcloud_{sentiment}_*.png')
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return img_to_base64(latest)

def get_latest_plot(prefix):
    pattern = os.path.join(STATIC_FOLDER, f'{prefix}_*.png')
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return img_to_base64(latest)

def get_sample_komentar():
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'komentar_labeled.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'text' in df.columns and 'sentimen' in df.columns:
            return df[['text', 'sentimen']].head(30).to_dict(orient='records')
    return []

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'app', 'templates'))
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'dataset')
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def label_sentiment(text):
    # Label otomatis berbasis keyword sederhana
    POS = ['bagus', 'mantap', 'welcome', 'selamat', 'good', 'luar biasa', 'sukses', 'terima kasih', 'keren', 'top', 'hebat', 'yes', 'semangat', 'love', '🔥', '👍', '❤']
    NEG = ['jelek', 'parah', 'gagal', 'out', 'kontol', 'anjir', 'anjirr', 'gblk', 'gila', 'nggak', 'gak', 'enggak', 'sial', 'bangsat', 'tolol', 'cok', 'cokkk', 'balikin', 'benci', 'ngaco', 'ngawur', 'down', 'no', '😡', '👎']
    text_l = text.lower()
    if any(w in text_l for w in POS):
        return 'positif'
    elif any(w in text_l for w in NEG):
        return 'negatif'
    else:
        return 'netral'

def predict_sentiment_lstm(texts, model, tokenizer, label_encoder):
    # texts: list of str
    # return: list of sentimen label
    if (model is None) or (tokenizer is None) or (label_encoder is None):
        # fallback ke keyword
        return [label_sentiment(t) for t in texts]
    # Tokenisasi dan padding
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=100)
    preds = model.predict(padded)
    labels = label_encoder.inverse_transform(preds.argmax(axis=1))
    return labels

def get_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Pastikan kolom text & waktu
            if 'text' in df.columns and 'createTimeISO' in df.columns:
                # Preprocessing
                df['text_clean'] = df['text'].astype(str).apply(preprocess_text)
                # Labeling sentimen otomatis jika belum ada
                if 'sentimen' not in df.columns or df['sentimen'].isnull().any():
                    if model and tokenizer and label_encoder:
                        df['sentimen'] = predict_sentiment_lstm(df['text_clean'].tolist(), model, tokenizer, label_encoder)
                    else:
                        df['sentimen'] = df['text'].astype(str).apply(label_sentiment)
                # Kolom waktu
                df['created_at'] = pd.to_datetime(df['createTimeISO'], errors='coerce')
            else:
                flash('Kolom text atau createTimeISO tidak ditemukan!', 'danger')
        except Exception as e:
            flash(f'Gagal membaca data: {e}', 'danger')
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    return df

def load_artifacts():
    model, tokenizer, label_encoder = None, None, None
    model_path = os.path.join('lstm_sentiment.h5')
    tokenizer_path = os.path.join('tokenizer_lstm.pkl')
    label_encoder_path = os.path.join('label_encoder.pkl')
    try:
        if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
            model = load_model(model_path)
            tokenizer = joblib.load(tokenizer_path)
            label_encoder = joblib.load(label_encoder_path)
    except Exception as e:
        flash(f'Gagal memuat model: {e}', 'danger')
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

@app.route('/', methods=['GET', 'POST'])
def index():
    table = get_sample_komentar()
    preview = table
    sentiment_counts = {'positif': 0, 'negatif': 0, 'netral': 0}
    if 'sentimen' in pd.DataFrame(table).columns:
        counts = pd.DataFrame(table)['sentimen'].value_counts().to_dict()
        for key in sentiment_counts.keys():
            sentiment_counts[key] = counts.get(key, 0)
    distribusi_img = get_latest_plot('sentiment')
    aktivitas_img = get_latest_plot('activity')
    wordcloud_pos = get_latest_wordcloud('positif')
    wordcloud_neg = get_latest_wordcloud('negatif')
    wordcloud_net = get_latest_wordcloud('netral')
    # Prediksi komentar baru
    pred_result = None
    comment_input = ''
    if request.method == 'POST' and 'komentar' in request.form:
        comment_input = request.form.get('komentar', '')
        cleaned = preprocess_text(comment_input)
        if model and tokenizer and label_encoder:
            sent = predict_sentiment_lstm([cleaned], model, tokenizer, label_encoder)[0]
        else:
            sent = label_sentiment(cleaned)
        pred_result = {
            'komentar': comment_input,
            'cleaned': cleaned,
            'sentimen': sent
        }

    # Status evaluasi model
    model_status = 'Model LSTM belum dilatih.'
    if model and tokenizer and label_encoder:
        model_status = 'Model LSTM siap digunakan.'

    # Load grafik training history LSTM
    lstm_training_img = None
    try:
        lstm_training_img = img_to_base64(os.path.join(STATIC_FOLDER, 'lstm_training.png'))
    except Exception:
        pass

    return render_template('index.html',
        table=table,
        distribusi_img=distribusi_img, aktivitas_img=aktivitas_img,
        wordcloud_pos=wordcloud_pos, wordcloud_neg=wordcloud_neg, wordcloud_net=wordcloud_net,
        sentiment_counts=sentiment_counts, pred_result=pred_result, comment_input=comment_input,
        model_status=model_status,
        lstm_training_img=lstm_training_img
    )

@app.route('/refresh')
def refresh():
    flash('Dataset direfresh!', 'info')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            save_path = os.path.join(os.path.dirname(__file__), 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')
            file.save(save_path)
            flash('Dataset berhasil diupload dan diganti.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Format file harus .csv', 'danger')
    return render_template('upload.html')

import threading

@app.route('/train', methods=['POST'])
def train():
    def background_train():
        from app.model import train_lstm_model
        csv_path = os.path.join(os.path.dirname(__file__), 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            if 'text_clean' not in df.columns:
                from app.preprocessing import preprocess_text
                df['text_clean'] = df['text'].astype(str).apply(preprocess_text)
                df.to_csv(csv_path, index=False)
            model_, tokenizer_, _ = train_lstm_model(csv_path)
            global model, tokenizer, label_encoder
            model, tokenizer = model_, tokenizer_
            # Tidak bisa flash di background thread
        except Exception as e:
            print(f'Gagal training model: {e}')
    threading.Thread(target=background_train, daemon=True).start()
    flash('Training model LSTM sedang berjalan di background. Silakan tunggu beberapa menit lalu refresh halaman.', 'info')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
