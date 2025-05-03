from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import base64
import glob
import logging
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Logging
logging.basicConfig(level=logging.INFO)

# Path
BASE_DIR = os.path.dirname(__file__)
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'lstm_sentiment.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer_lstm.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')

# Pastikan folder penting ada
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing sederhana
def preprocess_text(text):
    import re, string
    STOPWORDS = set([
        'yang', 'untuk', 'dengan', 'pada', 'dan', 'di', 'ke', 'dari', 'ini',
        'itu', 'atau', 'juga', 'karena', 'ada', 'tidak', 'sudah', 'saja',
        'sangat', 'lebih', 'agar', 'sebagai', 'jadi', 'oleh', 'kalau', 'dalam',
        'bisa', 'akan', 'mereka', 'kami', 'kita', 'anda', 'saya', 'dia', 'aku',
        'kamu', 'pun', 'apa', 'siapa', 'berapa', 'bagaimana', 'mengapa', 'semua',
        'hanya', 'masih', 'setelah', 'sebelum', 'harus', 'dapat'
    ])
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return ' '.join(tokens)

# Fungsi bantu
def label_sentiment(text):
    POS = ['bagus', 'mantap', 'welcome', 'selamat', 'good', 'luar biasa', 'sukses', 'terima kasih', 'keren', 'top', 'hebat', 'yes', 'semangat', 'love']
    NEG = ['jelek', 'parah', 'gagal', 'out', 'kontol', 'anjir', 'gblk', 'nggak', 'gak', 'sial', 'bangsat', 'tolol', 'cok', 'benci', 'ngaco']
    text = text.lower()
    if any(w in text for w in POS):
        return 'positif'
    elif any(w in text for w in NEG):
        return 'negatif'
    else:
        return 'netral'

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return None

def get_latest_plot(prefix):
    pattern = os.path.join(STATIC_FOLDER, f'{prefix}_*.png')
    files = glob.glob(pattern)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    return img_to_base64(latest)

def get_latest_wordcloud(sentiment):
    return get_latest_plot(f'wordcloud_{sentiment}')

# Load model
def load_artifacts():
    try:
        model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        tokenizer = joblib.load(TOKENIZER_PATH) if os.path.exists(TOKENIZER_PATH) else None
        label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
        return model, tokenizer, label_encoder
    except Exception as e:
        logging.warning(f'Gagal memuat model/tokenizer/label encoder: {e}')
        return None, None, None

model, tokenizer, label_encoder = load_artifacts()

def predict_sentiment_lstm(texts):
    if model and tokenizer and label_encoder:
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=100)
        preds = model.predict(padded)
        return label_encoder.inverse_transform(preds.argmax(axis=1))
    else:
        return [label_sentiment(t) for t in texts]

# WAJIB untuk deployment: expose variabel `app`
# Railway dan Gunicorn akan mencari variabel bernama "app" di file utama
# Pastikan variabel 'app' tersedia di global scope

app = Flask(__name__, template_folder='app/templates')
app.secret_key = 'supersecretkey'

@app.route('/', methods=['GET', 'POST'])
def index():
    komentar = ''
    hasil_prediksi = None

    if request.method == 'POST':
        komentar = request.form.get('komentar', '')
        clean = preprocess_text(komentar)
        hasil = predict_sentiment_lstm([clean])[0]
        hasil_prediksi = {'komentar': komentar, 'cleaned': clean, 'sentimen': hasil}

    # Dummy data sementara
    sample_data = pd.DataFrame([
        {'text': 'STY out!', 'sentimen': 'negatif'},
        {'text': 'Ayo semangat timnas', 'sentimen': 'positif'},
    ])
    counts = sample_data['sentimen'].value_counts().to_dict()
    distribusi_img = get_latest_plot('sentiment')
    wordclouds = {
        'positif': get_latest_wordcloud('positif'),
        'negatif': get_latest_wordcloud('negatif'),
        'netral': get_latest_wordcloud('netral'),
    }

    model_status = 'Model LSTM belum tersedia.' if not model else 'Model LSTM siap digunakan.'

    return render_template('index.html',
        table=sample_data.to_dict(orient='records'),
        sentiment_counts=counts,
        distribusi_img=distribusi_img,
        wordcloud_pos=wordclouds['positif'],
        wordcloud_neg=wordclouds['negatif'],
        wordcloud_net=wordclouds['netral'],
        pred_result=hasil_prediksi,
        comment_input=komentar,
        model_status=model_status,
        lstm_training_img=None
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            file.save(CSV_PATH)
            flash('Dataset berhasil diunggah.', 'success')
            return redirect(url_for('index'))
        else:
            flash('Harus file .csv', 'danger')
    return render_template('upload.html')

# Untuk development lokal
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
