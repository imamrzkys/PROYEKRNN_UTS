from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import pandas as pd
from src.utils import preprocess_and_label_csv, label_sentiment, label_emotion
from src.preprocessing import preprocess_text
from src.model import predict_sentiment_lstm, train_lstm_model, evaluate_lstm_model, load_lstm_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import base64
import threading
import time

app = Flask(__name__)
app.secret_key = 'uts-rnn-secret-key'

# Agar bisa di-import di app.py
from flask import Blueprint
bp = Blueprint('main', __name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), '../dataset/komentar_labeled.csv')
RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), '../dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')

# Helper untuk load dataset

def load_data():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    else:
        # Jika belum ada, proses dulu
        df = preprocess_and_label_csv(RAW_CSV_PATH, DATASET_PATH, text_col='text')
        return df

def plot_sentiment_distribution(df):
    plt.figure(figsize=(5,4))
    ax = df['sentimen'].value_counts().plot(kind='bar', color=['green','red','gray'])
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64

def plot_emotion_distribution(df, save_static=False):
    import matplotlib.pyplot as plt
    import io, base64, os
    plt.figure(figsize=(5,4))
    colors = ['#e53935','#3949ab','#43a047','#fb8c00','#8e24aa','#00acc1','#757575']
    emosi_order = ['marah','sedih','bahagia','takut','jijik','terkejut','netral']
    counts = df['emosi'].value_counts().reindex(emosi_order, fill_value=0)
    ax = counts.plot(kind='bar', color=colors)
    plt.title('Distribusi Emosi')
    plt.xlabel('Emosi')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    if save_static:
        static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
        os.makedirs(static_folder, exist_ok=True)
        plt.savefig(os.path.join(static_folder, 'emotion_distribution.png'))
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64

def plot_wordcloud(df, sentimen):
    text = ' '.join(df[df['sentimen']==sentimen]['text_clean'].astype(str))
    if not text.strip():
        text = 'kosong'
    wc = WordCloud(width=500, height=300, background_color='white').generate(text)
    buf = io.BytesIO()
    plt.figure(figsize=(6,3))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64

# Variabel global untuk progress
train_progress = {'status': 'idle', 'current': 0, 'total': 0, 'msg': '', 'done': False}

@app.route('/train_progress')
def get_train_progress():
    return jsonify(train_progress)

@bp.route('/', methods=['GET', 'POST'])
def index():
    df = load_data()
    pred_result = None
    comment_input = ''
    # Handle prediksi satu komentar
    if request.method == 'POST' and 'komentar' in request.form:
        comment_input = request.form.get('komentar')
        if not comment_input.strip():
            flash('Komentar tidak boleh kosong!', 'danger')
        else:
            cleaned = preprocess_text(comment_input)
            try:
                model, tokenizer = load_lstm_model()
                label, conf = predict_sentiment_lstm(cleaned, model, tokenizer)
            except Exception:
                label, conf = label_sentiment(cleaned), 1.0
            pred_result = {
                'komentar': comment_input,
                'cleaned': cleaned,
                'sentimen': label,
                'confidence': conf,
                'emosi': label_emotion(cleaned)
            }
    distribusi_img = plot_sentiment_distribution(df)
    emotion_img = plot_emotion_distribution(df, save_static=True)
    wordcloud_pos = plot_wordcloud(df, 'positif')
    wordcloud_neg = plot_wordcloud(df, 'negatif')
    wordcloud_net = plot_wordcloud(df, 'netral')
    aktivitas_img = plot_aktivitas_waktu(df)
    # Evaluasi model jika ada
    eval_metrics = None
    try:
        model, tokenizer = load_lstm_model()
        if model is not None:
            X = df['text_clean'].astype(str).tolist()
            y = df['sentimen'].map({'negatif':0,'netral':1,'positif':2}).values
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            X_seq = tokenizer.texts_to_sequences(X)
            X_pad = pad_sequences(X_seq, maxlen=50, padding='post')
            y_cat = to_categorical(y, num_classes=3)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42)
            eval_metrics = evaluate_lstm_model(model, X_test, y_test)
    except Exception:
        pass
    return render_template('index.html',
        distribusi_img=distribusi_img,
        emotion_img=emotion_img,
        wordcloud_pos=wordcloud_pos,
        wordcloud_neg=wordcloud_neg,
        wordcloud_net=wordcloud_net,
        aktivitas_img=aktivitas_img,
        pred_result=pred_result,
        comment_input=comment_input,
        table=df[['text','sentimen','emosi']].head(30).to_dict(orient='records'),
        eval_metrics=eval_metrics
    )

@app.route('/train', methods=['POST'])
def train():
    def train_thread():
        global train_progress
        try:
            train_progress['status'] = 'training'
            train_progress['current'] = 0
            train_progress['msg'] = 'Inisialisasi...'
            train_progress['done'] = False
            # Custom callback untuk update progress
            class ProgressCallback:
                def __init__(self, total_epochs):
                    self.total_epochs = total_epochs
                def on_epoch_end(self, epoch, logs=None):
                    train_progress['current'] = epoch + 1
                    train_progress['total'] = self.total_epochs
                    train_progress['msg'] = f"Epoch {epoch+1}/{self.total_epochs} - acc: {logs.get('accuracy',0):.3f} - val_acc: {logs.get('val_accuracy',0):.3f}"
            from src.model import train_lstm_model
            model, tokenizer, _ = train_lstm_model(DATASET_PATH, progress_callback=ProgressCallback)
            train_progress['status'] = 'done'
            train_progress['msg'] = 'Training selesai!'
            train_progress['done'] = True
        except Exception as e:
            train_progress['status'] = 'error'
            train_progress['msg'] = str(e)
            train_progress['done'] = True
    # Jalankan thread agar tidak blocking
    t = threading.Thread(target=train_thread)
    t.start()
    return render_template('train_progress.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash('Pilih file CSV terlebih dahulu!', 'danger')
            return redirect(url_for('upload'))
        filepath = os.path.join(os.path.dirname(__file__), '../dataset/uploaded.csv')
        file.save(filepath)
        preprocess_and_label_csv(filepath, DATASET_PATH, text_col='text')
        flash('Upload dan preprocessing berhasil!', 'success')
        return redirect(url_for('index'))
    return render_template('upload.html')

def plot_aktivitas_waktu(df):
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        aktivitas = df.groupby(df['created_at'].dt.date).size()
        plt.figure(figsize=(6, 2.5))  # Ukuran lebih kecil dan proporsional
        aktivitas.plot(marker='o', linewidth=2, color='#3949ab')
        plt.title('Aktivitas Komentar per Hari')
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah Komentar')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        # Batasi sumbu y agar tidak terlalu tinggi (misal, max 1.2x dari median jika outlier terlalu besar)
        if len(aktivitas) > 0:
            median = aktivitas.median()
            plt.ylim(0, max(aktivitas.max(), median*1.2))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64
    return ''

@bp.route('/refresh', methods=['GET', 'POST'])
def refresh():
    # Untuk re-preprocessing jika dataset berubah
    preprocess_and_label_csv(RAW_CSV_PATH, DATASET_PATH, text_col='text')
    flash('Dataset dan preprocessing di-refresh!', 'success')
    return redirect(url_for('main.index'))
