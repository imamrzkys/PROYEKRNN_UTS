from flask import Flask, render_template, request, redirect, url_for, flash
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
import glob
import logging
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import traceback
import io
import sys

# Logging
logging.basicConfig(level=logging.INFO)

# Path
BASE_DIR = os.path.dirname(__file__)
STATIC_FOLDER = os.path.join(BASE_DIR, 'src/static')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'lstm_sentiment.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer_lstm.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset/komentar_labeled.csv')
RAW_CSV_PATH = os.path.join(BASE_DIR, 'dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv')

# Pastikan folder penting ada
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- FLASK APP SETUP ---
app = Flask(__name__, static_folder='src/static', template_folder='src/templates')
app.secret_key = 'supersecretkey'

# Preprocessing sederhana
def preprocess_text(text):
    import re, string
    # Hapus emoji dan karakter non-alfabet kecuali tanda baca dasar
    text = re.sub(r'[^\w\s,.!?-]', '', text)  # buang emoji/karakter non-alfabet kecuali tanda baca dasar
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
    tokens = text.split()
    pos = any(word in tokens for word in POS)
    neg = any(word in tokens for word in NEG)
    # Deteksi negasi
    negasi = any(n in tokens for n in ['tidak', 'nggak', 'gak'])
    if negasi:
        if pos and not neg:
            return 'negatif'
        elif neg and not pos:
            return 'positif'
        elif pos and neg:
            return 'netral'
    if pos and not neg:
        return 'positif'
    elif neg and not pos:
        return 'negatif'
    elif pos and neg:
        return 'netral'
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

# --- LOAD MODEL, TOKENIZER, ENCODER ---
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

# Helper untuk load dataset
def load_data():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    else:
        # Jika belum ada, proses dulu
        from src.utils import preprocess_and_label_csv
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
        static_folder = os.path.join(os.path.dirname(__file__), 'static')
        os.makedirs(static_folder, exist_ok=True)
        plt.savefig(os.path.join(static_folder, 'emotion_distribution.png'))
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return img_base64

def plot_wordcloud(df, sentimen):
    from wordcloud import WordCloud
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

def plot_aktivitas_waktu(df):
    static_img_path = os.path.join(STATIC_FOLDER, 'activity_ed0fd13414fe4ecdb12789637f05e53a.png')
    if os.path.exists(static_img_path):
        with open(static_img_path, 'rb') as f:
            import base64
            return base64.b64encode(f.read()).decode()
    # Jika tidak ada file gambar, cek data seperti biasa
    if 'created_at' in df.columns and df['created_at'].notna().sum() > 0:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        aktivitas = df.groupby(df['created_at'].dt.date).size()
        if aktivitas.empty:
            return ''
        plt.figure(figsize=(6, 2.5))
        aktivitas.plot(marker='o', linewidth=2, color='#3949ab')
        plt.title('Aktivitas Komentar per Hari')
        plt.xlabel('Tanggal')
        plt.ylabel('Jumlah Komentar')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        if len(aktivitas) > 0:
            median = aktivitas.median()
            plt.ylim(0, max(aktivitas.max(), median*1.2))
        import io, base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return img_base64
    return ''

def ensure_created_at_exists():
    try:
        df = pd.read_csv(DATASET_PATH)
        if 'created_at' not in df.columns:
            import numpy as np
            df['created_at'] = pd.date_range('2025-04-01', periods=len(df), freq='H')
            df.to_csv(DATASET_PATH, index=False)
    except Exception:
        pass

# Pastikan kolom created_at ada sebelum web jalan
ensure_created_at_exists()

# Variabel global untuk progress
train_progress = {'status': 'idle', 'current': 0, 'total': 0, 'msg': '', 'done': False}

# --- FLASK APP SETUP ---
app = Flask(__name__, static_folder='src/static', template_folder='src/templates')
app.secret_key = 'supersecretkey'

# Helper: cek ekstensi file upload
ALLOWED_EXTENSIONS = {'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    komentar = ''
    hasil_prediksi = None

    if request.method == 'POST':
        komentar = request.form.get('komentar', '')
        clean = preprocess_text(komentar)
        hasil = predict_sentiment_lstm([clean])[0]
        hasil_prediksi = {'komentar': komentar, 'cleaned': clean, 'sentimen': hasil}

    # Ambil data asli jika ada
    df = None
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            logging.error(f"Gagal baca CSV: {e}")
    else:
        logging.warning("File dataset tidak ditemukan, tampilkan data dummy.")

    if df is not None and not df.empty:
        # Ensure 'sentimen' column exists
        if 'sentimen' not in df.columns:
            # Option 1: Set to 'netral' (or run batch labeling)
            df['text_clean'] = df['text'].astype(str).apply(preprocess_text)
            if model and tokenizer and label_encoder:
                seqs = tokenizer.texts_to_sequences(df['text_clean'].tolist())
                padded = pad_sequences(seqs, maxlen=100)
                preds = model.predict(padded)
                df['sentimen'] = label_encoder.inverse_transform(preds.argmax(axis=1))
            else:
                df['sentimen'] = df['text_clean'].apply(label_sentiment)
            df.to_csv(CSV_PATH, index=False)
        # --- Tambahan: selalu re-label sample_data agar hasil terbaru ---
        sample_data = df[['text', 'sentimen']].head(20).copy()
        sample_data['text_clean'] = sample_data['text'].astype(str).apply(preprocess_text)
        from src.utils import label_sentiment as better_label_sentiment
        sample_data['sentimen'] = sample_data['text_clean'].apply(better_label_sentiment)
        counts = df['sentimen'].value_counts().to_dict()
        distribusi_img = get_latest_plot('sentiment')
        wordclouds = {
            'positif': get_latest_wordcloud('positif'),
            'negatif': get_latest_wordcloud('negatif'),
            'netral': get_latest_wordcloud('netral'),
        }
    else:
        # Dummy data jika file tidak ada
        sample_data = pd.DataFrame([
            {'text': 'STY out!', 'sentimen': 'negatif'},
            {'text': 'Ayo semangat timnas', 'sentimen': 'positif'},
        ])
        counts = sample_data['sentimen'].value_counts().to_dict()
        distribusi_img = None
        wordclouds = {'positif': None, 'negatif': None, 'netral': None}

    # Aktivitas Komentar per Waktu
    aktivitas_img = None
    if df is not None:
        try:
            aktivitas_img = plot_aktivitas_waktu(df)
        except Exception as e:
            logging.error(f'Gagal generate grafik aktivitas: {e}')
            aktivitas_img = None

    model_status = 'Model LSTM belum tersedia.' if not model else 'Model LSTM siap digunakan.'

    return render_template('index.html',
        table=sample_data.to_dict(orient='records'),
        sentiment_counts=counts,
        distribusi_img=distribusi_img,
        wordcloud_pos=wordclouds['positif'],
        wordcloud_neg=wordclouds['negatif'],
        wordcloud_net=wordclouds['netral'],
        aktivitas_img=aktivitas_img,
        pred_result=hasil_prediksi,
        comment_input=komentar,
        model_status=model_status,
        lstm_training_img=None
    )

@app.route('/train_progress')
def get_train_progress():
    from flask import jsonify
    return jsonify(train_progress)

@app.route('/train', methods=['POST'])
def train():
    import threading
    from src.model import train_lstm_model, evaluate_lstm_model
    def train_thread():
        import os
        global train_progress
        try:
            # --- VALIDASI DATASET DAN LOGGING ---
            if not os.path.exists(DATASET_PATH):
                train_progress['status'] = 'error'
                train_progress['msg'] = f"Dataset tidak ditemukan: {DATASET_PATH}"
                train_progress['done'] = True
                return
            try:
                df = pd.read_csv(DATASET_PATH, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(DATASET_PATH, encoding='latin1')
            except Exception as e:
                train_progress['status'] = 'error'
                train_progress['msg'] = f"Gagal membaca dataset: {e}"
                train_progress['done'] = True
                return
            if df.empty:
                train_progress['status'] = 'error'
                train_progress['msg'] = 'Dataset kosong.'
                train_progress['done'] = True
                return
            if 'text' not in df.columns:
                train_progress['status'] = 'error'
                train_progress['msg'] = "Kolom 'text' tidak ditemukan pada dataset."
                train_progress['done'] = True
                return
            if 'sentimen' not in df.columns:
                train_progress['status'] = 'error'
                train_progress['msg'] = "Kolom 'sentimen' tidak ditemukan pada dataset."
                train_progress['done'] = True
                return
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
            # Pastikan stdout encoding UTF-8 agar log tidak error
            try:
                import io, os
                sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), encoding='utf-8', line_buffering=True)
            except Exception:
                try:
                    sys.stdout.reconfigure(encoding='utf-8')
                except Exception:
                    pass
            model, tokenizer, _ = train_lstm_model(DATASET_PATH, progress_callback=ProgressCallback, show_plot=True)
            train_progress['status'] = 'done'
            train_progress['msg'] = 'Training selesai! Grafik hasil training sudah tersedia.'
            train_progress['done'] = True
        except Exception as e:
            train_progress['status'] = 'error'
            train_progress['msg'] = f'{e}\n{traceback.format_exc()}'
            train_progress['done'] = True
    t = threading.Thread(target=train_thread)
    t.start()
    return render_template('train_progress.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Pilih file CSV terlebih dahulu!', 'danger')
            return redirect(url_for('upload'))
        if not allowed_file(file.filename):
            flash('Format file harus .csv!', 'danger')
            return redirect(url_for('upload'))
        filepath = os.path.join(os.path.dirname(__file__), 'dataset/uploaded.csv')
        file.save(filepath)
        try:
            from src.utils import preprocess_and_label_csv
            preprocess_and_label_csv(filepath, DATASET_PATH, text_col='text')
            flash('Upload dan preprocessing berhasil!', 'success')
        except Exception as e:
            flash(f'Gagal upload/preprocessing: {e}', 'danger')
            return redirect(url_for('upload'))
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/refresh', methods=['GET', 'POST'])
def refresh():
    from src.utils import preprocess_and_label_csv
    preprocess_and_label_csv(RAW_CSV_PATH, DATASET_PATH, text_col='text')
    flash('Dataset dan preprocessing di-refresh!', 'success')
    return redirect(url_for('index'))

# Untuk development lokal
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
