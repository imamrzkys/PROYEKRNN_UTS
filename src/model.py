import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import sys
import io

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../lstm_sentiment.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '../tokenizer_lstm.npy')

max_words = 3500
max_len = 50

# Label encoding
sentiment_map = {'negatif': 0, 'netral': 1, 'positif': 2}
def encode_label(y):
    return [sentiment_map[label] for label in y]
def decode_label(idx):
    inv = {v: k for k, v in sentiment_map.items()}
    return inv[idx] if idx in inv else 'netral'

def train_lstm_model(csv_path, save_model=True, show_plot=False, progress_callback=None):
    import sys
    import io
    try:
        sys.stdout_save = sys.stdout
        sys.stdout = io.TextIOWrapper(open('train_log.txt', 'wb', 0), encoding='utf-8', line_buffering=True)
    except Exception:
        pass
    try:
        df = pd.read_csv(csv_path)
        # Gunakan kolom 'text' sebagai input
        if 'text' not in df.columns:
            raise ValueError("Kolom 'text' tidak ditemukan pada dataset. Pastikan dataset memiliki kolom 'text' untuk komentar.")
        X = df['text'].astype(str).tolist()
        # Cek apakah kolom label tersedia
        if 'sentimen' not in df.columns:
            raise ValueError("Kolom label 'sentimen' tidak ditemukan pada dataset.\nSilakan tambahkan kolom 'sentimen' (misal: 'negatif', 'netral', 'positif') pada file CSV Anda sebelum melakukan training.")
        y = encode_label(df['sentimen'].tolist())
        tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(X)
        X_seq = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post')
        y_cat = to_categorical(y, num_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42)
        model = Sequential([
            Embedding(max_words, 128, input_length=max_len),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callbacks = []
        if progress_callback is not None:
            from tensorflow.keras.callbacks import Callback
            class KerasProgressCallback(Callback):
                def __init__(self, cb, total_epochs):
                    super().__init__()
                    self.cb = cb(total_epochs)
                def on_epoch_end(self, epoch, logs=None):
                    self.cb.on_epoch_end(epoch, logs)
            callbacks.append(KerasProgressCallback(progress_callback, total_epochs=8))
        history = model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=callbacks)
        if save_model:
            model.save(MODEL_PATH)
            np.save(TOKENIZER_PATH, tokenizer.to_json())
        if show_plot:
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.subplot(1,2,2)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.tight_layout()
            static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
            os.makedirs(static_folder, exist_ok=True)
            plt.savefig(os.path.join(static_folder, 'lstm_training.png'))
            plt.close()
    finally:
        try:
            sys.stdout.detach()
        except Exception:
            pass
        try:
            sys.stdout = sys.stdout_save
        except Exception:
            sys.stdout = sys.__stdout__
    return model, tokenizer, (X_test, y_test)

def evaluate_lstm_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=['negatif','netral','positif'])
    return {'confusion_matrix': cm, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'report': report}

def load_lstm_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import json
    with open(TOKENIZER_PATH, 'r') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    return model, tokenizer

def predict_sentiment_lstm(text, model=None, tokenizer=None):
    if model is None or tokenizer is None:
        model, tokenizer = load_lstm_model()
        if model is None:
            return 'netral', 0.0
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)
    idx = np.argmax(pred)
    label = decode_label(idx)
    confidence = float(np.max(pred))
    return label, confidence

if __name__ == "__main__":
    # Path CSV sudah otomatis ke hasil labeling
    csv_path = r"C:\Users\X395\OneDrive\Documents\KULIAH SEMESTER 4\TUGAS\PRAKTIKUM AI\tugasuts_rnn\dataset\komentar_labeled.csv"
    train_lstm_model(csv_path, show_plot=True)
