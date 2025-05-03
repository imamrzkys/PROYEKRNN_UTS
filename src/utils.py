import pandas as pd
from src.preprocessing import preprocess_text
from tqdm import tqdm  

# Keyword heuristik sederhana untuk labeling
POSITIVE_WORDS = [
    'bagus', 'mantap', 'keren', 'hebat', 'top', 'welcome', 'semangat', 'setuju', 'percaya', 'total football', 'selamat', 'good', 'sukses', 'support', 'percaya', 'bravo', 'terbaik', 'yes', 'oke', 'legend', '🔥', '❤', 'love', 'amazing', 'wow', 'super', 'harapan', 'semoga', 'maju', 'menang', 'juara',
    # tambahan kontekstual bola
    'welcome total football', 'goodbye sty', 'selamat tinggal parkir bus', 'welcome pk', 'welcome patrick', 'welcome kluivert', 'total football', 'bravo timnas', 'maju timnas', 'semangat timnas', 'timnas juara'
]
NEGATIVE_WORDS = [
    'out', 'blunder', 'kontol', 'gagal', 'jelek', 'buruk', 'parah', 'anjir', 'anjirr', 'anjing', 'gblk', 'goblok', 'gila', 'kacau', 'kalah', 'pecat', 'benci', 'tidak', 'ga', 'gak', 'no', 'down', 'dark', 'era kegelapan', 'cemas', 'balikin', 'salah', 'ngaco', 'ngawur', 'nggak', 'ngga', 'menyedihkan', 'ngapain', 'mending', 'keluar', 'quit', 'fomo', 'fail', 'gk', 'petrik out', 'kluivert out', 'patrick out', 'bodo', 'becus', 'bodoh', 'tolol', 'payah', 'lemah', 'malas', 'malu', 'hina', 'sampah', 'bangsat', 'brengsek', 'anjg', 'anj', 'kampret', 'sial', 'menyebalkan', 'menjijikkan', 'sakit hati', 'kecewa', 'tidak suka', 'tidak setuju', 'tidak mampu', 'ga becus', 'ga bisa', 'ga guna', 'ga jelas', 'ga paham', 'ga ngerti', 'ga penting', 'ga suka', 'ga setuju', 'ga mampu', 'ga layak',
    # tambahan kontekstual bola
    'kluivert out', 'patrick kluivert out', 'sty out', 'balikin papa sty', 'out kluivert', 'out patrick', 'out sty', 'parkir bus', 'latihan dulu', 'kalah 3-0', 'blunderrr', '#patrick kluivert outt'
]

# ===== EMOTION KEYWORDS =====
EMOTION_WORDS = {
    'marah': ['marah', 'kesal', 'benci', 'emosi', 'geram', 'dongkol', 'kzl', 'jengkel', 'anjing', 'bangsat', 'kampret', 'goblok', 'tolol', 'sial', 'parah', 'brengsek', 'maki', 'males', 'muak'],
    'sedih': ['sedih', 'kecewa', 'menangis', 'nangis', 'patah hati', 'galau', 'terharu', 'terluka', 'sakit hati', 'hampa', 'down', 'frustasi', 'putus asa'],
    'bahagia': ['bahagia', 'senang', 'gembira', 'happy', 'lega', 'puas', 'tertawa', 'ketawa', 'alhamdulillah', 'senyum', 'terhibur', 'syukur', 'bersyukur', 'enjoy', 'asik'],
    'takut': ['takut', 'ngeri', 'khawatir', 'cemas', 'was-was', 'deg-degan', 'parno', 'merinding'],
    'jijik': ['jijik', 'ilfeel', 'eneg', 'muak', 'ogah', 'risih'],
    'terkejut': ['kaget', 'terkejut', 'shock', 'astaga', 'waduh', 'wow', 'loh', 'hah', 'eh'],
    'netral': []
}

def label_sentiment(text):
    t = text.lower()
    # Cek kata positif dan negatif sebagai kata utuh (bukan substring saja)
    pos = any(f" {word} " in f" {t} " or t.startswith(word+" ") or t.endswith(" "+word) or t==word for word in POSITIVE_WORDS)
    neg = any(f" {word} " in f" {t} " or t.startswith(word+" ") or t.endswith(" "+word) or t==word for word in NEGATIVE_WORDS)
    if pos and not neg:
        return 'positif'
    elif neg and not pos:
        return 'negatif'
    elif pos and neg:
        return 'netral'  # ambigu
    else:
        return 'netral'

def label_emotion(text):
    t = text.lower()
    score = {emo:0 for emo in EMOTION_WORDS}
    for emo, words in EMOTION_WORDS.items():
        for word in words:
            if word in t:
                score[emo] += 1
    # Ambil emosi dengan skor tertinggi, default netral
    max_emo = max(score, key=score.get)
    if score[max_emo] == 0:
        return 'netral'
    return max_emo

import time

def preprocess_and_label_csv(input_csv, output_csv, text_col='text'):
    import os
    import sys
    import codecs
    print(f"Mulai proses labeling dataset: {input_csv}")
    start = time.time()
    # --- Robust file read: try utf-8, fallback to latin1 ---
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='latin1')
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file CSV: {e}")
    print(f"Jumlah data sebelum drop NA: {len(df)}")
    df = df.dropna(subset=[text_col])
    print(f"Jumlah data setelah drop NA: {len(df)}")
    # Buat kolom created_at jika tidak ada
    if 'created_at' not in df.columns:
        if 'createTimeISO' in df.columns:
            df['created_at'] = pd.to_datetime(df['createTimeISO'], errors='coerce')
            if df['created_at'].isna().all():
                df['created_at'] = pd.date_range('2025-04-01', periods=len(df), freq='H')
        else:
            df['created_at'] = pd.date_range('2025-04-01', periods=len(df), freq='H')
    else:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    # Normalisasi kolom teks
    df[text_col] = df[text_col].astype(str)
    # Handle duplikasi (berdasarkan teks dan waktu)
    df = df.drop_duplicates(subset=[text_col, 'created_at'])
    tqdm.pandas(desc="Preprocessing")
    df['text_clean'] = df[text_col].progress_apply(preprocess_text)
    tqdm.pandas(desc="Labeling Sentimen")
    df['sentimen'] = df['text_clean'].progress_apply(label_sentiment)
    tqdm.pandas(desc="Labeling Emosi")
    df['emosi'] = df['text_clean'].progress_apply(label_emotion)
    # Urutkan berdasarkan waktu jika memungkinkan
    df = df.sort_values('created_at')
    # Pilih kolom utama untuk web dan training
    cols_needed = [text_col, 'text_clean', 'sentimen', 'emosi', 'created_at']
    extra_cols = []
    for col in ['uid', 'uniqueId', 'videoWebUrl', 'cid', 'avatarThumbnail']:
        if col in df.columns and col not in cols_needed:
            extra_cols.append(col)
    cols_final = cols_needed + extra_cols
    cols_final = [col for col in cols_final if col in df.columns]
    df = df[cols_final]
    print("Distribusi sentimen:")
    print(df['sentimen'].value_counts())
    # --- Robust file write: always use utf-8 encoding ---
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8')
    except Exception as e:
        raise RuntimeError(f"Gagal menulis file CSV: {e}")
    print(f"Selesai labeling! Hasil disimpan di: {output_csv}")
    print(f"Waktu proses: {time.time() - start:.2f} detik")
    return df

# Contoh pemakaian (script)
if __name__ == '__main__':
    df = preprocess_and_label_csv(
        r'C:\Users\X395\OneDrive\Documents\KULIAH SEMESTER 4\TUGAS\PRAKTIKUM AI\tugasuts_rnn\dataset\dataset_tiktok-comments-scraper-task_2025-05-01_09-17-35-852.csv',
        r'C:\Users\X395\OneDrive\Documents\KULIAH SEMESTER 4\TUGAS\PRAKTIKUM AI\tugasuts_rnn\dataset\komentar_labeled.csv',
        text_col='text'
    )
    print(df[['text', 'text_clean', 'sentimen', 'emosi']].head())
