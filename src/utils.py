import pandas as pd
from src.preprocessing import preprocess_text
from tqdm import tqdm  

# Keyword heuristik sederhana untuk labeling
POSITIVE_WORDS = [
    'bagus', 'mantap', 'keren', 'hebat', 'top', 'welcome', 'semangat', 'setuju', 'percaya', 'total football', 'selamat', 'good', 'sukses', 'support', 'percaya', 'bravo', 'terbaik', 'yes', 'oke', 'legend', '🔥', '❤', 'love', 'amazing', 'wow', 'super', 'harapan', 'semoga', 'maju', 'menang', 'juara'
]
NEGATIVE_WORDS = [
    'out', 'blunder', 'kontol', 'gagal', 'jelek', 'buruk', 'parah', 'anjir', 'anjirr', 'anjirr', 'anjing', 'gblk', 'goblok', 'gila', 'kacau', 'kalah', 'pecat', 'benci', 'tidak', 'ga', 'gak', 'no', 'down', 'dark', 'era kegelapan', 'cemas', 'balikin', 'salah', 'ngaco', 'ngawur', 'nggak', 'ngga', 'menyedihkan', 'ngapain', 'mending', 'keluar', 'quit', 'fomo', 'fail', 'nggk', 'nggak', 'ngga', 'gk', 'gak', 'ga', 'petrik out', 'kluivert out', 'bodo', 'becus', 'bodoh', 'tolol', 'parah', 'payah', 'lemah', 'malas', 'malu', 'hina', 'sampah', 'bangsat', 'brengsek', 'anjg', 'anj', 'kampret', 'sial', 'menyebalkan', 'menjijikkan', 'sakit hati', 'kecewa', 'tidak suka', 'tidak setuju', 'tidak mampu', 'ga becus', 'ga bisa', 'ga guna', 'ga jelas', 'ga paham', 'ga ngerti', 'ga penting', 'ga suka', 'ga setuju', 'ga mampu', 'ga layak'
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
    pos = any(word in t for word in POSITIVE_WORDS)
    neg = any(word in t for word in NEGATIVE_WORDS)
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
    print(f"Mulai proses labeling dataset: {input_csv}")
    start = time.time()
    df = pd.read_csv(input_csv)
    print(f"Jumlah data sebelum drop NA: {len(df)}")
    df = df.dropna(subset=[text_col])
    print(f"Jumlah data setelah drop NA: {len(df)}")
    tqdm.pandas(desc="Preprocessing")
    df['text_clean'] = df[text_col].astype(str).progress_apply(preprocess_text)
    tqdm.pandas(desc="Labeling Sentimen")
    df['sentimen'] = df['text_clean'].progress_apply(label_sentiment)
    tqdm.pandas(desc="Labeling Emosi")
    df['emosi'] = df['text_clean'].progress_apply(label_emotion)
    df.to_csv(output_csv, index=False)
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
