import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Path ke dataset yang ada label sentimen
df = pd.read_csv('dataset/komentar_labeled.csv')

# Pastikan kolom sentimen ada
if 'sentimen' not in df.columns:
    raise ValueError('Kolom sentimen tidak ditemukan di dataset!')

label_encoder = LabelEncoder()
label_encoder.fit(df['sentimen'])

joblib.dump(label_encoder, 'label_encoder.pkl')
print('label_encoder.pkl berhasil dibuat!')
