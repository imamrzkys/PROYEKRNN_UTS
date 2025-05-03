import pandas as pd

# Baca dataset besar
input_path = 'dataset/komentar_labeled.csv'
output_path = 'dataset/komentar_labeled_small.csv'

try:
    df = pd.read_csv(input_path)
    n = min(1000, len(df))  # Maksimal 1000 baris atau semua jika kurang dari 1000
    df_small = df.sample(n=n, random_state=42)
    df_small.to_csv(output_path, index=False)
    print(f"Dataset kecil ({n} baris) berhasil disimpan ke {output_path}")
except Exception as e:
    print(f"Gagal membuat dataset kecil: {e}")
