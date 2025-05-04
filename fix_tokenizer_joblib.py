import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

# Coba load tokenizer dengan np.load (jika sebelumnya disimpan dengan np.save)
try:
    tokenizer = np.load('tokenizer_lstm.npy', allow_pickle=True).item()
    print('DEBUG: Berhasil load tokenizer dengan np.load')
except Exception as e:
    print('DEBUG: Gagal load dengan np.load, coba joblib.load')
    try:
        tokenizer = joblib.load('tokenizer_lstm.npy')
        print('DEBUG: Berhasil load tokenizer dengan joblib.load')
    except Exception as e2:
        print('ERROR: Tidak bisa load tokenizer sama sekali!')
        raise e2

# Simpan ulang dengan joblib.dump agar kompatibel Railway
joblib.dump(tokenizer, 'tokenizer_lstm.npy')
print('DEBUG: Tokenizer berhasil di-save ulang dengan joblib.dump!')
