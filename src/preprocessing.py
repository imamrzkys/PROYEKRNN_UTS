import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji

# Stopword Bahasa Indonesia (hardcoded)
stopword_list = set([
    'yang', 'untuk', 'dengan', 'pada', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'atau', 'juga', 'karena', 'ada', 'tidak', 'sudah', 'saja', 'sangat', 'lebih', 'agar', 'sebagai', 'jadi', 'oleh', 'kalau', 'dalam', 'bisa', 'akan', 'pada', 'mereka', 'kami', 'kita', 'anda', 'saya', 'dia', 'aku', 'kamu', 'pun', 'apa', 'siapa', 'apa', 'berapa', 'bagaimana', 'mengapa', 'semua', 'hanya', 'masih', 'setelah', 'sebelum', 'harus', 'dapat', 'lah', 'punya', 'kok', 'ya', 'nih', 'deh', 'dong', 'aja', 'kayak', 'banget', 'mah', 'lah', 'pun', 'si', 'biar', 'makanya', 'udah', 'juga', 'kan', 'toh', 'gue', 'gua', 'loe', 'lu', 'lo', 'kalo', 'kalau', 'tau', 'tahu', 'sampe', 'sampai', 'dgn', 'dr', 'tp', 'trs', 'trus', 'krn', 'jd', 'jadi', 'pd', 'pada', 'utk', 'untuk', 'dg', 'dengan', 'gw', 'bro', 'sis', 'om', 'gan', 'min', 'admin', 'ga', 'gak', 'nggak', 'ngga', 'gk', 'aja', 'udah', 'udah', 'makasih', 'terima kasih', 'thanks', 'thank', 'you', 'please', 'ok', 'oke', 'yes', 'no', 'iya', 'enggak', 'tidak', 'nggak', 'gak', 'ga', 'kok', 'lah', 'deh', 'dong', 'nih', 'mah', 'pun', 'si', 'biar', 'makanya', 'udah', 'juga', 'kan', 'toh', 'gue', 'gua', 'loe', 'lu', 'lo', 'kalo', 'kalau', 'tau', 'tahu', 'sampe', 'sampai', 'dgn', 'dr', 'tp', 'trs', 'trus', 'krn', 'jd', 'jadi', 'pd', 'pada', 'utk', 'untuk', 'dg', 'dengan', 'gw', 'bro', 'sis', 'om', 'gan', 'min', 'admin'
])
def get_stopwords():
    custom_stopwords = set([
        'yg', 'ga', 'gak', 'nya', 'nih', 'sih', 'deh', 'dong', 'aja', 'kayak', 'banget', 'dah', 'mah', 'lah', 'pun', 'si', 'biar', 'makanya', 'udah', 'udah', 'juga', 'kan', 'toh', 'kok', 'eh', 'ya', 'loh', 'gue', 'gua', 'loe', 'lu', 'lo', 'kalo', 'kalau', 'tau', 'tahu', 'sampe', 'sampai', 'dgn', 'dr', 'tp', 'trs', 'trus', 'krn', 'karena', 'jd', 'jadi', 'pd', 'pada', 'utk', 'untuk', 'dg', 'dengan', 'aja', 'aja', 'gw', 'gw', 'bro', 'sis', 'om', 'gan', 'min', 'admin'
    ])
    return stopword_list.union(custom_stopwords)

STOPWORDS = get_stopwords()

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove emoji & non-ASCII (pakai emoji library dan regex unicode)
    try:
        import emoji
        text = emoji.replace_emoji(text, replace='')
    except ImportError:
        pass  # fallback jika emoji lib tidak ada
    # Remove emoji unicode (regex unicode block, lebih luas)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF\u2000-\u206F\u2190-\u21FF\u2300-\u23FF\u2B50\u25A0-\u25FF]+', '', text, flags=re.UNICODE)
    # Remove non-ASCII (backup)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove angka
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    # Tokenisasi sederhana tanpa NLTK
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def stem_tokens(tokens):
    # Sastrawi bekerja per kalimat, jadi join dulu lalu split lagi
    return stemmer.stem(' '.join(tokens)).split()

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)

# Contoh penggunaan
if __name__ == '__main__':
    sample = "emang se bagus apah patrick kluivert?11 😹"  # Contoh komentar
    print("Asli:", sample)
    print("Preprocessing:", preprocess_text(sample))
