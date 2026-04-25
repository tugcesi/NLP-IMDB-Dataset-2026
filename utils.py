"""
utils.py – Paylaşılan NLP pipeline (IMDB Sentiment).
Çalıştırılmaz; save_model.py ve app.py tarafından import edilir.
"""

import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)   # TextBlob lemmatization
nltk.download('omw-1.4',   quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Notebook'taki clean_text fonksiyonu ile birebir aynı."""
    text = str(text).lower()
    text = re.sub(r'<.*?>',   '', text)   # HTML tagları (<br/> vb.)
    text = re.sub(r'[^\w\s]', '', text)   # noktalama
    text = re.sub(r'\d+',     '', text)   # rakamlar
    text = re.sub(r'\n|\r',   '', text)   # satır sonu
    return text.strip()

def lemmatize_tokens(text: str):
    """Notebook'taki 'ekkok' fonksiyonu — joblib için utils modülünde tanımlı."""
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]