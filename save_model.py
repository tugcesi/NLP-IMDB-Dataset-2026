"""
save_model.py – IMDB Sentiment Classifier
Sadece yerel bilgisayarda çalıştırılır.
Üretilen model.joblib ve vectorizer.joblib dosyalarını HF'e yükle.

Çalıştır: python save_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# utils.py'den import et — joblib 'utils.lemmatize_tokens' olarak kaydeder ✓
from utils import clean_text, lemmatize_tokens

# ── 1. Veri Yükle ─────────────────────────────────────────────────────────────
print("📂 Veri yükleniyor...")
train = pd.read_csv('train.csv')
val   = pd.read_csv('val.csv')
print(f"   Train: {train.shape} | Val: {val.shape}")

# ── 2. Label Encode ───────────────────────────────────────────────────────────
sentiment_map = {'positive': 1, 'negative': 0}
train['label'] = train['sentiment'].map(sentiment_map)
val['label']   = val['sentiment'].map(sentiment_map)

# ── 3. Metin Temizleme ────────────────────────────────────────────────────────
print("⚙️  Temizleniyor...")
train['review'] = train['review'].apply(clean_text)
val['review']   = val['review'].apply(clean_text)

# ── 4. Vektörizasyon ──────────────────────────────────────────────────────────
print("🔢 Vektörizasyon (CountVectorizer ngram 1-2 + lemmatize)...")
vect = CountVectorizer(
    ngram_range=(1, 2),
    analyzer=lemmatize_tokens,   # utils.lemmatize_tokens olarak kaydedilir ✓
    stop_words='english'
)
x_train = vect.fit_transform(train['review'])
x_val   = vect.transform(val['review'])

y_train = train['label']
y_val   = val['label']

print(f"   Vektör boyutu: {x_train.shape}")

# ── 5. Model Eğitimi ──────────────────────────────────────────────────────────
print("🚀 Model eğitiliyor (Logistic Regression)...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(x_train, y_train)

# ── 6. Değerlendirme (Val seti) ───────────────────────────────────────────────
preds = model.predict(x_val)
print(f"\n📊 Val Accuracy : {accuracy_score(y_val, preds):.4f}")
print(classification_report(y_val, preds, target_names=['Negative', 'Positive']))

# ── 7. Artifact Kaydet ────────────────────────────────────────────────────────
joblib.dump(model, 'model.joblib')
joblib.dump(vect,  'vectorizer.joblib')
print("✅ model.joblib ve vectorizer.joblib kaydedildi → HF'e yükle!")