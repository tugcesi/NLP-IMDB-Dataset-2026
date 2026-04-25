"""
app.py – IMDB Sentiment Classifier (Streamlit / Hugging Face Spaces)
HF Spaces bu dosyayı otomatik çalıştırır.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# utils.py'den import et — joblib'in ihtiyaç duyduğu modül yolu burada ✓
from utils import clean_text, lemmatize_tokens

# ── Sayfa Ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="wide"
)

# ── Artifact Yükleme ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)
    mp = os.path.join(base_dir, 'model.joblib')
    vp = os.path.join(base_dir, 'vectorizer.joblib')
    if not os.path.exists(mp) or not os.path.exists(vp):
        return None, None, f"Dosyalar bulunamadı: {mp}"
    return joblib.load(mp), joblib.load(vp), None

model, vect, err = load_artifacts()

if err:
    st.error(f"⚠️ {err}")
    st.info("Yerel ortamda `python save_model.py` çalıştırıp .joblib dosyalarını repoya ekle.")
    st.stop()

# ── Tahmin Fonksiyonu ─────────────────────────────────────────────────────────
def predict_review(review: str):
    cleaned    = clean_text(review)
    vectorized = vect.transform([cleaned])
    pred       = int(model.predict(vectorized)[0])
    proba      = model.predict_proba(vectorized)[0]
    return pred, proba

# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🎬 IMDB Sentiment Classifier")
st.caption("Kaggle Sentiment Analysis on IMDB Dataset 2026 | Logistic Regression + CountVectorizer (ngram 1-2)")
st.divider()

# ── Örnek Yorumlar ────────────────────────────────────────────────────────────
POSITIVE_EXAMPLES = [
    "This movie was absolutely brilliant! The acting was superb and the story was captivating.",
    "One of the best films I have ever seen. A masterpiece of storytelling.",
    "Fantastic performances by the entire cast. I was moved to tears.",
]
NEGATIVE_EXAMPLES = [
    "This was a complete waste of time. The plot made no sense whatsoever.",
    "Terrible acting, boring story, and a disappointing ending. Avoid this film.",
    "One of the worst movies I have ever seen. Absolutely dreadful.",
]

# Session state
if 'review_input' not in st.session_state:
    st.session_state['review_input'] = ''

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 Film Yorumu Girin")

    review_input = st.text_area(
        "Yorum:",
        value=st.session_state['review_input'],
        placeholder="Örn: This movie was absolutely brilliant! The acting was superb...",
        height=180,
        label_visibility="collapsed"
    )

    st.caption("💡 Örnek yorumlar:")
    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown("🟢 **Pozitif**")
        for ex in POSITIVE_EXAMPLES:
            if st.button(f"📌 {ex[:38]}…", key=f"p_{ex}", use_container_width=True):
                st.session_state['review_input'] = ex
                st.rerun()

    with ex_col2:
        st.markdown("🔴 **Negatif**")
        for ex in NEGATIVE_EXAMPLES:
            if st.button(f"📌 {ex[:38]}…", key=f"n_{ex}", use_container_width=True):
                st.session_state['review_input'] = ex
                st.rerun()

    predict_btn = st.button(
        "🔍 Tahmin Et", type="primary", use_container_width=True,
        disabled=(len(review_input.strip()) == 0)
    )

# ── Sonuç ─────────────────────────────────────────────────────────────────────
with col_result:
    st.subheader("📊 Sonuç")

    if predict_btn and review_input.strip():
        try:
            pred, proba = predict_review(review_input)
            prob_positive = proba[1]
            prob_negative = proba[0]

            if pred == 1:
                st.success("### 😊 POZİTİF YORUM")
                result_label = "Pozitif"
                gauge_color  = "#16A34A"
            else:
                st.error("### 😞 NEGATİF YORUM")
                result_label = "Negatif"
                gauge_color  = "#DC2626"

            st.metric(
                "Tahmin", result_label,
                delta=f"Güven: %{max(prob_positive, prob_negative)*100:.1f}"
            )

            # Gauge — Pozitif olasılığı göster
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_positive * 100,
                title={'text': "Pozitif Olasılık (%)"},
                number={'suffix': "%", 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': gauge_color},
                    'steps': [
                        {'range': [0,  40], 'color': '#FFE4E6'},
                        {'range': [40, 60], 'color': '#FEF9C3'},
                        {'range': [60, 100],'color': '#D1FAE5'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 3},
                        'thickness': 0.8, 'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Bar chart
            fig_bar = go.Figure(go.Bar(
                x=[prob_negative * 100, prob_positive * 100],
                y=['Negatif 😞', 'Pozitif 😊'],
                orientation='h',
                marker_color=['#DC2626', '#16A34A'],
                text=[f"%{prob_negative*100:.1f}", f"%{prob_positive*100:.1f}"],
                textposition='inside',
                textfont=dict(color='white', size=14)
            ))
            fig_bar.update_layout(
                xaxis=dict(range=[0, 100], title="Olasılık (%)"),
                height=160,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Hata: {e}")
            st.exception(e)

    else:
        st.info("👈 Sol tarafa bir film yorumu girin ve **Tahmin Et** butonuna tıklayın.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model",      "Logistic Regression")
        m2.metric("Vektörizör", "CountVectorizer")
        m3.metric("N-gram",     "1–2")

# ── Metin Analiz Detayı ───────────────────────────────────────────────────────
if predict_btn and review_input.strip():
    st.divider()
    st.subheader("🔬 Metin Analiz Detayı")

    cleaned = clean_text(review_input)
    tokens  = lemmatize_tokens(cleaned)
    d1, d2  = st.columns(2)

    with d1:
        st.markdown("**Orijinal Yorum:**")
        st.text_area("", review_input, height=100, disabled=True, label_visibility="collapsed")
        st.markdown("**Temizlenmiş Metin:**")
        st.text_area("", cleaned,      height=100, disabled=True, label_visibility="collapsed")

    with d2:
        st.markdown(f"**Tokenlar ({len(tokens)} adet):**")
        if tokens:
            st.dataframe(
                pd.DataFrame({'Token': tokens, 'Uzunluk': [len(t) for t in tokens]}),
                hide_index=True, use_container_width=True, height=200
            )
        else:
            st.warning("Stopword temizleme sonrası token kalmadı.")

# ── Hakkında ──────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    **Kaynak:** [Kaggle – Sentiment Analysis on IMDB Dataset 2026](https://www.kaggle.com/competitions/sentiment-analysis-on-imdb-dataset-2026)

    | | |
    |---|---|
    | Train  | 35,000 satır |
    | Val    | 7,500 satır |
    | Hedef  | positive → 1 · negative → 0 |

    **Pipeline:**
    `clean_text` (HTML + noktalama + rakam) → `lemmatize_tokens` (TextBlob + NLTK stopwords) → `CountVectorizer(ngram 1-2)` → `LogisticRegression`
    """)