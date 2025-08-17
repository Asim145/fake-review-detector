# dashboard.py
# ------------------------------
# Streamlit app for Fake Review Detection
# - Cloud-safe NLTK downloads (no 'punkt' needed)
# - Cached loading of model/vectorizer/scaler
# - Same preprocessing as in training (TF-IDF + sentiment + length + '!')
# ------------------------------

import os
import numpy as np
import joblib
import streamlit as st
from textblob import TextBlob
import nltk

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️", layout="centered")

# ---------- Ensure NLTK data exists (Cloud-safe) ----------
NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
# Make NLTK look in our local dir first
nltk.data.path = [NLTK_DIR] + nltk.data.path

@st.cache_resource(show_spinner=False)
def ensure_nltk_and_tools():
    """Download minimal NLTK corpora required and return tokenizer/stopwords/lemmatizer.
       Falls back to sklearn stopwords if NLTK download fails (keeps app alive)."""
    # Try to download the corpora we actually use (NO 'punkt' needed)
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, download_dir=NLTK_DIR, quiet=True)
            except Exception:
                pass  # we'll handle fallback below

    # Prefer NLTK tools if available
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import RegexpTokenizer
        tokenizer = RegexpTokenizer(r"\w+")
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        use_fallback = False
    except Exception:
        # --- Fallback: no NLTK at all (works in tight environments) ---
        import re
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        class NoOpLemmatizer:
            def lemmatize(self, x): return x
        class SimpleTokenizer:
            def tokenize(self, text): return re.findall(r"\w+", str(text).lower())

        tokenizer = SimpleTokenizer()
        stop_words = set(ENGLISH_STOP_WORDS)
        lemmatizer = NoOpLemmatizer()
        use_fallback = True

    return tokenizer, stop_words, lemmatizer, use_fallback

tokenizer, stop_words, lemmatizer, using_fallback = ensure_nltk_and_tools()

# ---------- Load serialized artifacts ----------
@st.cache_resource(show_spinner=True)
def load_artifacts():
    """Load model, TF-IDF vectorizer, and scaler saved from training."""
    try:
        model = joblib.load("fake_review_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, tfidf, scaler
    except FileNotFoundError as e:
        st.error(
            "❌ Missing artifact: " + str(e) +
            "\n\nPlace `fake_review_model.pkl`, `tfidf_vectorizer.pkl`, and `scaler.pkl` "
            "in the same folder as `dashboard.py`, then redeploy."
        )
        st.stop()

model, tfidf, scaler = load_artifacts()

# ---------- Text cleaning to match training ----------
def clean_text(text: str) -> str:
    tokens = tokenizer.tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# ---------- UI ----------
st.title("🕵️‍♂️ Fake Review Detector")
st.caption("Type/paste a review below and click **Analyze** to classify it as Genuine or Fake.")

# Show environment note if fallback is active
if using_fallback:
    st.info("Running with lightweight tokenizer/stopwords (fallback). "
            "If your training used NLTK lemmatization, results may differ slightly. "
            "To re-enable NLTK on cloud, ensure downloads succeed.")

with st.expander("What happens under the hood?"):
    st.markdown(
        "- Your text is cleaned and lemmatized (where available).\n"
        "- We generate **TF-IDF** features and compute 3 metadata features:\n"
        "  - Sentiment polarity (TextBlob)\n"
        "  - Review length (characters)\n"
        "  - Exclamation count (`!`)\n"
        "- The numeric features are scaled with the same **scaler** used in training.\n"
        "- Final features = [TF-IDF] + [sentiment, length, exclamations]."
    )

review = st.text_area("Review Text", height=160, placeholder="e.g., This product is absolutely amazing!!!")

def build_features(raw_text: str):
    """Replicate the training feature pipeline for a single text."""
    cleaned = clean_text(raw_text)
    vec = tfidf.transform([cleaned])                         # sparse -> toarray() before stacking
    sentiment = TextBlob(raw_text).sentiment.polarity
    length = len(raw_text)
    exclam = raw_text.count("!")
    meta = scaler.transform([[sentiment, length, exclam]])   # shape (1, 3)
    features = np.hstack((vec.toarray(), meta))              # shape (1, n_tfidf + 3)
    return features

# Optional: try to show a confidence score if available
def safe_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1][0]
        return "prob", float(proba)
    elif hasattr(model, "decision_function"):
        # Map decision score to 0-1 via logistic sigmoid (for display only)
        score = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-score))
        return "score", float(prob)
    else:
        return None, None

col1, col2 = st.columns([1, 1], vertical_alignment="bottom")
with col1:
    run = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

if run:
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            X = build_features(review)
            pred = int(model.predict(X)[0])
            mode, conf = safe_score(model, X)

        if pred == 1:
            st.error("⚠️ **Prediction: Fake Review**")
        else:
            st.success("✅ **Prediction: Genuine Review**")

        if conf is not None:
            if mode == "prob":
                st.caption(f"Model confidence (probability of *Fake*): **{conf:.3f}**")
            else:
                st.caption(f"Model confidence (scaled score for *Fake*): **{conf:.3f}**")

        with st.expander("Show extracted features"):
            sentiment = TextBlob(review).sentiment.polarity
            length = len(review)
            exclam = review.count("!")
            st.write(
                {
                    "sentiment": round(sentiment, 4),
                    "review_length": length,
                    "exclamation_count": exclam
                }
            )

st.divider()
st.caption(
    "Tip: To speed up cloud builds, keep only the packages your **dashboard** needs in `requirements.txt`. "
    "If the dashboard doesn’t use XGBoost/LightGBM, remove them."
)

