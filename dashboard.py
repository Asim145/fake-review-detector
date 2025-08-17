import streamlit as st
import joblib, numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk, os

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️", layout="centered")

# Ensure NLTK data exists on first run (no 'punkt' needed with RegexpTokenizer)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

tokenizer = RegexpTokenizer(r"\w+")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    tokens = tokenizer.tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_artifacts():
    model = joblib.load("fake_review_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, tfidf, scaler

model, tfidf, scaler = load_artifacts()

st.title("🕵️‍♂️ Fake Review Detector")
st.write("Enter a product review to check if it's fake or genuine.")

review = st.text_area("Review Text", height=150)

if st.button("Analyze"):
    cleaned = clean_text(review)
    vec = tfidf.transform([cleaned])
    sentiment = TextBlob(review).sentiment.polarity
    length = len(review)
    exclam = review.count("!")
    meta = scaler.transform([[sentiment, length, exclam]])
    x = np.hstack((vec.toarray(), meta))
    pred = model.predict(x)[0]
    st.success("✅ Genuine Review" if pred == 0 else "⚠️ Fake Review")
