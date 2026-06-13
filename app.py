# app.py

import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_models():
    classifier = joblib.load(
        "models/sentiment.pkl"
    )
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    return classifier, embedder
classifier, embedder = load_models()

st.title(
    "Movie Review Sentiment Analyzer"
)
review = st.text_area(
    "Enter Review"
)

if st.button("Analyze"):
    embedding = embedder.encode([review])
    prediction = classifier.predict(
        embedding
    )[0]
    probabilities = classifier.predict_proba(embedding)[0]

    positive_prob = probabilities[1] * 100
    negative_prob = probabilities[0] * 100

    if prediction == 1:
        st.success("Positive Review")
    else:
        st.error("Negative Review")
    
    st.metric(
        "Confidence",
        f"{max(probabilities)*100:.2f}%"
    )
    st.write(f"Positive Probability: {positive_prob:.2f}%")
    st.write(f"Negative Probability: {negative_prob:.2f}%")

    