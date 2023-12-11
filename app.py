import streamlit as st
from transformers import pipeline

def analyze_sentiment(text):
    classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
    result = classifier(text)
    return {"label": result[0]["label"], "score": result[0]["score"]}

def main():
    st.title("Sentiment tonality analyzer")

    user_input = st.text_area("Enter text for tonality analysis:", "")

    if st.button("Analyze"):
        if user_input:
            result = analyze_sentiment(user_input)
            st.subheader("Analysis result:")
            st.write(f"Tonality: {result['label']}")
            st.write(f"Confidence: {result['score']:.4f}")
        else:
            st.warning("Please enter text for analysis.")
