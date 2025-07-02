import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

# Cleaning function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Load data for wordcloud
text_df = pd.read_csv("wordcloud.csv")

# Load model and vectorizer
with open('disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Label dictionary
label_dict = {
    0: 'Depression',
    1: 'Diabetes, Type 2',
    2: 'High Blood Pressure'
}

# Streamlit Tabs
tab0, tab1 = st.tabs(["WordCloud", "Predict Disease"])

# WordCloud Tab
with tab0:
    st.title('WordCloud')
    if st.button("Generate Word Cloud"):
        text = " ".join(text_df['full_text'].astype(str))
        wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='Pastel1').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud)
        ax.axis('off')
        st.pyplot(fig)

# Prediction Tab
with tab1:
    st.title('Disease Prediction With Review')
    user_input = st.text_input("Enter your review:")

    if st.button("Predict"):
        cleaned_input = clean_text(user_input)

        if not cleaned_input.strip():
            st.warning("Invalid input: Please enter meaningful text to predict the condition.")
        else:
            vectorized_input = vectorizer.transform([cleaned_input])
            pred = model.predict(vectorized_input)
            label = label_dict[pred[0]]
            st.success(f"Predicted Condition: {label}")

    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter a review to analyze sentiment.")
        else:
            blob = TextBlob(user_input)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "Positive 😊"
            elif polarity < 0:
                sentiment = "Negative 😞"
            else:
                sentiment = "Neutral 😐"
            st.write(f"**Sentiment:** {sentiment}")

    if st.button("Generate WordCloud"):
        cleaned = clean_text(user_input)
        if cleaned.strip():
            st.markdown('**WordCloud for User Input**')
            wordcloud_user = WordCloud(width=800, height=400, background_color='black', colormap='Pastel1').generate(cleaned)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.imshow(wordcloud_user)
            ax1.axis('off')
            st.pyplot(fig1)
        else:
            st.warning("No meaningful words left after cleaning. Try entering more descriptive text.")
