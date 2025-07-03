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

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z:]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)
    
# Load model
with open('disease_model.pkl', 'rb') as file:
    model = pickle.load(file)
    
# Label dictionary
label_dict = {
  0: 'Depression',
  1: 'Diabetes, Type 2',
  2: 'High Blood Pressure'
}

# Load WordCloud data
text_df = pd.read_csv("wordcloud.csv")

# Streamlit UI
tab0, tab1 = st.tabs(["WordCloud", "Predict Disease"])
    
with tab0:
    st.title('WordCloud')
    if st.button("Generate Word Cloud"):
        text = " ".join(text_df['full_text'].astype(str))
        wordcloud = WordCloud(width=1000, height=600, background_color='black', colormap='Pastel1').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud)
        ax.axis('off')
        st.pyplot(fig)

with tab1:
    st.title('Disease Prediction With Review')
    user_input = st.text_input("Enter your review:")
    
    if st.button("Predict"):
        cleaned_input = clean_text(user_input)
        if cleaned_input.strip():  # Make sure there's valid input
            pred = model.predict([cleaned_input])
            label = label_dict[pred[0]]
            st.success(f"Predicted Condition: {label}")
        else:
            st.warning("Input doesn't contain enough meaningful words. Please write a proper review.")
            st.caption("Hint: Please enter a detailed review mentioning drug name, condition, or symptoms.")

    if st.button("Analyze Sentiment"):
       blob = TextBlob(user_input)
       polarity = blob.sentiment.polarity
       if polarity > 0:
           sentiment = "Positive 😊"
       elif polarity < 0:
           sentiment = "Negative 😞"
       else:
           sentiment = "Neutral 😐"
       st.write(f"**Sentiment:** {sentiment}")

    if st.button('Generate WordCloud of Review'):
        cleaned = clean_text(user_input)
        if cleaned.strip():
            wordcloud_user = WordCloud(width=800, height=400, background_color='black', colormap='Pastel1').generate(cleaned)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.imshow(wordcloud_user)
            ax1.axis('off')
            st.pyplot(fig1)
        else:
            st.warning("No meaningful words found. Try entering a longer review.")
