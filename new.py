import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

st.title('Emotion classification')
df=pd.read_csv('cleaned_emotion_classify_data.csv')
df

# ------------------------------------------------------------------------------------------------
# Training model
from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer()
X = df['Comment']
Y = df['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15) #Splitting dataset


# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=2000)),
                     ('clf', LogisticRegression())])


# #Training model
model = pipeline.fit(X_train, y_train)
# ---------------------------------------------------------------------------------------------------
text = st.text_area("Enter Text")
if st.button("Submit"):

    text_data = {'predict_news':[text]}
    text_data_df = pd.DataFrame(text_data)

    predicted_emotion = model.predict(text_data_df['predict_news']) 
    st.write("Predicted emotion = ",predicted_emotion[0])
else:
    st.write("please enter emotion")
