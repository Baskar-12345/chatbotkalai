# -*- coding: utf-8 -*-
"""chatbot.py"""

import os
import nltk
import ssl
import streamlit as st
import random
import json
import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import tensorflow_hub as hub

# Handle SSL issues
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load JSON data
with open('intents.json', 'r') as file:
    data = json.load(file)

# Vectorizer and Classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

intents = data['intents']  # Extract the list of intents

tags = []
patterns = []

# Iterate through each intent in the list
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Load image recognition model from TensorFlow Hub
model_url = 'https://tfhub.dev/google/imagenet/inception_v3/classification/4'
model = hub.load(model_url)

def preprocess_image(image):
    image = image.resize((299, 299))
    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

def predict_image(image):
    image_np = preprocess_image(image)
    predictions = model(image_np)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Chatbot with Image Recognition")
    
    # Text-based interaction
    user_input = st.text_input("You:")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100)
        
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thanks")
            st.stop()

    # Image-based interaction
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction = predict_image(image)
        st.write(f"Predicted class index: {prediction}")
        
        # Map the prediction index to specific responses if needed
        st.write("Response based on image recognition...")

if __name__ == '__main__':
    main()
