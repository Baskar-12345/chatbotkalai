import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import streamlit as st
from PIL import Image
import io

# Load a pre-trained model from TensorFlow Hub (example)
model_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4'
model = hub.load(model_url)

# Define a function to preprocess the image for the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to the model's input size
    image_np = np.array(image) / 255.0  # Normalize the image
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

# Define a function to make predictions using the model
def predict_image(image):
    image_np = preprocess_image(image)
    predictions = model(image_np)
    predicted_class = np.argmax(predictions.numpy())
    return predicted_class

# Define a Streamlit app
def main():
    st.title("Image Recognition Chatbot")
    st.write("Upload an image to classify it.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Make predictions
        predicted_class = predict_image(image)
        
        # Display the result
        st.write(f"Predicted Class: {predicted_class}")
        
        # Provide a chatbot-like response based on the predicted class
        response = f"I detected class {predicted_class}. How can I assist you further?"
        st.write(response)

if __name__ == "__main__":
    main()
