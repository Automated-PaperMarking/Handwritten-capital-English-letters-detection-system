import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load trained CNN model
@st.cache_resource
def load_my_model():
    return load_model("model/handwritten_capital_model.h5")

model = load_my_model()

# App title
st.set_page_config(page_title="Handwritten Capital Letter Recognition", layout="centered")
st.title("🖊️ Handwritten Capital Letter Recognition")
st.write("Upload an image of a **handwritten capital letter (A–Z)** to identify it. "
         "Make sure it's on a **white background**, and the letter is dark or black.")

# File uploader
uploaded_file = st.file_uploader("Upload a PNG/JPG image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Step 1: Load and display uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', width=200)

    # Step 2: Preprocess
    image = image.resize((28, 28))          # Resize to 28x28
    image = ImageOps.invert(image)          # Invert (black bg, white letter)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Step 3: Show preprocessed image
    st.subheader("🧪 Preprocessed Image")
    st.image(img_array.reshape(28, 28), width=150, clamp=True, caption="What the model sees")

    # Step 4: Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_letter = chr(predicted_index + 65)  # A-Z

    st.success(f"✅ Predicted Letter: **{predicted_letter}**")

    # Step 5: Confidence display
    st.subheader("📊 Prediction Confidence (Top 5)")
    top_5 = prediction[0].argsort()[-5:][::-1]
    for i in top_5:
        conf = prediction[0][i] * 100
        st.write(f"**{chr(i + 65)}**: {conf:.2f}%")
