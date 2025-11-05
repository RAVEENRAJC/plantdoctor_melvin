import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image

# =====================================================
# 🎯 APP CONFIGURATION
# =====================================================
st.set_page_config(page_title="🌿 Plant Doctor - Melvin", layout="wide")
st.title("🌿 Plant Disease Identifier (InceptionV3)")
st.markdown("Upload a leaf image to identify the disease using a pre-trained InceptionV3 model.")

# =====================================================
# 📥 MODEL DOWNLOAD & LOADING
# =====================================================
@st.cache_resource
def load_inception_model():
    model_path = "inception_model_trained.h5"
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive... please wait ⏳")
        # Google Drive Direct Download ID
        url = "https://drive.google.com/uc?id=1otdySTbPYGGiarh7rWNWwpobl_z2PXYF"
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    else:
        st.info("✅ Model already available locally.")

    model = tf.keras.models.load_model(model_path)
    st.success("🎯 Model loaded successfully!")
    return model

# Load model once
model = load_inception_model()

# =====================================================
# 🖼 IMAGE UPLOAD SECTION
# =====================================================
uploaded_file = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img_resized = cv2.resize(img, (224, 224))  # Adjust if model expects different size
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    # Predict
    st.info("🔍 Running model prediction...")
    preds = model.predict(img_expanded)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    # Display result
    st.subheader("🧠 Prediction Result")
    st.write(f"**Predicted Class ID:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.success("✅ Prediction complete!")

else:
    st.warning("Please upload an image to proceed.")

# =====================================================
# 📘 FOOTER
# =====================================================
st.markdown("---")
st.markdown("Made with ❤️ by **Raveen Raj & Melvin** | Powered by InceptionV3")

