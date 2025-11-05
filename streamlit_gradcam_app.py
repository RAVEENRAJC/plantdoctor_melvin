import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import io
import base64

# Page config
st.set_page_config(
    page_title="Plant Disease Grad-CAM Explorer",
    page_icon="🌿",
    layout="wide",
)

# ---------- Helper functions ----------
@st.cache_resource
def load_model_from_file(model_path: str):
    """Load a Keras model from given path. Cached for performance."""
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def load_class_names_from_dir(directory_path: str):
    if not os.path.isdir(directory_path):
        return None
    classes = sorted([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
    class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
    return {v: k for k, v in class_indices.items()}

def preprocess_pil_image(pil_img: Image.Image, target_size=(299, 299)):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(target_size)
    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def generate_gradcam_heatmap(model, img_array, class_index, last_conv_layer_name='mixed10'):
    # Build a model that maps the input image to the activations of the last conv layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise ValueError(f"Could not find layer '{last_conv_layer_name}'. Check available layer names. Error: {e}")

    grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    # Weigh the convolution outputs with the averaged gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.sum(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap /= np.max(heatmap)
    return heatmap


def overlay_heatmap(original_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # original_img: numpy uint8 HxWx3 (RGB)
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    # cv2 uses BGR; convert original to BGR for correct overlay then convert back
    orig_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb, heatmap_color


def get_top_k_predictions(preds, k=5):
    preds = np.squeeze(preds)
    top_k = preds.argsort()[-k:][::-1]
    return [(int(i), float(preds[i])) for i in top_k]


def to_bytes_image(np_img):
    pil = Image.fromarray(np_img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    byte_im = buf.getvalue()
    return byte_im


# ---------- UI ----------
st.title("🌿 Plant Disease Diagnosis — Grad-CAM Explorer")
st.write("Upload an image or use a sample, load your trained InceptionV3 model (.h5), and inspect model explanations (Grad-CAM).")

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    model_path_input = st.text_input("Model path (relative to app root)", value="inception_model_trained.h5")
    uploaded_model = st.file_uploader("Or upload model (.h5)", type=["h5"], help="If you upload a model file it will be used instead of the model path.")
    uploaded_sample_dir = st.text_input("Optional: directory with labelled test images (for auto class names)", value="")

    last_conv_layer_name = st.text_input("Last conv layer name", value="mixed10", help="Change if your model uses a different block name.")
    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.4)
    top_k = st.slider("Top-K predictions to show", 1, 10, 5)
    colormap_options = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "SPRING": cv2.COLORMAP_SPRING,
        "SUMMER": cv2.COLORMAP_SUMMER,
        "AUTUMN": cv2.COLORMAP_AUTUMN,
    }
    colormap_name = st.selectbox("Colormap", list(colormap_options.keys()), index=0)
    st.markdown("---")
    st.caption("Tip: If the model fails to load on Streamlit Cloud, upload the .h5 file directly using the uploader above.")

# Load class names if directory provided
class_names = None
if uploaded_sample_dir:
    class_names = load_class_names_from_dir(uploaded_sample_dir)

# Load or receive model file
model = None
if uploaded_model is not None:
    # Save uploaded model to a temp file and load
    st.info("Loading uploaded model — this may take a while.")
    temp_model_path = os.path.join("/tmp", uploaded_model.name)
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    try:
        model = load_model_from_file(temp_model_path)
        st.success("Model loaded from uploaded file.")
    except Exception as e:
        st.error(f"Failed to load uploaded model: {e}")
else:
    # Try to load from provided path
    if os.path.exists(model_path_input):
        try:
            model = load_model_from_file(model_path_input)
            st.success(f"Model loaded from: {model_path_input}")
        except Exception as e:
            st.error(f"Failed to load model from path: {e}")
    else:
        st.warning("Model not found at the path. Upload a model (.h5) or push it to the app repository.")

# Image input
st.markdown("## Image input")
col1, col2 = st.columns([1, 3])
with col1:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    use_sample = st.checkbox("Use sample image shipped in repo (if available)")
    sample_path = st.text_input("Optional sample image path (relative)", value="")
    if use_sample and sample_path and os.path.exists(sample_path):
        try:
            sample_img = Image.open(sample_path)
            st.image(sample_img, caption="Sample image (repo)")
        except Exception:
            st.warning("Could not load sample image from path.")

with col2:
    if uploaded_image is None and not (use_sample and sample_path and os.path.exists(sample_path)):
        st.info("Upload an image above or provide a sample image path.")

# Process on button click
run_button = st.button("Run Prediction & Grad-CAM")

if run_button:
    if model is None:
        st.error("No model available. Please upload a .h5 model or specify a valid model path.")
    else:
        # Acquire image
        if uploaded_image is not None:
            pil_img = Image.open(uploaded_image)
        elif use_sample and sample_path and os.path.exists(sample_path):
            pil_img = Image.open(sample_path)
        else:
            st.error("No image provided.")
            st.stop()

        # Preprocess
        img_array = preprocess_pil_image(pil_img, target_size=(299, 299))

        # Predict
        preds = model.predict(img_array)
        topk = get_top_k_predictions(preds, k=top_k)

        # Attempt to resolve class names mapping
        if class_names is None:
            # try to infer number of classes and create generic labels
            num_classes = preds.shape[-1]
            class_names = {i: f"Class_{i}" for i in range(num_classes)}

        predicted_idx = int(np.argmax(preds, axis=1)[0])
        predicted_label = class_names.get(predicted_idx, f"Class_{predicted_idx}")

        st.success(f"Predicted: {predicted_label}")

        # Try Grad-CAM
        try:
            heatmap = generate_gradcam_heatmap(model, img_array, predicted_idx, last_conv_layer_name=last_conv_layer_name)
        except Exception as e:
            st.error(f"Grad-CAM generation failed: {e}")
            heatmap = np.zeros((7, 7))

        # Convert original PIL to numpy RGB
        orig_rgb = np.array(pil_img.convert('RGB').resize((299, 299))).astype(np.uint8)

        overlay_img, heatmap_color = overlay_heatmap(orig_rgb, heatmap, alpha=alpha, colormap=colormap_options[colormap_name])

        # Layout results
        st.markdown("### Results")
        left, right = st.columns([1, 1])
        with left:
            st.image(orig_rgb, caption=f"Input Image — Predicted: {predicted_label}")
            st.download_button("Download original image", data=to_bytes_image(orig_rgb), file_name="original.png", mime="image/png")
        with right:
            st.image(overlay_img, caption="Grad-CAM Overlay")
            st.download_button("Download overlay image", data=to_bytes_image(overlay_img), file_name="gradcam_overlay.png", mime="image/png")

        st.markdown("---")
        st.markdown("### Prediction scores (Top-K)")
        for idx, score in topk:
            lbl = class_names.get(idx, f"Class_{idx}")
            st.write(f"**{lbl}** — {score:.4f}")

        st.markdown("---")
        st.markdown("### Debug / Notes")
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
        st.write(f"Last conv layer used: {last_conv_layer_name}")

        st.info("If the heatmap looks blank, try a different `last_conv_layer_name` (check your model architecture) or ensure the image is centered and similar to training images.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit — drop your .h5 model into the app root or upload it to try live explanations.")


