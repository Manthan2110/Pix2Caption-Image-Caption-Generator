import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import os
from PIL import Image
from io import BytesIO
import requests
import tempfile


# --- Page Configuration ---
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Typography Enhancement (CSS) ---
st.markdown("""
<style>
    html, body, [class*='st-']  { font-size: 1.1rem; }
    h1 { font-size: 2.6rem !important; }
    h2, h3 { font-size: 1.8rem !important; }
    [data-testid='stSidebar'] h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("📖 About This Project")
    st.info(
        "This application is a real-time image captioning tool that uses a sophisticated deep learning model to 'see' and describe an image in plain English."
    )
    
    # st.header("⚙️ Technology Stack")
    # st.markdown("""
    # This project is built on a foundation of cutting-edge technologies:
    # - **Library**: Keras with a TensorFlow backend for building and training the neural network.
    # - **Architecture**: A CNN (DenseNet-201) for powerful image feature extraction, paired with an LSTM for sequential text generation.
    # - **Dataset**: Trained on the extensive Flickr8k dataset, which contains thousands of images with multiple human-generated captions.
    # - **App Framework**: Streamlit, for creating this interactive and responsive web application.
    # """)
    
    st.header("👨‍💻 Developer")
    st.markdown("Developed with ❤️ by Manthan Jadav")

# --- Model Loading ---
@st.cache_resource
def load_all_models_local():
    """Load models and tokenizer from the local directory."""
    try:
        caption_model = load_model("models/caption_model_.keras")
        feature_extractor = load_model("models/feature_extractor_.keras")
        with open("models/tokenizer_.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return caption_model, feature_extractor, tokenizer
    except Exception as e:
        st.error(f"Error loading local models: {e}")
        return None, None, None


# --- Caption Generation ---
def generate_caption(image_bytes, _models, max_length=34, img_size=224):
    """Generate a caption for the provided image bytes."""
    caption_model, feature_extractor, tokenizer = _models
    try:
        img = load_img(BytesIO(image_bytes), target_size=(img_size, img_size))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        image_features = feature_extractor.predict(img_array, verbose=0)

        in_text = "start"
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = caption_model.predict([image_features, sequence], verbose=0)
            yhat_index = np.argmax(yhat)
            word = tokenizer.index_word.get(yhat_index, None)
            
            if word is None or word == "end":
                break
            
            in_text += " " + word
        
        caption = in_text.replace("start", "").strip()
        return caption.capitalize() + "."
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return None

# --- Main Application ---
def main():
    # --- Header ---
    st.title("🖼️ AI Image Caption Generator")
    st.write("**Welcome! Ever wondered how a computer sees the world? This tool gives you a glimpse.** ")
    st.write("Upload any image, and our advanced AI model will analyze its contents and generate a human-like descriptive caption. "
        "It's the perfect showcase of how Computer Vision and Natural Language Processing work together."
    )

    # # --- How It Works Expander ---
    # with st.expander("🤔 Curious about the technology? Click here to learn more!"):
    #     st.markdown("""
    #     The magic behind this app is an **Encoder-Decoder** model, a popular architecture for sequence-to-sequence tasks. Here's a simplified breakdown:

    #     #### 1. The Encoder: Seeing the Image
    #     A **Convolutional Neural Network (CNN)**, specifically the powerful **DenseNet-201** model, acts as the encoder. It processes the raw pixels of your image and "encodes" its most important visual features—like objects, colors, and textures—into a compact numerical vector. Think of this as the AI's internal summary of what it sees.

    #     #### 2. The Decoder: Describing the Scene
    #     A **Long Short-Term Memory (LSTM)** network acts as the decoder. This type of network excels at understanding sequences, making it perfect for language. It takes the feature vector from the encoder and, word by word, generates the most probable sequence of text to describe those features. It pays attention to what it has already written to create a coherent and context-aware sentence.
    #     """)
    
    # st.markdown("---")

    # --- Load Models ---
    models = load_all_models_local()
    if not all(models):
        st.stop()
    
    # --- Image Upload and Processing ---
    with st.container(border=True):
        st.header("📤 Upload Your Image")
        uploaded_image = st.file_uploader(
            "Choose an image file...", 
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_image:
            img_bytes = uploaded_image.getvalue()
            pil_img = Image.open(BytesIO(img_bytes))
            
            col1, col2 = st.columns([0.6, 0.4])
            
            with col1:
                st.image(img_bytes, caption="Your Uploaded Image")
            
            with col2:
                st.metric("Width", f"{pil_img.width}px")
                st.metric("Height", f"{pil_img.height}px")
                st.metric("File Size", f"{len(img_bytes) / 1024:.2f} KB")

            if st.button("🚀 Generate Caption", type="primary", use_container_width=True):
                with st.spinner("🤖 Analyzing image and crafting the perfect caption..."):
                    caption = generate_caption(img_bytes, models)
                
                if caption:
                    st.session_state.caption = caption
                    st.session_state.image = img_bytes
                    st.session_state.filename = uploaded_image.name
                else:
                    st.error("Could not generate a caption. Please try another image.")

    # --- Display Results ---
    if 'caption' in st.session_state:
        st.markdown("---")
        st.header("📜 Generated Caption")
        st.subheader(st.session_state.caption)

        st.download_button(
            label="📥 Download Caption (.txt)",
            data=st.session_state.caption,
            file_name=f"caption_{os.path.splitext(st.session_state.filename)[0]}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Clean up session state after displaying
        del st.session_state.caption
        del st.session_state.image
        del st.session_state.filename

if __name__ == "__main__":
    main()