import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import faiss
import requests
import os
from requests.exceptions import ReadTimeout, ConnectionError

# =============================
# CONFIGURATION
# =============================
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Increase Hugging Face download timeout
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "90"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        st.info("⏳ Loading CLIP model... (this may take a minute on first run)")
        model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        st.success("✅ Model loaded successfully!")
        return model, processor
    except (ReadTimeout, ConnectionError):
        st.error("❌ Network timeout while loading model. Please check your internet connection and retry.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, processor = load_model()

# =============================
# LOAD PRODUCT DATA
# =============================
@st.cache_data
def load_products(csv_path="products.txt"):
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["image_url"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {csv_path}. Please ensure it exists in your app directory.")
        st.stop()

df_products = load_products()
st.sidebar.success(f"✅ Loaded {len(df_products)} products")

# =============================
# DOWNLOAD IMAGES AND BUILD INDEX
# =============================
@st.cache_resource
def build_index(df):
    st.info("🧠 Building image embeddings... please wait.")
    images = []
    valid_rows = []
    for _, row in df.iterrows():
        try:
            response = requests.get(row["image_url"], timeout=20)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)
            valid_rows.append(row)
        except Exception as e:
            print("Error loading image:", row["image_url"], e)
            continue

    if not images:
        st.error("❌ No images could be loaded. Check URLs in products.csv.")
        st.stop()

    # Compute embeddings
    model.eval()
    all_embs = []
    with torch.no_grad():
        for img in images:
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
    embs = np.vstack(all_embs).astype("float32")

    # Build FAISS index
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    valid_df = pd.DataFrame(valid_rows)
    st.success("✅ Image index built successfully!")
    return index, embs, valid_df

index, product_embs, df_valid = build_index(df_products)

# =============================
# STREAMLIT UI
# =============================
st.title("🛍 Visual Product Matcher")
st.markdown(
    """
    Upload a product image, and the app will find *visually similar products*
    from the dataset using CLIP embeddings and FAISS similarity search.
    """
)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Uploaded Image", width=250)

    with st.spinner("🔍 Finding similar products..."):
        # Compute embedding for query image
        with torch.no_grad():
            inputs = processor(images=query_image, return_tensors="pt").to(DEVICE)
            q_emb = model.get_image_features(**inputs)
            q_emb = q_emb / q_emb.norm(p=2, dim=-1, keepdim=True)
            q_emb = q_emb.cpu().numpy().astype("float32")

        # Search top-k
        top_k = st.slider("Number of results", 3, 15, 6)
        D, I = index.search(q_emb, top_k)
        scores, indices = D[0], I[0]

        # Display results in grid layout
        st.subheader("🔎 Top Matches")
        cols = st.columns(3)
        for i, idx in enumerate(indices):
            row = df_valid.iloc[idx]
            with cols[i % 3]:
                st.image(row["image_url"], width=200)
                st.markdown(
                    f"{row['name']}<br>{row['category']}<br>💠 Score: {scores[i]:.3f}",
                    unsafe_allow_html=True,
                )
else:
    st.info("👆 Upload an image to start matching.")