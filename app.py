import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import os

# =============================
# CONFIGURATION
# =============================
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, processor = load_model()

# =============================
# LOAD PRECOMPUTED PRODUCT EMBEDDINGS
# =============================
@st.cache_resource
def load_index(emb_path="product_embs.npy", csv_path="valid_products.csv"):
    try:
        embs = np.load(emb_path)
        df_valid = pd.read_csv(csv_path)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        st.sidebar.success(f"✅ Loaded {len(df_valid)} products with embeddings")
        return index, embs, df_valid
    except FileNotFoundError:
        st.error("❌ Precomputed embeddings or valid_products.csv not found. "
                 "Please generate them locally first.")
        st.stop()

index, product_embs, df_valid = load_index()

# =============================
# STREAMLIT UI
# =============================
st.title("🛍️ Visual Product Matcher")
st.markdown(
    """
    Upload a product image, and the app will find **visually similar products**
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
                    f"**{row['name']}**<br>{row['category']}<br>💠 Score: `{scores[i]:.3f}`",
                    unsafe_allow_html=True,
                )
else:
    st.info("👆 Upload an image to start matching.")
