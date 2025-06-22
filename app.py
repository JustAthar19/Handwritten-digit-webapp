import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from generate_digit_images import generate_images

st.set_page_config(page_title="Handwritten Digit Image Generator", layout="centered")

st.title("🖋️ Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    images = generate_images(digit)

    st.subheader(f"Generated images of digit {digit}")
    cols = st.columns(5)

    for idx, col in enumerate(cols):
        with col:
            st.image(images[idx].squeeze(), width=80, caption=f"Sample {idx + 1}")
