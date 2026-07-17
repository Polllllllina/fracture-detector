import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import gdown

st.set_page_config(page_title="Bone Fracture Detection", layout="wide")
st.title("Bone Fracture Detection on X-ray Images")

model_choice = st.radio(
    "Select model:",
    ["Baseline (YOLOv8n)", "Fast (YOLOv8n tuned)", "Accurate (YOLOv8m tuned)"],
    horizontal=True
)

@st.cache_resource
def load_model(model_name):
    if model_name == "Baseline (YOLOv8n)":
        return YOLO("best.pt")
    elif model_name == "Fast (YOLOv8n tuned)":
        return YOLO("best_fast.pt")
    else:
        url = "https://drive.google.com/uc?id=1M59f3dXN6A8lBFok5IjHxTBS6y55KVyt"
        output = "best_accurate.pt"
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        return YOLO(output)

model = load_model(model_choice)

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    if st.button("Detect Fractures", type="primary"):
        with st.spinner("Analyzing..."):
            results = model(np.array(image))
            for r in results:
                plotted = r.plot()
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            with col2:
                st.image(plotted_rgb, caption="Result", use_container_width=True)

            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"Found {len(boxes)} suspicious area(s)")
            else:
                st.info("No fractures detected")
