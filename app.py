import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
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

CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128)
]

def draw_boxes(image, results):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]
                color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                
                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((x1, y1 - 20), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 20), text, fill="white", font=font)
    
    return image

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Original", use_container_width=True)

    if st.button("Detect Fractures", type="primary"):
        with st.spinner("Analyzing..."):
            results = model(np.array(image))
            result_image = draw_boxes(image, results)

            with col2:
                st.image(result_image, caption="Result", use_container_width=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"Found {len(boxes)} suspicious area(s)")
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    st.write(f"- {results[0].names[cls_id]}: {conf:.2f}")
            else:
                st.info("No fractures detected")
