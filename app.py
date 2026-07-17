import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import gdown

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1d23 0%, #232834 50%, #1e2129 100%);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .main-header {
        background: linear-gradient(135deg, #2d3340, #363d4a);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #a8b2c1;
        font-size: 1.15rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: #2d3340;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f8cff, #3d6fd9);
        color: white;
        border: none;
        padding: 0.9rem 2.2rem;
        font-size: 1.05rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79,140,255,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79,140,255,0.4);
        background: linear-gradient(135deg, #5d99ff, #4f8cff);
    }
    .result-card {
        background: #2d3340;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7385;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    p, span, label, div {
        color: #d1d7e2 !important;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    .stSelectbox > div > div {
        background-color: #2d3340 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] {
        background: #2d3340;
        border: 2px dashed rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 2rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #4f8cff;
        background: #333a49;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🦴 Bone Fracture Detection</h1>
    <p>AI-powered X-ray analysis for fracture detection</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    model_choice = st.selectbox(
        "Select Model",
        ["Fast (YOLOv8n)", "Accurate (YOLOv8m)"],
        label_visibility="collapsed"
    )

model_descriptions = {
    "Fast (YOLOv8n)": {"speed": "Fast", "accuracy": "Standard", "color": "#4f8cff"},
    "Accurate (YOLOv8m)": {"speed": "Moderate", "accuracy": "High", "color": "#10b981"}
}

desc = model_descriptions[model_choice]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0; font-size: 1rem;">SPEED</h3>
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem;">{desc['speed']}</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0; font-size: 1rem;">ACCURACY</h3>
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem;">{desc['accuracy']}</h2>
    </div>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    if model_name == "Fast (YOLOv8n)":
        return YOLO("best_fast.pt")
    else:
        url = "https://drive.google.com/uc?id=1M59f3dXN6A8lBFok5IjHxTBS6y55KVyt"
        output = "best_accurate.pt"
        if not os.path.exists(output):
            with st.spinner("Downloading model..."):
                gdown.download(url, output, quiet=False)
        return YOLO(output)

model = load_model(model_choice)

CLASS_COLORS = [
    "#4f8cff", "#10b981", "#f59e0b",
    "#ef4444", "#8b5cf6", "#ec4899",
    "#06b6d4"
]

def draw_boxes(image, results):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
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
                
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
                
                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((x1, y1 - 22), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1 + 3, y1 - 22), text, fill="white", font=font)
    
    return image

uploaded_file = st.file_uploader(
    "Upload an X-ray image for analysis",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)

    if st.button("Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Processing image..."):
            results = model(np.array(image))
            result_image = draw_boxes(image, results)

            with col2:
                st.markdown("### Detection Results")
                st.image(result_image, use_container_width=True)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.markdown(f"""
                <div style="background: #1e3a2f; padding: 1.2rem; border-radius: 12px; 
                            margin: 1rem 0; border: 1px solid rgba(16,185,129,0.3);">
                    <h3 style="color: #10b981; margin: 0; font-size: 1.2rem;">Found {len(boxes)} Potential Fracture(s)</h3>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(len(boxes))
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    
                    with cols[i]:
                        st.markdown(f"""
                        <div class="result-card" style="border-top: 3px solid {color};">
                            <p style="color: {color}; font-weight: 600; margin: 0 0 0.5rem 0;">{results[0].names[cls_id]}</p>
                            <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem;">{conf:.1%}</h2>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #2d3340; padding: 1.5rem; border-radius: 12px; 
                            margin: 1rem 0; border: 1px solid rgba(255,255,255,0.06);">
                    <h3 style="color: #a8b2c1; margin: 0;">No Fractures Detected</h3>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Bone Fracture Detection System | Powered by YOLOv8</p>
    <p style="font-size: 0.85rem;">For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)
