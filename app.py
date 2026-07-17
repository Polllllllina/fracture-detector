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
        background: #0a0a0a;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .main-header {
        background: #111111;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 3px solid rgba(204,136,227,0.4);
        box-shadow: 0 4px 24px rgba(204,136,227,0.08);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #9999aa;
        font-size: 1.15rem;
        margin-top: 0.5rem;
    }
    .model-hint {
        color: #cc88e3;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #141414;
        padding: 1.5rem;
        border-radius: 14px;
        border: 3px solid rgba(204,136,227,0.4);
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 2px 12px rgba(204,136,227,0.08);
    }
    .stButton > button {
        background: #cc88e3;
        color: #0a0a0a;
        border: none;
        padding: 0.9rem 2.2rem;
        font-size: 1.05rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #d9a3ed;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(204,136,227,0.35);
    }
    .result-card {
        background: #141414;
        padding: 1.5rem;
        border-radius: 14px;
        border: 3px solid rgba(204,136,227,0.35);
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #555566;
        margin-top: 3rem;
        border-top: 3px solid rgba(204,136,227,0.2);
    }
    p, span, label, div {
        color: #cccccc !important;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
    }
    .stSelectbox > div > div {
        background-color: #141414 !important;
        border: 3px solid rgba(204,136,227,0.45) !important;
        border-radius: 10px !important;
    }
    .stSelectbox [data-baseweb="select"] [aria-selected="true"] {
        color: #cc88e3 !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }
    .stSelectbox [data-baseweb="select"] div {
        color: #cc88e3 !important;
        font-size: 1.15rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox div[role="option"] div {
        color: #cc88e3 !important;
        font-size: 1.1rem !important;
        background-color: #141414 !important;
    }
    [data-testid="stFileUploader"] {
        background: #141414;
        border: 3px dashed rgba(204,136,227,0.35);
        border-radius: 14px;
        padding: 2rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #cc88e3;
        background: #1a1a1a;
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
    st.markdown('<p class="model-hint">Select a model to use for detection</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "Select Model",
        ["Fast (YOLOv8n)", "Accurate (YOLOv8m)"],
        label_visibility="collapsed"
    )

model_descriptions = {
    "Fast (YOLOv8n)": {"speed": "Fast", "accuracy": "Standard", "color": "#cc88e3"},
    "Accurate (YOLOv8m)": {"speed": "Moderate", "accuracy": "High", "color": "#10b981"}
}

desc = model_descriptions[model_choice]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0; font-size: 0.9rem; letter-spacing: 1px;">SPEED</h3>
        <h2 style="color: #ffffff; margin: 0; font-size: 1.8rem;">{desc['speed']}</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0; font-size: 0.9rem; letter-spacing: 1px;">ACCURACY</h3>
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
    "#cc88e3", "#10b981", "#f59e0b",
    "#ef4444", "#60a5fa", "#ec4899",
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
                
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
                
                text = f"{label} {conf:.2f}"
                bbox = draw.textbbox((x1, y1 - 22), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1 + 3, y1 - 22), text, fill="#0a0a0a", font=font)
    
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
                <div style="background: #0d1a14; padding: 1.2rem; border-radius: 12px; 
                            margin: 1rem 0; border: 3px solid rgba(16,185,129,0.3);">
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
                <div style="background: #141414; padding: 1.5rem; border-radius: 12px; 
                            margin: 1rem 0; border: 3px solid rgba(204,136,227,0.25);">
                    <h3 style="color: #777788; margin: 0;">No Fractures Detected</h3>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Bone Fracture Detection System | Powered by YOLOv8</p>
    <p style="font-size: 0.85rem;">For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)
