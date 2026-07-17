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
        background: linear-gradient(135deg, #0a0a1a 0%, #12122a 50%, #0d0d24 100%);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a3e 0%, #1e1e4a 50%, #162052 100%);
        padding: 2.5rem;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(233,69,96,0.3);
        box-shadow: 0 8px 32px rgba(233,69,96,0.1);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892b0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #12123a);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(233,69,96,0.2);
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-size: 1.1rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3);
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(233,69,96,0.5);
        background: linear-gradient(135deg, #ff5a75, #e94560);
    }
    .result-card {
        background: linear-gradient(135deg, #1a1a3e, #12123a);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(100,255,218,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #8892b0;
        margin-top: 3rem;
        border-top: 1px solid rgba(233,69,96,0.2);
    }
    p, span, label, div {
        color: #ccd6f6 !important;
    }
    h1, h2, h3, h4 {
        color: #e6f1ff !important;
    }
    .stSelectbox > div > div {
        background-color: #1a1a3e !important;
        border: 1px solid rgba(233,69,96,0.3) !important;
        border-radius: 12px !important;
        color: white !important;
    }
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1a1a3e, #12123a);
        border: 2px dashed rgba(233,69,96,0.4);
        border-radius: 16px;
        padding: 2rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #e94560;
        background: linear-gradient(135deg, #1e1e4a, #1a1a3e);
    }
    .stSuccess {
        background-color: rgba(0,200,83,0.1) !important;
        border: 1px solid rgba(0,200,83,0.3) !important;
    }
    .stInfo {
        background-color: rgba(100,255,218,0.1) !important;
        border: 1px solid rgba(100,255,218,0.3) !important;
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
    "Fast (YOLOv8n)": {"speed": "Fast", "accuracy": "Standard", "color": "#e94560"},
    "Accurate (YOLOv8m)": {"speed": "Moderate", "accuracy": "High", "color": "#64ffda"}
}

desc = model_descriptions[model_choice]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0;">Speed</h3>
        <h2 style="color: #e6f1ff; margin: 0;">{desc['speed']}</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']}; margin: 0 0 0.5rem 0;">Accuracy</h3>
        <h2 style="color: #e6f1ff; margin: 0;">{desc['accuracy']}</h2>
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
    (233, 69, 96), (100, 255, 218), (255, 193, 7),
    (0, 200, 83), (100, 149, 237), (255, 105, 180),
    (191, 64, 191)
]

def draw_boxes(image, results):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
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
                bbox = draw.textbbox((x1, y1 - 25), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1 + 2, y1 - 25), text, fill="white", font=font)
    
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
                <div style="background: linear-gradient(135deg, #0a2e1a, #0d3d24); 
                            padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
                            border: 1px solid rgba(0,200,83,0.3);
                            box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                    <h3 style="color: #64ffda; margin: 0;">Found {len(boxes)} Potential Fracture(s)</h3>
                </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns(len(boxes))
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    color_hex = '#%02x%02x%02x' % color
                    
                    with cols[i]:
                        st.markdown(f"""
                        <div class="result-card" style="border-left: 4px solid {color_hex};">
                            <h4 style="color: {color_hex};">{results[0].names[cls_id]}</h4>
                            <h2 style="color: #e6f1ff;">{conf:.1%}</h2>
                            <p style="color: #8892b0;">confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1a1a3e, #12123a); 
                            padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
                            border: 1px solid rgba(100,255,218,0.1);
                            box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                    <h3 style="color: #8892b0; margin: 0;">No Fractures Detected</h3>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Bone Fracture Detection System | Powered by YOLOv8</p>
    <p style="font-size: 0.9rem;">For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)
