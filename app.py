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
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.8rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .main-header p {
        color: #a0a0b0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #16213e, #1a1a2e);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #e94560;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c23152);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(233,69,96,0.5);
    }
    .result-card {
        background: #16213e;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #0f3460;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #a0a0b0;
        margin-top: 3rem;
        border-top: 1px solid #0f3460;
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
    "Accurate (YOLOv8m)": {"speed": "Moderate", "accuracy": "High", "color": "#533483"}
}

desc = model_descriptions[model_choice]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']};">Speed</h3>
        <h2 style="color: white;">{desc['speed']}</h2>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']};">Accuracy</h3>
        <h2 style="color: white;">{desc['accuracy']}</h2>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="color: {desc['color']};">Model</h3>
        <h2 style="color: white;">{model_choice.split()[0]}</h2>
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
    (233, 69, 96), (0, 212, 255), (83, 52, 131),
    (255, 193, 7), (0, 200, 83), (255, 61, 0),
    (156, 39, 176)
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
                <div style="background: linear-gradient(135deg, #1b4332, #2d6a4f); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                            box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
                    <h3 style="color: #95d5b2; margin: 0;">Found {len(boxes)} Potential Fracture(s)</h3>
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
                            <h2 style="color: white;">{conf:.1%}</h2>
                            <p style="color: #a0a0b0;">confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); 
                            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
                            box-shadow: 0 4px 16px rgba(0,0,0,0.2);">
                    <h3 style="color: #a0a0b0; margin: 0;">No Fractures Detected</h3>
                </div>
                """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p>Bone Fracture Detection System | Powered by YOLOv8</p>
    <p style="font-size: 0.9rem;">For research and educational purposes only</p>
</div>
""", unsafe_allow_html=True)
