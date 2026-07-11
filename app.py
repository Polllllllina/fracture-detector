import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

st.set_page_config(page_title="Детекция переломов", layout="wide")
st.title("🦴 Детекция переломов на рентгеновских снимках")

# Выбор модели
model_choice = st.radio(
    "Выберите модель:",
    ["⚡ Быстрая (YOLOv8n)", "🎯 Точная (YOLOv8x)"],
    horizontal=True
)

# Загружаем модель из текущей папки
@st.cache_resource
def load_model(path):
    return YOLO(path)

# Путь к файлу модели в той же папке, что и app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

if "Быстрая" in model_choice:
    model = load_model(MODEL_PATH)
    st.info("⚡ Быстрая модель — результат за секунды")
else:
    model = load_model(MODEL_PATH)
    st.warning("🎯 Точная модель — скоро заменим на настоящую")

# Загрузка изображения
uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Оригинал", use_container_width=True)
    
    if st.button("🔍 Найти переломы", type="primary"):
        with st.spinner("Анализирую снимок..."):
            results = model(np.array(image))
            
            for r in results:
                plotted = r.plot()
                plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(plotted_rgb, caption="Результат", use_container_width=True)
            
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"✅ Найдено подозрительных областей: {len(boxes)}")
            else:
                st.info("ℹ️ Переломов не обнаружено")

st.caption("MVP версия | Данные: Bone Fracture Detection (Roboflow)")
