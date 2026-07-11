import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Детекция переломов", layout="wide")
st.title("🦴 Детекция переломов на рентгеновских снимках")

# Выбор модели
model_choice = st.radio(
    "Выберите модель:",
    ["⚡ Быстрая (YOLOv8n)", "🎯 Точная (YOLOv8x)"],
    horizontal=True
)

# Загружаем модель (пока заглушка — одна и та же)
@st.cache_resource
def load_model(path):
    return YOLO(path)

if "Быстрая" in model_choice:
    model = load_model("/content/runs/detect/fracture_fast/weights/best.pt")
    st.info("⚡ Быстрая модель — результат за секунды")
else:
    # Пока используем ту же модель, потом заменим
    model = load_model("/content/runs/detect/fracture_fast/weights/best.pt")
    st.warning("🎯 Точная модель — подождёт загрузку настоящих весов")

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
