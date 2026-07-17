# 🦴 Bone Fracture Detection

AI-powered X-ray analysis system for automated bone fracture detection using YOLOv8.

## Live Demo

**[Open Application on Streamlit Cloud](https://fracture-detector-y44nt4vmpj6xreewxtzn4h.streamlit.app/)**

---

## Project Overview

This project implements a web application for detecting bone fractures in X-ray images. Users can upload an X-ray and choose between two YOLOv8 models for inference.

The system supports 7 fracture classes:

| Class |
|-------|
| Elbow Positive |
| Fingers Positive |
| Forearm Fracture |
| Humerus Fracture |
| Humerus |
| Shoulder Fracture |
| Wrist Positive |

---

## Grading Criteria

| # | Criteria | Max Score | Status |
|---|----------|-----------|--------|
| 1a | Model tuning: mAP@0.5 < 0.6 | 1 |✅ |
| 1b | Model tuning: mAP@0.5 >= 0.6 | 4 |  |
| 2 | Backend implementation | 3 | ✅ FastAPI + Streamlit |
| 3 | Frontend implementation | 2 | ✅ Streamlit UI |
| 4 | Model selection (2+ models) | 4 | ✅ Fast + Accurate |
| 5 | GitHub repository | 1 | ✅ |
| 6 | Video presentation | 2 | ✅ |
| 7 | Streamlit Cloud deployment | 4 | ✅ |
| **Total** | | **16** | |

---

##  Models

| Model | Architecture | mAP@0.5 | Speed | Description |
|-------|-------------|---------|-------|-------------|
| Fast | YOLOv8n | 0.XXX | ~5ms | Lightweight, suitable for CPU |
| Accurate | YOLOv8m | 0.XXX | ~15ms | Higher accuracy, GPU recommended |

*Metrics obtained on the Bone Fracture Detection dataset.*

---

##  Repository Structure
fracture-detector/
├── app.py # Streamlit web application
├── train.py # Model training script
├── best_fast.pt # Fast model weights
├── best_accurate.pt # Accurate model weights (Google Drive)
├── requirements.txt # Python dependencies
├── packages.txt # System dependencies
└── README.md # Project documentation
---

## Installation

```bash
git clone https://github.com/Polllllllina/fracture-detector.git
cd fracture-detector
pip install -r requirements.txt
streamlit run app.py
