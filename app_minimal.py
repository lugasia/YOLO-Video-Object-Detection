#!/usr/bin/env python3
"""
Minimal YOLO Video Object Detection App for testing deployment
"""

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="YOLO Test",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YOLO Video Object Detection")
st.write("Testing deployment...")

# Test imports
try:
    import ultralytics
    st.success("✅ Ultralytics imported successfully!")
except ImportError as e:
    st.error(f"❌ Ultralytics import failed: {e}")

try:
    import cv2
    st.success("✅ OpenCV imported successfully!")
except ImportError as e:
    st.error(f"❌ OpenCV import failed: {e}")

try:
    import torch
    st.success(f"✅ PyTorch imported successfully! Version: {torch.__version__}")
except ImportError as e:
    st.error(f"❌ PyTorch import failed: {e}")

# Test YOLO model loading
if st.button("Test YOLO Model Loading"):
    try:
        from ultralytics import YOLO
        with st.spinner("Loading YOLO model..."):
            model = YOLO('yolov8n.pt')
        st.success("✅ YOLO model loaded successfully!")
    except Exception as e:
        st.error(f"❌ YOLO model loading failed: {e}")

st.write("---")
st.write("If all tests pass, the full app should work!")
