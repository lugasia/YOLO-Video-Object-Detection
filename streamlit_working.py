#!/usr/bin/env python3
"""
Working Streamlit App for YOLO Video Detection
Simplified version that works with the virtual environment
"""

import streamlit as st
import os
import json
import time
from ultralytics import YOLO
import tempfile

# Page configuration
st.set_page_config(
    page_title="YOLO Video Detection",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_yolo_model():
    """Load YOLO model"""
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video_with_yolo(video_file, confidence=0.5):
    """Process video using YOLO"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Load model
        model = load_yolo_model()
        if model is None:
            return None, None
        
        # Process video
        start_time = time.time()
        results = model.predict(
            temp_path,
            conf=confidence,
            save=True,
            project="streamlit_results",
            name="detections"
        )
        processing_time = time.time() - start_time
        
        # Analyze results
        total_detections = 0
        class_counts = {}
        
        for result in results:
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            'total_detections': total_detections,
            'class_counts': class_counts,
            'processing_time': processing_time,
            'total_frames': len(results)
        }, results
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 YOLO Video Object Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Upload Video", "📊 Results", "ℹ️ Info"])
    
    with tab1:
        st.header("Upload Video for Detection")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process with YOLO"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / (1024*1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video with YOLO..."):
                    results, raw_results = process_video_with_yolo(uploaded_file, confidence)
                
                if results:
                    st.success("✅ Video processing completed!")
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.raw_results = raw_results
                    st.session_state.video_name = uploaded_file.name
                    
                    # Show quick summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Detections", results['total_detections'])
                    with col2:
                        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                    with col3:
                        fps = results['total_frames'] / results['processing_time']
                        st.metric("FPS", f"{fps:.1f}")
    
    with tab2:
        st.header("Detection Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Detection summary
            st.subheader("📊 Detection Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Statistics</h4>
                    <p><strong>Total Detections:</strong> {results['total_detections']:,}</p>
                    <p><strong>Processing Time:</strong> {results['processing_time']:.2f} seconds</p>
                    <p><strong>Total Frames:</strong> {results['total_frames']:,}</p>
                    <p><strong>FPS:</strong> {results['total_frames'] / results['processing_time']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Objects Detected")
                if results['class_counts']:
                    for class_name, count in results['class_counts'].items():
                        percentage = (count / results['total_detections']) * 100
                        st.write(f"**{class_name}**: {count:,} ({percentage:.1f}%)")
                else:
                    st.info("No objects detected")
            
            # Show output file location
            st.subheader("📁 Output Files")
            output_path = "streamlit_results/detections"
            if os.path.exists(output_path):
                files = os.listdir(output_path)
                for file in files:
                    if file.endswith('.mp4'):
                        st.success(f"✅ Processed video saved: {output_path}/{file}")
                        st.info("You can download this file from your file system")
            else:
                st.warning("Output files not found")
        
        else:
            st.info("Please upload and process a video first to see results.")
    
    with tab3:
        st.header("System Information")
        
        st.markdown("""
        ### 🎯 About This System
        
        This is a **YOLO Video Object Detection System** that can:
        - Detect objects in video files
        - Identify multiple object classes
        - Provide real-time processing
        - Generate annotated videos
        
        ### 🛠️ Technical Details
        
        - **Model**: YOLOv8 Nano
        - **Framework**: Ultralytics
        - **Environment**: Virtual environment with compatible packages
        - **Performance**: Real-time capable
        
        ### 📋 Supported Object Classes
        
        YOLOv8 can detect 80+ object classes including:
        - Vehicles (car, truck, bus, motorcycle)
        - People
        - Animals
        - Common objects
        
        ### 🚀 How to Use
        
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Adjust confidence threshold if needed
        3. Click "Process Video"
        4. View results and download processed video
        """)

if __name__ == "__main__":
    main() 