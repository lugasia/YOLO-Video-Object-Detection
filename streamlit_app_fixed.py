#!/usr/bin/env python3
"""
YOLO Video Object Detection App - Fixed Version
No OpenCV dependency to avoid NumPy compatibility issues on Streamlit Cloud
"""

import streamlit as st
import numpy as np
from ultralytics import YOLO
import time
import os
import tempfile
from PIL import Image
import json
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="YOLO Video Object Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-box {
        border: 2px solid #1f77b4;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_yolo_model(model_path="yolov8n.pt"):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video_stream(video_file, confidence_threshold=0.5, model_name="yolov8n.pt"):
    """Process uploaded video file using ultralytics directly"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name
    
    try:
        # Load model
        model = load_yolo_model(model_name)
        if model is None:
            return None, None, None
        
        # Process video with ultralytics
        start_time = time.time()
        results = model.predict(
            temp_video_path,
            conf=confidence_threshold,
            save=True,
            project="streamlit_results",
            name="detections"
        )
        processing_time = time.time() - start_time
        
        # Analyze results
        total_detections = 0
        class_counts = {}
        frame_analysis = {}
        
        for i, result in enumerate(results):
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                frame_detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    confidence_score = float(box.conf[0].cpu().numpy())
                    
                    # Count all detections
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Store frame-level detection
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence_score,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
                
                frame_analysis[i] = frame_detections
        
        # Create summary
        summary = {
            'total_detections': total_detections,
            'total_frames': len(results),
            'processing_time': processing_time,
            'fps': len(results) / processing_time,
            'class_counts': class_counts,
            'frame_analysis': frame_analysis,
            'model_used': model_name
        }
        
        # Clean up temporary input file
        os.unlink(temp_video_path)
        
        return summary, results, "streamlit_results/detections"
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        return None, None, None

def create_detection_chart(class_counts):
    """Create a bar chart of detections"""
    if not class_counts:
        return None
    
    # Prepare data for plotting
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Create bar chart
    fig = px.bar(
        x=classes,
        y=counts,
        title="Object Detection Results",
        labels={'x': 'Object Class', 'y': 'Number of Detections'},
        color=counts,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400
    )
    
    return fig

def create_confidence_distribution(frame_analysis):
    """Create confidence distribution chart"""
    if not frame_analysis:
        return None
    
    # Collect all confidence scores
    confidences = []
    for frame_data in frame_analysis.values():
        for det in frame_data:
            confidences.append(det['confidence'])
    
    if not confidences:
        return None
    
    # Create histogram
    fig = px.histogram(
        x=confidences,
        title="Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Detections'},
        nbins=20
    )
    
    fig.update_layout(height=400)
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 YOLO Video Object Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Choose the YOLO model size. Larger models are more accurate but slower."
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower threshold catches more objects, higher threshold is more selective"
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Upload Video", "📊 Results Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Video for Object Detection")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to detect objects"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / (1024*1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Process button
            if st.button("🚀 Detect Objects", type="primary"):
                with st.spinner("Processing video with YOLO..."):
                    summary, results, output_dir = process_video_stream(
                        uploaded_file, confidence_threshold, model_option
                    )
                
                if summary:
                    st.success("✅ Video processing completed!")
                    
                    # Store results in session state
                    st.session_state.summary = summary
                    st.session_state.results = results
                    st.session_state.output_dir = output_dir
                    st.session_state.video_name = uploaded_file.name
                    
                    # Show quick summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Detections", summary['total_detections'])
                    with col2:
                        st.metric("Processing Time", f"{summary['processing_time']:.2f}s")
                    with col3:
                        st.metric("FPS", f"{summary['fps']:.1f}")
                    with col4:
                        st.metric("Model Used", model_option.split('.')[0])
                else:
                    st.error("❌ Processing failed. Please check the video file and try again.")
    
    with tab2:
        st.header("📊 Detection Results Analysis")
        
        if 'summary' in st.session_state:
            summary = st.session_state.summary
            
            # Detection summary
            st.subheader("📈 Detection Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Statistics</h4>
                    <p><strong>Total Detections:</strong> {summary['total_detections']:,}</p>
                    <p><strong>Processing Time:</strong> {summary['processing_time']:.2f} seconds</p>
                    <p><strong>Total Frames:</strong> {summary['total_frames']:,}</p>
                    <p><strong>FPS:</strong> {summary['fps']:.1f}</p>
                    <p><strong>Model Used:</strong> {summary['model_used']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Object Classes Detected")
                if summary['class_counts']:
                    for class_name, count in sorted(summary['class_counts'].items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / summary['total_detections']) * 100
                        st.write(f"**{class_name}**: {count:,} ({percentage:.1f}%)")
                else:
                    st.info("No objects detected")
            
            # Charts
            st.subheader("📊 Visualizations")
            
            # Detection chart
            if summary['class_counts']:
                chart = create_detection_chart(summary['class_counts'])
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Confidence distribution
            if 'frame_analysis' in summary:
                conf_chart = create_confidence_distribution(summary['frame_analysis'])
                if conf_chart:
                    st.plotly_chart(conf_chart, use_container_width=True)
            
            # Output files
            st.subheader("📁 Output Files")
            if 'output_dir' in st.session_state and os.path.exists(st.session_state.output_dir):
                files = os.listdir(st.session_state.output_dir)
                for file in files:
                    if file.endswith('.mp4'):
                        st.success(f"✅ Processed video saved: {st.session_state.output_dir}/{file}")
                        st.info("You can download this file from your file system")
            else:
                st.warning("Output files not found")
        
        else:
            st.info("Please upload and process a video first to see results.")
    
    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 YOLO Video Object Detection System
        
        This system provides real-time object detection for video files using state-of-the-art YOLO (You Only Look Once) models.
        
        ### 🚀 Features
        
        - **Multiple Model Options**: Choose from different YOLO model sizes
        - **Adjustable Confidence**: Fine-tune detection sensitivity
        - **Real-time Processing**: Fast video processing with progress tracking
        - **Comprehensive Analysis**: Detailed detection statistics and visualizations
        - **Export Capabilities**: Save processed videos and detection results
        
        ### 📋 Supported Object Classes
        
        The system can detect 80+ object classes including:
        - Vehicles (cars, trucks, buses, motorcycles)
        - People and animals
        - Common objects and items
        - Construction equipment
        
        ### 🛠️ Technical Details
        
        - **Framework**: Ultralytics YOLO
        - **Models**: YOLOv8 (nano, small, medium, large, xlarge)
        - **Processing**: GPU-accelerated when available
        - **Output**: Annotated videos with bounding boxes
        
        ### 🚀 How to Use
        
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Select appropriate model size
        3. Adjust confidence threshold
        4. Click "Detect Objects"
        5. View results in the analysis tab
        
        ### 💡 Tips for Best Results
        
        - Use lower confidence thresholds (0.2-0.4) to catch more objects
        - Try larger models (yolov8m, yolov8l, yolov8x) for better accuracy
        - Check the processed video to verify detections
        - Use appropriate model size for your hardware capabilities
        
        ### 🔧 Troubleshooting
        
        - **No detections**: Try lowering the confidence threshold
        - **Slow processing**: Use smaller models or shorter videos
        - **Memory issues**: Use smaller models or process shorter segments
        - **File format issues**: Convert to MP4 format if needed
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <p><strong>YOLO Video Object Detection System</strong> | Powered by Ultralytics</p>
        <p>For more information, visit the <a href="https://github.com/ultralytics/ultralytics" target="_blank">Ultralytics repository</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
