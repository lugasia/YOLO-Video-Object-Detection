#!/usr/bin/env python3
"""
Video Object Detection App - Complete Version
With logo, real YOLO detection, and video download functionality
"""

import streamlit as st
import numpy as np
import time
import os
import tempfile
from PIL import Image
import json
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import shutil

# Page configuration
st.set_page_config(
    page_title="InteliATE - Video Object Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with logo
st.markdown("""
<style>
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .logo-image {
        width: 200px;
        height: auto;
        object-fit: contain;
    }
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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

def display_logo():
    """Display the InteliATE logo"""
    try:
        logo_path = "inteli-ai-black.webp"
        if os.path.exists(logo_path):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div class="logo-container">
                    <img src="data:image/webp;base64,{}" class="logo-image">
                </div>
                """.format(get_image_base64(logo_path)), unsafe_allow_html=True)
        else:
            st.warning("Logo file not found: inteli-ai-black.webp")
    except Exception as e:
        st.warning(f"Error loading logo: {e}")

def get_available_models():
    """Get list of available YOLO models"""
    models = []
    model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            # Get file size for display
            file_size = os.path.getsize(model_file) / (1024*1024)  # MB
            models.append(f"{model_file} ({file_size:.1f}MB)")
    
    return models

def get_model_filename(model_option):
    """Extract the actual model filename from the display option"""
    if "(" in model_option:
        return model_option.split(" (")[0]
    return model_option

def create_mock_detection_results(video_name, confidence_threshold=0.5):
    """Create mock detection results for demonstration when YOLO is not available"""
    import random
    
    # Mock classes that might be detected
    possible_classes = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat']
    
    # Generate mock results
    total_frames = random.randint(30, 120)  # 1-4 seconds at 30fps
    total_detections = 0
    class_counts = {}
    frame_analysis = {}
    
    for frame_id in range(total_frames):
        # Random number of detections per frame
        num_detections = random.randint(0, 3) if random.random() > 0.3 else 0
        
        frame_detections = []
        for _ in range(num_detections):
            class_name = random.choice(possible_classes)
            confidence = random.uniform(confidence_threshold, 0.95)
            
            # Only count detections above threshold
            if confidence >= confidence_threshold:
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections += 1
                
                frame_detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [random.uniform(0, 1) for _ in range(4)]  # Mock bbox
                })
        
        if frame_detections:
            frame_analysis[frame_id] = frame_detections
    
    processing_time = random.uniform(2.0, 8.0)
    
    return {
        'total_detections': total_detections,
        'total_frames': total_frames,
        'processing_time': processing_time,
        'fps': total_frames / processing_time,
        'class_counts': class_counts,
        'frame_analysis': frame_analysis,
        'model_used': 'mock_detector',
        'video_name': video_name
    }

def process_video_with_yolo(video_path, confidence_threshold=0.5, model_name="yolov8n.pt"):
    """Process video with real YOLO detection"""
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(model_name)
        
        # Process video
        start_time = time.time()
        results = model.predict(
            video_path,
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
            'model_used': model_name,
            'video_name': os.path.basename(video_path)
        }
        
        return summary, results, "streamlit_results/detections"
        
    except ImportError:
        st.warning("⚠️ Ultralytics not available. Using mock detection.")
        return None, None, None
    except Exception as e:
        st.error(f"Error with YOLO detection: {e}")
        return None, None, None

def process_video_stream(video_file, confidence_threshold=0.5, model_name="yolov8n.pt"):
    """Process uploaded video file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name
    
    try:
        # Try real YOLO detection first
        summary, results, output_dir = process_video_with_yolo(temp_video_path, confidence_threshold, model_name)
        
        if summary is None:
            # Fall back to mock detection
            st.info("ℹ️ Using mock detection for demonstration purposes.")
            summary = create_mock_detection_results(video_file.name, confidence_threshold)
            output_dir = "mock_output"
        
        # Clean up temporary input file
        os.unlink(temp_video_path)
        
        return summary, results, output_dir
        
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

def create_detection_timeline(frame_analysis):
    """Create a timeline of detections across frames"""
    if not frame_analysis:
        return None
    
    # Prepare data
    frames = []
    detection_counts = []
    
    for frame_id in sorted(frame_analysis.keys()):
        frames.append(frame_id)
        detection_counts.append(len(frame_analysis[frame_id]))
    
    if not frames:
        return None
    
    # Create line chart
    fig = px.line(
        x=frames,
        y=detection_counts,
        title="Detections Over Time",
        labels={'x': 'Frame Number', 'y': 'Number of Detections'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def get_video_download_button(output_dir, video_name):
    """Create download button for processed video"""
    try:
        if os.path.exists(output_dir):
            # Look for the processed video file
            for file in os.listdir(output_dir):
                if file.endswith('.mp4'):
                    video_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(video_path) / (1024*1024)  # MB
                    
                    with open(video_path, "rb") as f:
                        video_data = f.read()
                    
                    st.download_button(
                        label=f"📹 Download Processed Video ({file_size:.1f} MB)",
                        data=video_data,
                        file_name=f"detected_{video_name}",
                        mime="video/mp4"
                    )
                    return True
        
        return False
    except Exception as e:
        st.error(f"Error creating video download: {e}")
        return False

def main():
    # Display logo
    display_logo()
    
    # Header
    st.markdown('<h1 class="main-header">🎥 Video Object Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Check available models
    available_models = get_available_models()
    
    if available_models:
        # Model selection
        model_option = st.sidebar.selectbox(
            "Select YOLO Model",
            available_models,
            help="Choose the YOLO model. Larger models are more accurate but slower."
        )
        st.sidebar.success(f"✅ {len(available_models)} YOLO model(s) available")
        
        # Show model info
        model_filename = get_model_filename(model_option)
        st.sidebar.info(f"Selected: {model_filename}")
    else:
        model_option = "mock"
        st.sidebar.warning("⚠️ No YOLO models found. Using mock detection.")
        st.sidebar.info("💡 Models may be available locally but not on Streamlit Cloud")
    
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
                with st.spinner("Processing video..."):
                    # Get actual model filename for processing
                    model_filename = get_model_filename(model_option)
                    summary, results, output_dir = process_video_stream(
                        uploaded_file, confidence_threshold, model_filename
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
                        st.metric("Model Used", summary['model_used'])
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
                
                # Timeline chart
                timeline_chart = create_detection_timeline(summary['frame_analysis'])
                if timeline_chart:
                    st.plotly_chart(timeline_chart, use_container_width=True)
            
            # Download section
            st.subheader("📥 Download Results")
            
            # Create JSON results for download
            json_data = json.dumps(summary, indent=2)
            st.download_button(
                label="📄 Download JSON Results",
                data=json_data,
                file_name=f"detection_results_{summary['video_name']}.json",
                mime="application/json"
            )
            
            # Video download
            if 'output_dir' in st.session_state:
                video_downloaded = get_video_download_button(
                    st.session_state.output_dir, 
                    st.session_state.video_name
                )
                if not video_downloaded:
                    st.info("📹 Processed video will be available for download when using real YOLO detection")
        
        else:
            st.info("Please upload and process a video first to see results.")
    
    with tab3:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### 🎯 InteliATE Video Object Detection Demo System
        
        This system provides advanced object detection for video files using state-of-the-art YOLO models.
        
        ### 🚀 Features
        
        - **Real YOLO Detection**: Uses actual YOLO models when available
        - **Multiple Model Support**: Choose from different YOLO model sizes
        - **Adjustable Confidence**: Fine-tune detection sensitivity
        - **Real-time Processing**: Fast video processing with progress tracking
        - **Comprehensive Analysis**: Detailed detection statistics and visualizations
        - **Export Capabilities**: Download processed videos and detection results
        
        ### 📋 Supported Object Classes
        
        The system can detect 80+ object classes including:
        - Vehicles (cars, trucks, buses, motorcycles)
        - People and animals
        - Common objects and items
        - Construction equipment
        
        ### 🛠️ Technical Details
        
        - **Framework**: Ultralytics YOLO (when available)
        - **Models**: YOLOv8 (nano, small, medium, large, xlarge)
        - **Processing**: GPU-accelerated when available
        - **Output**: Annotated videos with bounding boxes
        
        ### 🚀 How to Use
        
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Select appropriate YOLO model
        3. Adjust confidence threshold
        4. Click "Detect Objects"
        5. View results and download processed video
        
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
    st.markdown("""
    <div class="footer">
        <p><strong>InteliATE Video Object Detection Demo System</strong></p>
        <p>Powered by YOLO and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
