import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import tempfile
from PIL import Image
import json
from yolo_video_detector import YOLOVideoDetector

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

def process_video_stream(video_file, confidence_threshold=0.5):
    """Process uploaded video file"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        temp_video_path = tmp_file.name
    
    try:
        # Initialize detector
        detector = YOLOVideoDetector(confidence_threshold=confidence_threshold)
        
        # Process video
        output_path = "temp_output.mp4"
        detector.process_video_file(temp_video_path, output_path)
        
        # Get results
        summary = detector.get_detection_summary()
        
        # Clean up temporary input file
        os.unlink(temp_video_path)
        
        return detector, summary, output_path
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        return None, None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 YOLO Video Object Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
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
        help="Minimum confidence score for detections"
    )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📹 Video Upload", "📊 Analysis", "📱 Real-time"])
    
    with tab1:
        st.header("Upload Video for Detection")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process with YOLO object detection"
        )
        
        if uploaded_file is not None:
            # Display video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            
            st.write("**File Details:**")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video with YOLO..."):
                    detector, summary, output_path = process_video_stream(
                        uploaded_file, confidence_threshold
                    )
                
                if detector and summary:
                    st.success("Video processing completed!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Frames", summary['total_frames'])
                    
                    with col2:
                        st.metric("Total Detections", summary['total_detections'])
                    
                    with col3:
                        st.metric("Processing Time", f"{summary['processing_time']:.2f}s")
                    
                    # Display processed video
                    if os.path.exists(output_path):
                        st.video(output_path)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Processed Video",
                                data=file.read(),
                                file_name="processed_video.mp4",
                                mime="video/mp4"
                            )
                    
                    # Store results in session state for analysis tab
                    st.session_state.detector = detector
                    st.session_state.summary = summary
    
    with tab2:
        st.header("Detection Analysis")
        
        if 'detector' in st.session_state and 'summary' in st.session_state:
            detector = st.session_state.detector
            summary = st.session_state.summary
            
            # Detection statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Detection Summary")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Statistics</h4>
                    <p><strong>Total Frames:</strong> {summary['total_frames']}</p>
                    <p><strong>Total Detections:</strong> {summary['total_detections']}</p>
                    <p><strong>Processing Time:</strong> {summary['processing_time']:.2f} seconds</p>
                    <p><strong>FPS:</strong> {summary['total_frames'] / summary['processing_time']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Object Classes Detected")
                if summary['class_counts']:
                    # Create simple bar chart using st.bar_chart
                    chart_data = summary['class_counts']
                    st.bar_chart(chart_data)
                else:
                    st.info("No detections found in the video.")
            
            # Detection details
            st.subheader("Detection Details")
            if detector.detection_history:
                # Show sample detections
                st.write("**Sample Detections (First 10 frames):**")
                for i, frame_data in enumerate(detector.detection_history[:10]):
                    if frame_data['detections']:
                        st.write(f"Frame {frame_data['frame']}:")
                        for detection in frame_data['detections']:
                            st.write(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
                    else:
                        st.write(f"Frame {frame_data['frame']}: No detections")
                
                # Detection frequency table
                st.subheader("Detection Frequency")
                detection_freq = {}
                for frame_data in detector.detection_history:
                    for detection in frame_data['detections']:
                        class_name = detection['class_name']
                        if class_name not in detection_freq:
                            detection_freq[class_name] = {'count': 0, 'confidences': []}
                        detection_freq[class_name]['count'] += 1
                        detection_freq[class_name]['confidences'].append(detection['confidence'])
                
                # Display as table
                if detection_freq:
                    freq_data = []
                    for class_name, data in detection_freq.items():
                        avg_confidence = sum(data['confidences']) / len(data['confidences'])
                        freq_data.append({
                            'Object Class': class_name,
                            'Count': data['count'],
                            'Avg Confidence': f"{avg_confidence:.3f}"
                        })
                    
                    st.table(freq_data)
        
        else:
            st.info("Please process a video first to see analysis results.")
    
    with tab3:
        st.header("Real-time Detection")
        
        st.info("""
        **Real-time Detection Features:**
        - Live webcam feed with YOLO detection
        - Real-time object tracking
        - Instant detection results
        """)
        
        # Webcam option
        if st.button("📹 Start Webcam Detection", type="primary"):
            st.warning("Real-time webcam detection requires camera access and will open in a separate window.")
            st.info("Press 'q' to quit the webcam detection window.")
            
            # Note: In a real implementation, you would integrate webcam feed here
            # For now, we'll show a placeholder
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>🔴 Webcam Detection</h3>
                <p>Camera feed would be displayed here with real-time YOLO detections.</p>
                <p>Press 'q' to quit detection.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 