#!/usr/bin/env python3
"""
Improved Streamlit App for Excavator Detection
Handles construction equipment classification better
"""

import streamlit as st
import os
import json
import time
from ultralytics import YOLO
import tempfile

# Page configuration
st.set_page_config(
    page_title="Excavator Detection",
    page_icon="🚧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b35;
        margin: 0.5rem 0;
    }
    .construction-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_construction_mapping():
    """Get mapping for construction equipment"""
    return {
        'train': '🚂 Train (Possible Excavator/Construction Equipment)',
        'truck': '🚛 Truck (Possible Dump Truck/Construction Vehicle)',
        'car': '🚗 Car/Vehicle',
        'bus': '🚌 Bus',
        'boat': '🚢 Boat',
        'airplane': '✈️ Airplane',
        'person': '👤 Person',
        'motorcycle': '🏍️ Motorcycle',
        'bicycle': '🚲 Bicycle'
    }

def load_yolo_model(model_name="yolov8n.pt"):
    """Load YOLO model"""
    try:
        model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_video_for_excavators(video_file, confidence=0.3, model_name="yolov8n.pt"):
    """Process video with focus on construction equipment"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Load model
        model = load_yolo_model(model_name)
        if model is None:
            return None, None
        
        # Process video
        start_time = time.time()
        results = model.predict(
            temp_path,
            conf=confidence,
            save=True,
            project="excavator_streamlit_results",
            name="detections"
        )
        processing_time = time.time() - start_time
        
        # Analyze results
        total_detections = 0
        class_counts = {}
        construction_detections = {}
        construction_mapping = get_construction_mapping()
        
        for i, result in enumerate(results):
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    confidence_score = float(box.conf[0].cpu().numpy())
                    
                    # Count all detections
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Focus on construction-related detections
                    if class_name in construction_mapping:
                        if class_name not in construction_detections:
                            construction_detections[class_name] = []
                        construction_detections[class_name].append({
                            'frame': i,
                            'confidence': confidence_score,
                            'description': construction_mapping[class_name]
                        })
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            'total_detections': total_detections,
            'class_counts': class_counts,
            'construction_detections': construction_detections,
            'processing_time': processing_time,
            'total_frames': len(results),
            'model_used': model_name
        }, results
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚧 Excavator & Construction Equipment Detection</h1>', unsafe_allow_html=True)
    
    # Important notice
    st.markdown("""
    <div class="construction-alert">
        <h4>⚠️ Important Notice</h4>
        <p>YOLOv8 may classify construction equipment (excavators, bulldozers, etc.) as similar objects like "train" or "truck". 
        This is normal behavior for general-purpose object detection models.</p>
        <p><strong>💡 Tip:</strong> Look for "train" detections - these might actually be excavators!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Larger models are more accurate but slower"
    )
    
    # Confidence threshold
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Lower threshold catches more objects (including possible excavators)"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Upload Video", "🚧 Construction Analysis", "📊 All Results", "ℹ️ Info"])
    
    with tab1:
        st.header("Upload Video for Construction Equipment Detection")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to detect construction equipment"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size = uploaded_file.size / (1024*1024)
            st.info(f"📁 File: {uploaded_file.name} ({file_size:.2f} MB)")
            
            # Process button
            if st.button("🚀 Detect Construction Equipment", type="primary"):
                with st.spinner("Processing video for construction equipment..."):
                    results, raw_results = process_video_for_excavators(
                        uploaded_file, confidence, model_option
                    )
                
                if results:
                    st.success("✅ Video processing completed!")
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.raw_results = raw_results
                    st.session_state.video_name = uploaded_file.name
                    
                    # Show quick summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Detections", results['total_detections'])
                    with col2:
                        st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                    with col3:
                        fps = results['total_frames'] / results['processing_time']
                        st.metric("FPS", f"{fps:.1f}")
                    with col4:
                        st.metric("Model Used", model_option.split('.')[0])
    
    with tab2:
        st.header("🚧 Construction Equipment Analysis")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            construction_detections = results['construction_detections']
            
            if construction_detections:
                st.subheader("🔍 Construction Equipment Detected")
                
                for class_name, detections in construction_detections.items():
                    construction_mapping = get_construction_mapping()
                    description = construction_mapping.get(class_name, class_name)
                    
                    with st.expander(f"{description} ({len(detections)} detections)"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Total Detections:** {len(detections)}")
                            
                            # High confidence detections
                            high_conf = [d for d in detections if d['confidence'] > 0.7]
                            st.write(f"**High Confidence (>70%):** {len(high_conf)}")
                            
                            # Frame range
                            frames = [d['frame'] for d in detections]
                            if frames:
                                st.write(f"**Frame Range:** {min(frames)} - {max(frames)}")
                        
                        with col2:
                            # Show sample detections
                            st.write("**Sample Detections:**")
                            for det in detections[:5]:
                                st.write(f"Frame {det['frame']}: {det['confidence']:.3f}")
                        
                        # Special note for train detections
                        if class_name == 'train':
                            st.warning("""
                            🚂 **Note:** These 'train' detections might actually be **excavators** or other construction equipment!
                            YOLOv8 often misclassifies large construction machinery as trains due to similar visual characteristics.
                            """)
            else:
                st.info("No construction equipment detected. Try lowering the confidence threshold.")
        else:
            st.info("Please upload and process a video first to see construction analysis.")
    
    with tab3:
        st.header("📊 All Detection Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Detection summary
            st.subheader("📈 Detection Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Statistics</h4>
                    <p><strong>Total Detections:</strong> {results['total_detections']:,}</p>
                    <p><strong>Processing Time:</strong> {results['processing_time']:.2f} seconds</p>
                    <p><strong>Total Frames:</strong> {results['total_frames']:,}</p>
                    <p><strong>FPS:</strong> {results['total_frames'] / results['processing_time']:.1f}</p>
                    <p><strong>Model Used:</strong> {results['model_used']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("All Objects Detected")
                if results['class_counts']:
                    for class_name, count in sorted(results['class_counts'].items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / results['total_detections']) * 100
                        st.write(f"**{class_name}**: {count:,} ({percentage:.1f}%)")
                else:
                    st.info("No objects detected")
            
            # Show output file location
            st.subheader("📁 Output Files")
            output_path = "excavator_streamlit_results/detections"
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
    
    with tab4:
        st.header("ℹ️ System Information")
        
        st.markdown("""
        ### 🚧 About This System
        
        This is a **Construction Equipment Detection System** that can:
        - Detect construction equipment in video files
        - Identify vehicles that might be excavators, bulldozers, etc.
        - Provide detailed analysis of construction-related objects
        - Handle misclassification issues common with general-purpose models
        
        ### 🛠️ Technical Details
        
        - **Model**: YOLOv8 (multiple sizes available)
        - **Framework**: Ultralytics
        - **Environment**: Virtual environment with compatible packages
        - **Performance**: Real-time capable
        
        ### 🚂 Important: Train vs Excavator
        
        YOLOv8 often classifies construction equipment as "train" because:
        - Both are large, heavy machinery
        - Similar visual characteristics
        - Limited training data for specific construction equipment
        - General-purpose model limitations
        
        ### 📋 Construction Equipment Mapping
        
        The system maps detections to possible construction equipment:
        - **🚂 Train** → Possible Excavator/Bulldozer/Construction Equipment
        - **🚛 Truck** → Possible Dump Truck/Construction Vehicle
        - **🚗 Car** → Construction Vehicle/Equipment Transport
        - **🚌 Bus** → Construction Crew Transport
        
        ### 🚀 How to Use
        
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Select appropriate model size
        3. Adjust confidence threshold (lower = more detections)
        4. Click "Detect Construction Equipment"
        5. Check the Construction Analysis tab for results
        
        ### 💡 Tips for Better Results
        
        - Use lower confidence thresholds (0.2-0.4) to catch more objects
        - Try larger models (yolov8m, yolov8l, yolov8x) for better accuracy
        - Look for "train" detections - these might be excavators!
        - Check high-confidence detections for better accuracy
        """)

if __name__ == "__main__":
    main() 