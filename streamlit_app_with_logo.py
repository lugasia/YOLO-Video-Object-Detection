#!/usr/bin/env python3
"""
YOLO Video Object Detection App with Inteli AI Logo
Enhanced version with company branding
"""

import streamlit as st
import os
import json
import time
from ultralytics import YOLO
import tempfile
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Inteli AI - YOLO Video Detection",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS with logo styling
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
        margin: 0.5rem 0;
    }
    .construction-alert {
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

def display_logo():
    """Display the Inteli AI logo with original black background"""
    try:
        # Load and display logo
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
            st.error("Logo file not found: inteli-ai-black.webp")
    except Exception as e:
        st.error(f"Error loading logo: {e}")

def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    import base64
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

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

def process_video_for_excavators(video_file, confidence=0.5, model_name="yolov8n.pt"):
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
            project="inteli_ai_results",
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
    # Display logo and header
    display_logo()
    
    # Main title
    st.markdown('<h1 class="main-header">🎥 YOLO Video Object Detection System</h1>', unsafe_allow_html=True)
    
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
    st.sidebar.title("⚙️ Settings")
    
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
        value=0.5,
        step=0.05,
        help="Lower threshold catches more objects (including possible excavators)"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Upload Video", "🚧 Construction Analysis", "📊 All Results", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Video for Object Detection")
        
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
                with st.spinner("Processing video with Inteli AI..."):
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
                
                # Download button for construction analysis
                st.subheader("📥 Download Processed Video")
                output_path = "inteli_ai_results/detections"
                if os.path.exists(output_path):
                    files = [f for f in os.listdir(output_path) if f.endswith('.mp4')]
                    if files:
                        video_file_path = os.path.join(output_path, files[0])
                        if os.path.exists(video_file_path):
                            with open(video_file_path, "rb") as video_file:
                                video_bytes = video_file.read()
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.download_button(
                                    label="🎬 Download Detected Video",
                                    data=video_bytes,
                                    file_name=f"construction_detection_{st.session_state.get('video_name', 'video.mp4')}",
                                    mime="video/mp4",
                                    help="Download the processed video with construction equipment detection"
                                )
                            with col2:
                                file_size_mb = len(video_bytes) / (1024 * 1024)
                                st.metric("File Size", f"{file_size_mb:.1f} MB")
                        else:
                            st.warning("Video file not found")
                    else:
                        st.warning("No processed video found")
                else:
                    st.warning("Output directory not found")
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
            
            # Show output file location and download button
            st.subheader("📁 Output Files")
            output_path = "inteli_ai_results/detections"
            if os.path.exists(output_path):
                files = os.listdir(output_path)
                for file in files:
                    if file.endswith('.mp4'):
                        st.success(f"✅ Processed video saved: {output_path}/{file}")
                        
                        # Download button for the processed video
                        video_file_path = os.path.join(output_path, file)
                        if os.path.exists(video_file_path):
                            with open(video_file_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                
                            # Create download button
                            st.download_button(
                                label="📥 Download Detected Video",
                                data=video_bytes,
                                file_name=f"detected_{st.session_state.get('video_name', 'video.mp4')}",
                                mime="video/mp4",
                                help="Download the processed video with object detection annotations"
                            )
                            
                            # Show file info
                            file_size_mb = len(video_bytes) / (1024 * 1024)
                            st.info(f"📊 File size: {file_size_mb:.2f} MB")
                        else:
                            st.warning("Video file not found")
                
                # Download JSON results
                st.subheader("📊 Download Detection Results")
                results_json = json.dumps(results, indent=2)
                st.download_button(
                    label="📄 Download JSON Results",
                    data=results_json,
                    file_name=f"detection_results_{st.session_state.get('video_name', 'video').replace('.', '_')}.json",
                    mime="application/json",
                    help="Download detailed detection results in JSON format"
                )
            else:
                st.warning("Output files not found")
        
        else:
            st.info("Please upload and process a video first to see results.")
    
    with tab4:
        st.header("ℹ️ About Inteli AI")
        
        st.markdown("""
        ### 🚀 About Inteli AI
        
        **Inteli AI** is a leading artificial intelligence company specializing in computer vision and object detection solutions.
        
        ### 🎯 Our YOLO Video Detection System
        
        This system provides:
        - **Real-time Object Detection**: Using state-of-the-art YOLOv8 models
        - **Construction Equipment Recognition**: Specialized detection for construction sites
        - **High Accuracy**: Optimized parameters for minimal false positives
        - **User-Friendly Interface**: Easy-to-use web application
        
        ### 🛠️ Technical Features
        
        - **Multiple Model Options**: From fast nano models to high-accuracy large models
        - **Adjustable Confidence**: Fine-tune detection sensitivity
        - **Comprehensive Analysis**: Detailed detection statistics and visualizations
        - **Export Capabilities**: Save processed videos and detection results
        
        ### 📋 Supported Object Classes
        
        The system can detect 80+ object classes including:
        - Construction equipment (trucks, excavators, bulldozers)
        - Vehicles (cars, buses, motorcycles)
        - People and animals
        - Common objects and items
        
        ### 🚀 How to Use
        
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Select appropriate model size
        3. Adjust confidence threshold (lower = more detections)
        4. Click "Detect Objects"
        5. View results in the analysis tabs
        
        ### 💡 Tips for Best Results
        
        - Use lower confidence thresholds (0.2-0.4) to catch more objects
        - Try larger models (yolov8m, yolov8l, yolov8x) for better accuracy
        - Look for "train" detections - these might be excavators!
        - Check high-confidence detections for better accuracy
        
        ### 📞 Contact
        
        For more information about Inteli AI solutions, please visit our website or contact our team.
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>Powered by Inteli AI</strong> | Advanced Computer Vision Solutions</p>
        <p>YOLO Video Object Detection System v2.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 