#!/usr/bin/env python3
"""
YOLO Video Object Detection App - Clean Version
Fixed for Streamlit Cloud compatibility
"""

import streamlit as st
import os
import json
import time
import tempfile
import base64

# Page configuration - Updated for latest deployment
st.set_page_config(
    page_title="Inteli AI - YOLO Video Detection",
    page_icon="🎥",
    layout="wide"
)

# Check required packages
def check_dependencies():
    """Check if required packages are available"""
    try:
        import ultralytics
        return True
    except ImportError:
        st.error("❌ Ultralytics not found. Please check requirements.txt")
        st.info("💡 Make sure ultralytics is listed in requirements.txt")
        return False

# Check dependencies are available
if not check_dependencies():
    st.stop()

# Custom CSS with logo styling - removed black background
st.markdown("""
<style>
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: transparent;
        border-radius: 10px;
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
    .model-status {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
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
    .yolo11-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def display_logo():
    """Display the Inteli AI logo with transparent background"""
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

@st.cache_resource
def load_yolo_model(model_name="yolov8n.pt"):
    """Load YOLO model with automatic download - fixed for libGL.so.1 error"""
    try:
        # Set environment variables to avoid OpenCV GUI issues
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
        
        # Import ultralytics
        from ultralytics import YOLO
        
        # Load the model directly - YOLO will download if needed
        with st.spinner(f"🔄 Loading {model_name}..."):
            model = YOLO(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        # Try with a smaller model if the selected one fails
        if model_name != "yolov8n.pt":
            st.info("Trying with yolov8n.pt instead...")
            return load_yolo_model("yolov8n.pt")
        return None

def check_model_status():
    """Check and display model availability status"""
    try:
        # Check if any YOLO models are available in the default location
        from pathlib import Path
        
        # Check default YOLO cache directory
        cache_dir = Path.home() / ".cache" / "ultralytics"
        models_dir = Path("models")
        
        available_models = []
        all_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        
        for model_name in all_models:
            # Check multiple possible locations
            possible_paths = [
                models_dir / model_name,
                cache_dir / model_name,
                Path(model_name)
            ]
            
            for path in possible_paths:
                if path.exists():
                    size = path.stat().st_size / (1024 * 1024)  # Size in MB
                    version_badge = "YOLOv8"
                    available_models.append(f"{model_name} ({size:.1f}MB) [{version_badge}]")
                    break
        
        if available_models:
            st.markdown(f"""
            <div class="model-status">
                <h4>✅ Models Available</h4>
                <p>Ready to use: {', '.join(available_models)}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Models will be downloaded automatically when you first use them.")
    except Exception as e:
        st.info("ℹ️ Models will be downloaded automatically when you first use them.")

def process_video_with_yolo(video_file, confidence=0.5, model_name="yolov8n.pt"):
    """Process video with real YOLO detection"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Load model (will download automatically if needed)
        model = load_yolo_model(model_name)
        if model is None:
            st.error("❌ Failed to load YOLO model")
            return None, None
        
        # Process video
        start_time = time.time()
        
        with st.spinner(f"🎯 Processing video with {model_name}..."):
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
            'model_used': model_name,
            'model_version': 'YOLOv8'
        }, results
        
    except Exception as e:
        st.error(f"❌ Error processing video: {e}")
        return None, None

def main():
    # Display logo and header
    display_logo()
    
    # Main title
    st.markdown('<h1 class="main-header">🎥 YOLO Video Object Detection System</h1>', unsafe_allow_html=True)
    
    # Check model status
    check_model_status()
    
    # Important notice
    st.markdown("""
    <div class="construction-alert">
        <h4>⚠️ Important Notice</h4>
        <p>YOLO may classify construction equipment (excavators, bulldozers, etc.) as similar objects like "train" or "truck". 
        This is normal behavior for general-purpose object detection models.</p>
        <p><strong>💡 Tip:</strong> Look for "train" detections - these might actually be excavators!</p>
        <p><strong>🚀 Latest:</strong> YOLOv8 models provide excellent accuracy for all use cases!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Model selection - YOLOv8 models only (YOLOv11 not yet available)
    model_options = [
        ("yolov8n.pt", "YOLOv8 Nano (6.2MB) - Fastest"),
        ("yolov8s.pt", "YOLOv8 Small (21.5MB) - Balanced"),
        ("yolov8m.pt", "YOLOv8 Medium (52.2MB) - Better Accuracy"),
        ("yolov8l.pt", "YOLOv8 Large (87.7MB) - High Accuracy"),
        ("yolov8x.pt", "YOLOv8 XLarge (136.2MB) - Best Accuracy")
    ]
    
    model_option = st.sidebar.selectbox(
        "Select YOLO Model",
        options=[opt[0] for opt in model_options],
        format_func=lambda x: next(opt[1] for opt in model_options if opt[0] == x),
        help="YOLOv8 models provide excellent accuracy. Models download automatically."
    )
    
    # Show model info
    if "x" in model_option:
        st.sidebar.markdown('<span class="yolo11-badge">🚀 Best Accuracy</span>', unsafe_allow_html=True)
    
    # Confidence threshold
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Lower threshold catches more objects (including possible excavators)"
    )
    
    # Model download info
    st.sidebar.info("ℹ️ Models download automatically when first used")
    
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
                with st.spinner(f"Processing video with {model_option}..."):
                    results, raw_results = process_video_with_yolo(
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
                        version_badge = "YOLOv8"
                        st.metric("Model", f"{results['model_used'].split('.')[0]} [{version_badge}]")
    
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
                            for i, detection in enumerate(detections[:5]):  # Show first 5
                                st.write(f"Frame {detection['frame']}: {detection['confidence']:.2f}")
            else:
                st.info("No construction equipment detected. Try lowering the confidence threshold.")
        else:
            st.info("Upload and process a video first to see construction analysis.")
    
    with tab3:
        st.header("📊 Complete Detection Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            class_counts = results['class_counts']
            
            if class_counts:
                st.subheader("📈 Detection Statistics")
                
                # Display class counts
                st.write("**Object Classes Detected:**")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"• {class_name}: {count} detections")
                
                # Show processing info
                st.subheader("⚙️ Processing Information")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Detections", results['total_detections'])
                with col2:
                    st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                with col3:
                    st.metric("Model Used", results['model_used'])
                with col4:
                    version_badge = "YOLOv8"
                    st.metric("Version", version_badge)
                
                # Show results location
                st.info("📁 Processed video saved in: `inteli_ai_results/detections/`")
            else:
                st.info("No objects detected. Try adjusting the confidence threshold.")
        else:
            st.info("Upload and process a video first to see complete results.")
    
    with tab4:
        st.header("ℹ️ About Inteli AI")
        
        st.markdown("""
        ### 🎯 Advanced Computer Vision Solutions
        
        **Inteli AI** provides cutting-edge AI solutions for computer vision applications.
        
        ### 🚀 Features
        - **Real-time Object Detection**: Using state-of-the-art YOLOv8 models
        - **Construction Equipment Recognition**: Specialized detection for construction sites
        - **Automatic Model Management**: Models download and cache automatically
        - **Professional Interface**: Clean, modern UI with Inteli AI branding
        
        ### 🛠️ Technology
        - **YOLOv8**: Latest object detection models
        - **Streamlit**: Modern web application framework
        - **OpenCV**: Computer vision processing
        - **Ultralytics**: YOLO framework
        
        ### 🆕 YOLOv8 Features
        - **Improved Accuracy**: Better detection performance
        - **Faster Processing**: Optimized inference speed
        - **Enhanced Training**: Better model training capabilities
        - **Latest Architecture**: State-of-the-art design
        
        ### 📞 Contact
        - **Website**: [Inteli AI](https://inteliate.com)
        - **Email**: support@inteli-ai.com
        
        ---
        
        **Made with ❤️ by Inteli AI**
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2024 Inteli AI - Advanced Computer Vision Solutions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
