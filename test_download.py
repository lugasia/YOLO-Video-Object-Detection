#!/usr/bin/env python3
"""
Test script to demonstrate download functionality
"""

import streamlit as st
import os
import json
from download_utils import (
    get_video_download_button,
    get_json_download_button,
    create_download_section,
    create_construction_download_section,
    create_enhanced_download_section,
    create_ensemble_download_section,
    find_processed_videos
)

def main():
    st.title("📥 Download Functionality Test")
    st.write("This page demonstrates the download functionality for processed videos and results.")
    
    # Test data
    test_results = {
        "total_detections": 1500,
        "processing_time": 25.5,
        "total_frames": 573,
        "class_counts": {
            "truck": 800,
            "car": 400,
            "person": 200,
            "bus": 100
        },
        "model_used": "yolov8n.pt"
    }
    
    # Test different output directories
    test_dirs = [
        "inteli_ai_results/detections",
        "excavator_streamlit_results/detections", 
        "enhanced_simple_results/detections",
        "ensemble_results/detections",
        "output_detected_video.mp4"  # Single file
    ]
    
    st.header("🔍 Available Output Directories")
    
    for test_dir in test_dirs:
        with st.expander(f"📁 {test_dir}"):
            if os.path.exists(test_dir):
                if os.path.isdir(test_dir):
                    files = os.listdir(test_dir)
                    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                    
                    if video_files:
                        st.success(f"✅ Found {len(video_files)} video file(s)")
                        for video_file in video_files:
                            st.write(f"  - {video_file}")
                        
                        # Test download button
                        video_path = os.path.join(test_dir, video_files[0])
                        if get_video_download_button(
                            video_path=video_path,
                            button_label=f"📥 Download {video_files[0]}",
                            file_name=f"test_{video_files[0]}",
                            help_text=f"Download {video_files[0]}"
                        ):
                            st.success("✅ Download button created successfully")
                    else:
                        st.info("No video files found in this directory")
                else:
                    # Single file
                    if get_video_download_button(
                        video_path=test_dir,
                        button_label=f"📥 Download {os.path.basename(test_dir)}",
                        file_name=f"test_{os.path.basename(test_dir)}",
                        help_text=f"Download {os.path.basename(test_dir)}"
                    ):
                        st.success("✅ Download button created successfully")
            else:
                st.warning("❌ Directory/file not found")
    
    st.header("📄 JSON Download Test")
    
    # Test JSON download
    if get_json_download_button(
        data=test_results,
        button_label="📄 Download Test Results",
        file_name="test_detection_results.json",
        help_text="Download test detection results"
    ):
        st.success("✅ JSON download button created successfully")
    
    st.header("🎯 Specialized Download Sections")
    
    # Test different download sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚧 Construction Detection")
        create_construction_download_section(
            results=test_results,
            video_name="test_video.mp4",
            output_dir="inteli_ai_results/detections"
        )
    
    with col2:
        st.subheader("🚀 Enhanced Detection")
        create_enhanced_download_section(
            results=test_results,
            video_name="test_video.mp4",
            output_dir="enhanced_simple_results/detections"
        )
    
    st.header("📊 File Information")
    
    # Show file sizes for existing files
    st.subheader("📁 File Sizes")
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            if os.path.isdir(test_dir):
                files = os.listdir(test_dir)
                video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                for video_file in video_files:
                    video_path = os.path.join(test_dir, video_file)
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    st.write(f"📹 {video_file}: {file_size_mb:.2f} MB")
            else:
                # Single file
                file_size_mb = os.path.getsize(test_dir) / (1024 * 1024)
                st.write(f"📹 {os.path.basename(test_dir)}: {file_size_mb:.2f} MB")
    
    st.header("💡 Usage Instructions")
    
    st.markdown("""
    ### How to Use Download Functions
    
    1. **Basic Video Download**:
    ```python
    from download_utils import get_video_download_button
    
    get_video_download_button(
        video_path="path/to/video.mp4",
        button_label="📥 Download Video",
        file_name="my_video.mp4"
    )
    ```
    
    2. **JSON Results Download**:
    ```python
    from download_utils import get_json_download_button
    
    get_json_download_button(
        data=results_dict,
        button_label="📄 Download Results",
        file_name="results.json"
    )
    ```
    
    3. **Complete Download Section**:
    ```python
    from download_utils import create_download_section
    
    create_download_section(
        results=results_dict,
        video_name="original_video.mp4",
        output_dir="results/detections"
    )
    ```
    
    4. **Specialized Download Sections**:
    ```python
    # For construction detection
    create_construction_download_section(results, video_name, output_dir)
    
    # For enhanced detection
    create_enhanced_download_section(results, video_name, output_dir)
    
    # For ensemble detection
    create_ensemble_download_section(results, video_name, output_dir)
    ```
    """)

if __name__ == "__main__":
    main()
