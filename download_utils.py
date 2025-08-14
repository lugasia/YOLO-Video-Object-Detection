#!/usr/bin/env python3
"""
Download Utilities for Object Detection Results
Provides functions to download processed videos and results
"""

import os
import json
import streamlit as st
from typing import Dict, Optional, List
import base64

def get_video_download_button(video_path: str, button_label: str = "📥 Download Video", 
                            file_name: Optional[str] = None, help_text: str = "Download processed video") -> bool:
    """
    Create a download button for a video file
    
    Args:
        video_path: Path to the video file
        button_label: Label for the download button
        file_name: Custom filename for download (optional)
        help_text: Help text for the button
        
    Returns:
        True if button was created successfully, False otherwise
    """
    if not os.path.exists(video_path):
        st.warning(f"Video file not found: {video_path}")
        return False
    
    try:
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
        
        # Get file size
        file_size_mb = len(video_bytes) / (1024 * 1024)
        
        # Create download button
        st.download_button(
            label=button_label,
            data=video_bytes,
            file_name=file_name or os.path.basename(video_path),
            mime="video/mp4",
            help=help_text
        )
        
        # Show file info
        st.info(f"📊 File size: {file_size_mb:.2f} MB")
        return True
        
    except Exception as e:
        st.error(f"Error creating download button: {e}")
        return False

def get_json_download_button(data: Dict, button_label: str = "📄 Download JSON", 
                           file_name: str = "detection_results.json", 
                           help_text: str = "Download detection results") -> bool:
    """
    Create a download button for JSON data
    
    Args:
        data: Dictionary data to download
        button_label: Label for the download button
        file_name: Filename for download
        help_text: Help text for the button
        
    Returns:
        True if button was created successfully, False otherwise
    """
    try:
        results_json = json.dumps(data, indent=2)
        
        st.download_button(
            label=button_label,
            data=results_json,
            file_name=file_name,
            mime="application/json",
            help=help_text
        )
        return True
        
    except Exception as e:
        st.error(f"Error creating JSON download button: {e}")
        return False

def find_processed_videos(output_dir: str) -> List[str]:
    """
    Find processed video files in output directory
    
    Args:
        output_dir: Directory to search for videos
        
    Returns:
        List of video file paths
    """
    video_files = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(output_dir, file))
    
    return video_files

def create_download_section(results: Dict, video_name: str, output_dir: str, 
                          section_title: str = "📥 Download Results") -> None:
    """
    Create a comprehensive download section with video and JSON downloads
    
    Args:
        results: Detection results dictionary
        video_name: Original video filename
        output_dir: Directory containing processed videos
        section_title: Title for the download section
    """
    st.subheader(section_title)
    
    # Find processed videos
    video_files = find_processed_videos(output_dir)
    
    if video_files:
        # Download video
        video_path = video_files[0]  # Use first video found
        video_filename = f"detected_{video_name}"
        
        col1, col2 = st.columns([2, 1])
        with col1:
            get_video_download_button(
                video_path=video_path,
                button_label="🎬 Download Detected Video",
                file_name=video_filename,
                help_text="Download the processed video with object detection annotations"
            )
        with col2:
            # Show file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                st.metric("File Size", f"{file_size:.1f} MB")
    else:
        st.info("No processed video found. Video processing may still be in progress.")
    
    # Download JSON results
    json_filename = f"detection_results_{video_name.replace('.', '_')}.json"
    get_json_download_button(
        data=results,
        button_label="📄 Download JSON Results",
        file_name=json_filename,
        help_text="Download detailed detection results in JSON format"
    )

def create_construction_download_section(results: Dict, video_name: str, output_dir: str) -> None:
    """
    Create a specialized download section for construction equipment detection
    
    Args:
        results: Detection results dictionary
        video_name: Original video filename
        output_dir: Directory containing processed videos
    """
    st.subheader("📥 Download Construction Detection Results")
    
    # Find processed videos
    video_files = find_processed_videos(output_dir)
    
    if video_files:
        # Download video
        video_path = video_files[0]
        video_filename = f"construction_detection_{video_name}"
        
        col1, col2 = st.columns([2, 1])
        with col1:
            get_video_download_button(
                video_path=video_path,
                button_label="🚧 Download Construction Detection Video",
                file_name=video_filename,
                help_text="Download the processed video with construction equipment detection"
            )
        with col2:
            # Show file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                st.metric("File Size", f"{file_size:.1f} MB")
    else:
        st.info("No processed video found. Video processing may still be in progress.")
    
    # Download JSON results
    json_filename = f"construction_results_{video_name.replace('.', '_')}.json"
    get_json_download_button(
        data=results,
        button_label="📄 Download Construction Results",
        file_name=json_filename,
        help_text="Download detailed construction equipment detection results in JSON format"
    )

def create_enhanced_download_section(results: Dict, video_name: str, output_dir: str) -> None:
    """
    Create a specialized download section for enhanced detection results
    
    Args:
        results: Detection results dictionary
        video_name: Original video filename
        output_dir: Directory containing processed videos
    """
    st.subheader("📥 Download Enhanced Detection Results")
    
    # Find processed videos
    video_files = find_processed_videos(output_dir)
    
    if video_files:
        # Download video
        video_path = video_files[0]
        video_filename = f"enhanced_detection_{video_name}"
        
        col1, col2 = st.columns([2, 1])
        with col1:
            get_video_download_button(
                video_path=video_path,
                button_label="🚀 Download Enhanced Detection Video",
                file_name=video_filename,
                help_text="Download the processed video with enhanced object detection"
            )
        with col2:
            # Show file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                st.metric("File Size", f"{file_size:.1f} MB")
    else:
        st.info("No processed video found. Video processing may still be in progress.")
    
    # Download JSON results
    json_filename = f"enhanced_results_{video_name.replace('.', '_')}.json"
    get_json_download_button(
        data=results,
        button_label="📄 Download Enhanced Results",
        file_name=json_filename,
        help_text="Download detailed enhanced detection results in JSON format"
    )

def create_ensemble_download_section(results: Dict, video_name: str, output_dir: str) -> None:
    """
    Create a specialized download section for ensemble detection results
    
    Args:
        results: Detection results dictionary
        video_name: Original video filename
        output_dir: Directory containing processed videos
    """
    st.subheader("📥 Download Ensemble Detection Results")
    
    # Find processed videos
    video_files = find_processed_videos(output_dir)
    
    if video_files:
        # Download video
        video_path = video_files[0]
        video_filename = f"ensemble_detection_{video_name}"
        
        col1, col2 = st.columns([2, 1])
        with col1:
            get_video_download_button(
                video_path=video_path,
                button_label="🤖 Download Ensemble Detection Video",
                file_name=video_filename,
                help_text="Download the processed video with ensemble object detection"
            )
        with col2:
            # Show file size
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)
                st.metric("File Size", f"{file_size:.1f} MB")
    else:
        st.info("No processed video found. Video processing may still be in progress.")
    
    # Download JSON results
    json_filename = f"ensemble_results_{video_name.replace('.', '_')}.json"
    get_json_download_button(
        data=results,
        button_label="📄 Download Ensemble Results",
        file_name=json_filename,
        help_text="Download detailed ensemble detection results in JSON format"
    )

def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def format_file_size(file_size_mb: float) -> str:
    """
    Format file size for display
    
    Args:
        file_size_mb: File size in MB
        
    Returns:
        Formatted file size string
    """
    if file_size_mb < 1:
        return f"{file_size_mb * 1024:.1f} KB"
    elif file_size_mb < 1024:
        return f"{file_size_mb:.1f} MB"
    else:
        return f"{file_size_mb / 1024:.1f} GB"

def create_multiple_video_download_section(video_files: List[str], video_name: str, 
                                         section_title: str = "📥 Download Videos") -> None:
    """
    Create download section for multiple video files
    
    Args:
        video_files: List of video file paths
        video_name: Original video filename
        section_title: Title for the download section
    """
    st.subheader(section_title)
    
    if not video_files:
        st.info("No processed videos found.")
        return
    
    for i, video_path in enumerate(video_files):
        if os.path.exists(video_path):
            # Create unique filename
            base_name = os.path.splitext(video_name)[0]
            ext = os.path.splitext(video_path)[1]
            download_filename = f"{base_name}_detection_{i+1}{ext}"
            
            # Get file size
            file_size_mb = get_file_size_mb(video_path)
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                get_video_download_button(
                    video_path=video_path,
                    button_label=f"📥 Download Video {i+1}",
                    file_name=download_filename,
                    help_text=f"Download processed video {i+1}"
                )
            with col2:
                st.metric("Size", format_file_size(file_size_mb))
            with col3:
                st.write(f"`{os.path.basename(video_path)}`")
        else:
            st.warning(f"Video file not found: {video_path}")

# Example usage functions for different detection types
def create_detection_download_section(detection_type: str, results: Dict, video_name: str, 
                                    output_dir: str) -> None:
    """
    Create download section based on detection type
    
    Args:
        detection_type: Type of detection ('basic', 'construction', 'enhanced', 'ensemble')
        results: Detection results dictionary
        video_name: Original video filename
        output_dir: Directory containing processed videos
    """
    if detection_type == 'construction':
        create_construction_download_section(results, video_name, output_dir)
    elif detection_type == 'enhanced':
        create_enhanced_download_section(results, video_name, output_dir)
    elif detection_type == 'ensemble':
        create_ensemble_download_section(results, video_name, output_dir)
    else:
        create_download_section(results, video_name, output_dir)
