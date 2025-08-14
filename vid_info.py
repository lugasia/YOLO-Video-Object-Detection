#!/usr/bin/env python3
"""
Video Information Script for vid.mov
Analyzes the video file and provides information
"""

import os
import json
import time

def analyze_video_file():
    """Analyze the vid.mov file"""
    print("🎥 Video Analysis for vid.mov")
    print("=" * 50)
    
    video_file = "vid.mov"
    
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Get basic file information
    file_size = os.path.getsize(video_file)
    file_size_mb = file_size / (1024*1024)
    
    print(f"✅ Video file found: {video_file}")
    print(f"📁 File size: {file_size_mb:.2f} MB ({file_size:,} bytes)")
    
    # Get file modification time
    mod_time = os.path.getmtime(video_file)
    mod_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
    print(f"📅 Last modified: {mod_time_str}")
    
    # Check if we have previous results
    print(f"\n📊 Previous Processing Results:")
    if os.path.exists('detection_results.json'):
        print("✅ Previous detection results available")
        try:
            with open('detection_results.json', 'r') as f:
                data = json.load(f)
            summary = data['summary']
            print(f"   - Previous video: {summary['total_frames']:,} frames")
            print(f"   - Previous detections: {summary['total_detections']:,} objects")
        except:
            print("   - Previous results file exists but may be corrupted")
    else:
        print("❌ No previous detection results found")
    
    # Provide solutions
    print(f"\n🔧 Solutions for Processing vid.mov:")
    print("=" * 40)
    
    print("\n1️⃣ **Dependency Fix (Recommended):**")
    print("   Create a new virtual environment with compatible packages:")
    print("   ```bash")
    print("   python -m venv yolo_env")
    print("   source yolo_env/bin/activate  # On macOS/Linux")
    print("   pip install 'numpy<2.0.0'")
    print("   pip install ultralytics opencv-python")
    print("   python standalone_vid_processor.py")
    print("   ```")
    
    print("\n2️⃣ **Use Online YOLO Services:**")
    print("   - Upload vid.mov to online YOLO demo sites")
    print("   - Use Google Colab with YOLO")
    print("   - Use cloud-based AI services")
    
    print("\n3️⃣ **Alternative Processing:**")
    print("   - Use different video analysis tools")
    print("   - Convert video to images and process individually")
    print("   - Use pre-trained models in different frameworks")
    
    print("\n4️⃣ **Manual Analysis:**")
    print("   - Extract frames from video manually")
    print("   - Use image processing tools")
    print("   - Analyze video content manually")
    
    # Create a summary file
    video_info = {
        'filename': video_file,
        'size_bytes': file_size,
        'size_mb': file_size_mb,
        'modified_time': mod_time_str,
        'analysis_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'ready_for_processing',
        'issues': ['numpy_compatibility_conflict'],
        'solutions': [
            'create_virtual_environment',
            'use_online_services',
            'alternative_processing',
            'manual_analysis'
        ]
    }
    
    with open('vid_analysis_info.json', 'w') as f:
        json.dump(video_info, f, indent=2)
    
    print(f"\n📄 Analysis summary saved to: vid_analysis_info.json")
    
    return video_info

def main():
    """Main function"""
    analyze_video_file()
    
    print(f"\n🎯 Next Steps:")
    print("1. Try the virtual environment solution")
    print("2. Consider online YOLO services")
    print("3. Use alternative processing methods")
    print("4. Contact for additional support")

if __name__ == "__main__":
    main() 