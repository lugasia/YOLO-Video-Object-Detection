#!/usr/bin/env python3
"""
Demo script to showcase YOLO video detection results
"""

import os
import json
import subprocess
import webbrowser
from pathlib import Path

def show_demo():
    """Show demo of the detection results"""
    print("🎥 YOLO Video Detection System - Demo")
    print("=" * 50)
    
    # Check for processed files
    files = {
        'Original Video': 'Screen Recording 2025-08-12 at 21.56.27.mov',
        'Processed Video': 'output_detected_video.mp4',
        'Detection Results': 'detection_results.json',
        'YOLO Model': 'yolov8n.pt'
    }
    
    print("\n📁 Available Files:")
    for name, filename in files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename) / (1024*1024)
            print(f"✅ {name}: {filename} ({size:.1f} MB)")
        else:
            print(f"❌ {name}: {filename} (not found)")
    
    # Show detection summary
    if os.path.exists('detection_results.json'):
        print("\n📊 Detection Summary:")
        try:
            with open('detection_results.json', 'r') as f:
                data = json.load(f)
            
            summary = data['summary']
            print(f"   Total frames processed: {summary['total_frames']}")
            print(f"   Total detections: {summary['total_detections']}")
            print(f"   Processing time: {summary['processing_time']:.2f} seconds")
            print(f"   Average FPS: {summary['total_frames'] / summary['processing_time']:.2f}")
            
            print(f"\n🏷️ Objects Detected:")
            for class_name, count in summary['class_counts'].items():
                percentage = (count / summary['total_detections']) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"   Error reading results: {e}")
    
    # Show usage instructions
    print("\n🚀 How to Use:")
    print("1. Command Line Processing:")
    print("   python yolo_video_detector.py")
    
    print("\n2. Web Interface (Streamlit):")
    print("   streamlit run streamlit_app_simple.py")
    print("   Then open http://localhost:8501 in your browser")
    
    print("\n3. Real-time API (Flask):")
    print("   python flask_api.py")
    print("   Then open http://localhost:5000 in your browser")
    
    # Offer to open web interfaces
    print("\n🌐 Open Web Interfaces:")
    try:
        choice = input("Open Streamlit interface? (y/n): ").lower().strip()
        if choice == 'y':
            print("Opening Streamlit interface...")
            subprocess.Popen(["streamlit", "run", "streamlit_app_simple.py"])
            webbrowser.open("http://localhost:8501")
    except KeyboardInterrupt:
        print("\nDemo completed!")
    
    print("\n🎯 System Features:")
    print("✅ Real-time object detection with YOLOv8")
    print("✅ Video file processing with bounding boxes")
    print("✅ Multiple object class recognition")
    print("✅ Confidence threshold adjustment")
    print("✅ Detection analytics and statistics")
    print("✅ Export processed videos and results")
    print("✅ Web-based user interface")
    print("✅ API endpoints for integration")

if __name__ == "__main__":
    show_demo() 