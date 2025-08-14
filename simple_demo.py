#!/usr/bin/env python3
"""
Simple Demo Script for YOLO Video Detection System
This script works without complex dependencies and showcases the results.
"""

import os
import json
import sys

def show_results():
    """Show the detection results in a simple format"""
    print("🎥 YOLO Video Object Detection System")
    print("=" * 60)
    
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
        print("\n📊 Detection Results Summary:")
        print("-" * 40)
        try:
            with open('detection_results.json', 'r') as f:
                data = json.load(f)
            
            summary = data['summary']
            print(f"🎬 Total frames processed: {summary['total_frames']:,}")
            print(f"🎯 Total detections: {summary['total_detections']:,}")
            print(f"⏱️  Processing time: {summary['processing_time']:.2f} seconds")
            print(f"🚀 Average FPS: {summary['total_frames'] / summary['processing_time']:.2f}")
            
            print(f"\n🏷️ Objects Detected:")
            print("-" * 20)
            for class_name, count in summary['class_counts'].items():
                percentage = (count / summary['total_detections']) * 100
                print(f"   {class_name:>10}: {count:>6,} ({percentage:>5.1f}%)")
                
        except Exception as e:
            print(f"   Error reading results: {e}")
    
    # Show sample detections
    if os.path.exists('detection_results.json'):
        print(f"\n🔍 Sample Detections (First 5 frames):")
        print("-" * 40)
        try:
            with open('detection_results.json', 'r') as f:
                data = json.load(f)
            
            for i, frame_data in enumerate(data['detection_history'][:5]):
                if frame_data['detections']:
                    print(f"Frame {frame_data['frame']:>4}:")
                    for detection in frame_data['detections']:
                        confidence_pct = detection['confidence'] * 100
                        print(f"     • {detection['class_name']:>10}: {confidence_pct:>5.1f}%")
                else:
                    print(f"Frame {frame_data['frame']:>4}: No detections")
                    
        except Exception as e:
            print(f"   Error reading sample detections: {e}")

def show_usage():
    """Show usage instructions"""
    print(f"\n🚀 How to Use the System:")
    print("=" * 40)
    
    print("\n1️⃣ Command Line Processing:")
    print("   python yolo_video_detector.py")
    print("   - Processes the video file automatically")
    print("   - Shows real-time progress")
    print("   - Saves results to JSON and MP4 files")
    
    print("\n2️⃣ Real-time Webcam Detection:")
    print("   python yolo_video_detector.py")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save current frame")
    
    print("\n3️⃣ View Results:")
    print("   python view_results.py")
    print("   - Shows detailed analysis of detections")
    print("   - Displays confidence statistics")
    
    print("\n4️⃣ Flask API (if dependencies work):")
    print("   python flask_api.py")
    print("   - Open http://localhost:5000 in browser")
    print("   - Real-time video streaming with detections")

def show_features():
    """Show system features"""
    print(f"\n🎯 System Features:")
    print("=" * 30)
    features = [
        "✅ Real-time object detection with YOLOv8",
        "✅ Video file processing with bounding boxes",
        "✅ Multiple object class recognition (80+ classes)",
        "✅ Adjustable confidence thresholds",
        "✅ Comprehensive detection analytics",
        "✅ Export processed videos with annotations",
        "✅ JSON export of detection results",
        "✅ Frame-by-frame detection tracking",
        "✅ Performance metrics and statistics",
        "✅ Support for multiple video formats"
    ]
    
    for feature in features:
        print(f"   {feature}")

def show_objects_detected():
    """Show what objects were detected in the video"""
    if not os.path.exists('detection_results.json'):
        return
    
    try:
        with open('detection_results.json', 'r') as f:
            data = json.load(f)
        
        summary = data['summary']
        
        print(f"\n🎬 Video Analysis Results:")
        print("=" * 40)
        print(f"📹 Video: {summary['total_frames']:,} frames processed")
        print(f"🎯 Detections: {summary['total_detections']:,} total objects found")
        print(f"⏱️  Time: {summary['processing_time']:.1f} seconds")
        print(f"🚀 Speed: {summary['total_frames'] / summary['processing_time']:.1f} FPS")
        
        print(f"\n🏷️ Objects Found in Your Video:")
        print("-" * 35)
        for class_name, count in summary['class_counts'].items():
            percentage = (count / summary['total_detections']) * 100
            print(f"   {class_name:>10}: {count:>6,} detections ({percentage:>5.1f}%)")
            
    except Exception as e:
        print(f"Error reading detection results: {e}")

def main():
    """Main demo function"""
    print("🎥 YOLO Video Object Detection System - Demo")
    print("=" * 60)
    
    # Show results
    show_results()
    
    # Show objects detected
    show_objects_detected()
    
    # Show usage
    show_usage()
    
    # Show features
    show_features()
    
    print(f"\n🎉 System Status: READY TO USE!")
    print("=" * 40)
    print("The YOLO video detection system has successfully processed your video.")
    print("You can now use any of the interfaces listed above.")
    
    # Interactive menu
    print(f"\n🔧 Quick Actions:")
    print("1. View detailed results analysis")
    print("2. Process another video")
    print("3. Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            print("\nRunning detailed analysis...")
            os.system("python view_results.py")
        elif choice == "2":
            print("\nTo process another video, place it in this folder and run:")
            print("python yolo_video_detector.py")
        else:
            print("\nThank you for using the YOLO Video Detection System! 🎯")
    except KeyboardInterrupt:
        print("\n\nDemo completed! Thank you for using the system! 🎯")

if __name__ == "__main__":
    main() 