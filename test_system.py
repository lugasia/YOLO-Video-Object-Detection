#!/usr/bin/env python3
"""
Test script for YOLO Video Detection System
This script tests the system with the existing video file in the workspace.
"""

import os
import sys
import time
from yolo_video_detector import YOLOVideoDetector

def test_video_processing():
    """Test video processing with the existing video file"""
    print("🎥 Testing YOLO Video Detection System")
    print("=" * 50)
    
    # Check if video file exists
    video_file = "Screen Recording 2025-08-12 at 21.56.27.mov"
    
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        print("Please ensure the video file is in the current directory.")
        return False
    
    print(f"✅ Found video file: {video_file}")
    print(f"📁 File size: {os.path.getsize(video_file) / (1024*1024):.2f} MB")
    
    try:
        # Initialize detector with lower confidence for testing
        print("\n🔧 Initializing YOLO detector...")
        detector = YOLOVideoDetector(confidence_threshold=0.3)
        print("✅ YOLO detector initialized successfully")
        
        # Process video (first 100 frames for quick testing)
        print("\n🎬 Processing video (first 100 frames for testing)...")
        start_time = time.time()
        
        # Create a modified version that processes only first 100 frames
        cap = detector.model.predict(video_file, conf=detector.confidence_threshold, 
                                   verbose=False, stream=True, max_det=100)
        
        frame_count = 0
        total_detections = 0
        
        for result in cap:
            if frame_count >= 100:  # Limit to 100 frames for testing
                break
                
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                if detections_in_frame > 0:
                    print(f"Frame {frame_count}: {detections_in_frame} detections")
            
            frame_count += 1
        
        processing_time = time.time() - start_time
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results:")
        print(f"   - Frames processed: {frame_count}")
        print(f"   - Total detections: {total_detections}")
        print(f"   - Processing time: {processing_time:.2f} seconds")
        print(f"   - FPS: {frame_count / processing_time:.2f}")
        
        if total_detections > 0:
            print(f"   - Average detections per frame: {total_detections / frame_count:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def test_frame_processing():
    """Test single frame processing"""
    print("\n🖼️ Testing single frame processing...")
    
    try:
        # Initialize detector
        detector = YOLOVideoDetector(confidence_threshold=0.5)
        
        # Create a simple test image (you could also load the first frame of the video)
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process frame
        annotated_frame, detections = detector.process_frame(test_frame)
        
        print(f"✅ Frame processing test completed")
        print(f"   - Input frame shape: {test_frame.shape}")
        print(f"   - Output frame shape: {annotated_frame.shape}")
        print(f"   - Detections found: {len(detections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in frame processing test: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting YOLO Video Detection System Tests")
    print("=" * 60)
    
    # Test 1: Check dependencies
    print("\n1️⃣ Checking dependencies...")
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✅ All required dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False
    
    # Test 2: Frame processing
    frame_test = test_frame_processing()
    
    # Test 3: Video processing
    video_test = test_video_processing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Frame Processing: {'✅ PASS' if frame_test else '❌ FAIL'}")
    print(f"   Video Processing: {'✅ PASS' if video_test else '❌ FAIL'}")
    
    if frame_test and video_test:
        print("\n🎉 All tests passed! The system is ready to use.")
        print("\n📖 Next steps:")
        print("   1. Run 'python yolo_video_detector.py' for command-line processing")
        print("   2. Run 'streamlit run streamlit_app.py' for web interface")
        print("   3. Run 'python flask_api.py' for real-time API")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    return frame_test and video_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 