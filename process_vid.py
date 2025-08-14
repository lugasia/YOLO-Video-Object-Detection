#!/usr/bin/env python3
"""
Process vid.mov with YOLO Video Detection System
"""

import os
import time
from yolo_video_detector import YOLOVideoDetector

def main():
    """Process the vid.mov file"""
    print("🎥 Processing vid.mov with YOLO Video Detection System")
    print("=" * 60)
    
    # Check if video file exists
    video_file = "vid.mov"
    
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Get file info
    file_size = os.path.getsize(video_file) / (1024*1024)
    print(f"✅ Found video file: {video_file}")
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Initialize detector
    print("\n🔧 Initializing YOLO detector...")
    detector = YOLOVideoDetector(confidence_threshold=0.5)
    print("✅ YOLO detector initialized successfully")
    
    # Process video
    print(f"\n🎬 Processing video: {video_file}")
    output_video = "vid_detected.mp4"
    
    start_time = time.time()
    detector.process_video_file(video_file, output_video)
    
    # Save detection results
    results_file = "vid_detection_results.json"
    detector.save_detection_results(results_file)
    
    # Print summary
    summary = detector.get_detection_summary()
    print("\n" + "=" * 60)
    print("📊 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"🎬 Total frames processed: {summary['total_frames']:,}")
    print(f"🎯 Total detections: {summary['total_detections']:,}")
    print(f"⏱️  Processing time: {summary['processing_time']:.2f} seconds")
    print(f"🚀 Average FPS: {summary['total_frames'] / summary['processing_time']:.2f}")
    
    print(f"\n🏷️ Objects Detected:")
    for class_name, count in summary['class_counts'].items():
        percentage = (count / summary['total_detections']) * 100
        print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📁 Output Files:")
    print(f"   Processed Video: {output_video}")
    print(f"   Detection Results: {results_file}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
    print("🎉 Video processing completed successfully!")

if __name__ == "__main__":
    main() 