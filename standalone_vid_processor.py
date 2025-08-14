#!/usr/bin/env python3
"""
Standalone Video Processor for vid.mov
Uses minimal dependencies to avoid NumPy conflicts
"""

import os
import time
import json
from ultralytics import YOLO

def process_video_standalone():
    """Process vid.mov using only essential dependencies"""
    print("🎥 Processing vid.mov with YOLO (Standalone Mode)")
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
    
    # Initialize YOLO model
    print("\n🔧 Loading YOLO model...")
    try:
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Process video
    print(f"\n🎬 Processing video: {video_file}")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        # Use YOLO's built-in video processing
        results = model.predict(
            video_file,
            conf=0.5,
            save=True,
            project="vid_results",
            name="detections"
        )
        
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        # Analyze results
        total_detections = 0
        class_counts = {}
        
        for result in results:
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"🎯 Total detections: {total_detections:,}")
        
        if class_counts:
            print(f"\n🏷️ Objects Detected:")
            for class_name, count in class_counts.items():
                percentage = (count / total_detections) * 100
                print(f"   {class_name}: {count:,} ({percentage:.1f}%)")
        
        # Save results summary
        results_summary = {
            'video_file': video_file,
            'processing_time': processing_time,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('vid_processing_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Processed Video: vid_results/detections/vid.mp4")
        print(f"   Results Summary: vid_processing_summary.json")
        
        print("\n🎉 Video processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return

def main():
    """Main function"""
    process_video_standalone()

if __name__ == "__main__":
    main() 