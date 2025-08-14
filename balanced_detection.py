#!/usr/bin/env python3
"""
Balanced Detection Script
Uses moderate confidence to reduce false positives while detecting real objects
"""

import os
import time
import json
from ultralytics import YOLO

def process_video_balanced(video_path, confidence=0.5):
    """Process video with balanced confidence to reduce false positives"""
    print("🎥 Processing video with BALANCED CONFIDENCE (0.5)")
    print("=" * 60)
    
    # Load model
    try:
        print(f"🔧 Loading YOLOv8 model...")
        model = YOLO("yolov8n.pt")
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print(f"\n🎬 Processing video: {video_path}")
    print(f"⚙️ Confidence threshold: {confidence} (BALANCED - reduce false positives)")
    print("This should catch real objects while filtering out false positives...")
    
    start_time = time.time()
    
    try:
        # Process video with balanced confidence
        results = model.predict(
            video_path,
            conf=confidence,  # Balanced confidence
            save=True,
            project="balanced_results",
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
        
        print(f"🎯 Total detections: {total_detections:,}")
        
        # Show all detections
        if class_counts:
            print(f"\n🏷️ Objects Detected (Balanced Confidence):")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%)")
        
        # Check for suspicious detections
        suspicious = ['motorcycle', 'airplane', 'boat', 'train']
        found_suspicious = []
        for class_name in suspicious:
            if class_name in class_counts:
                found_suspicious.append(f"{class_name}: {class_counts[class_name]}")
        
        if found_suspicious:
            print(f"\n⚠️ Still detected suspicious objects:")
            for item in found_suspicious:
                print(f"   - {item}")
        else:
            print(f"\n✅ No suspicious objects detected!")
        
        # Save results
        results_summary = {
            'video_file': video_path,
            'confidence_threshold': confidence,
            'processing_time': processing_time,
            'total_detections': total_detections,
            'total_frames': len(results),
            'fps': len(results) / processing_time,
            'class_counts': class_counts,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('balanced_detection_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Processed Video: balanced_results/detections/")
        print(f"   Results Summary: balanced_detection_results.json")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

def main():
    """Main function"""
    print("🎯 Balanced Object Detection")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Process with balanced confidence
    print(f"\n🎯 Processing with balanced confidence (0.5)...")
    results = process_video_balanced(video_file, confidence=0.5)
    
    if results:
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Check 'balanced_detection_results.json' for detailed analysis")
        print(f"🎬 Check the processed video to see the results")
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    main() 