#!/usr/bin/env python3
"""
Improved Object Detection Script
Better parameters, multiple models, and post-processing to reduce false positives
"""

import os
import time
import json
from ultralytics import YOLO
import tempfile

def get_available_models():
    """Get list of available YOLO models"""
    models = {
        'yolov8n.pt': 'Nano (fastest, smallest)',
        'yolov8s.pt': 'Small (balanced)',
        'yolov8m.pt': 'Medium (better accuracy)',
        'yolov8l.pt': 'Large (high accuracy)',
        'yolov8x.pt': 'XLarge (best accuracy)'
    }
    return models

def load_yolo_model(model_path="yolov8n.pt"):
    """Load YOLO model with better error handling"""
    try:
        print(f"🔧 Loading model: {model_path}")
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return None

def process_video_improved(video_path, model_path="yolov8n.pt", confidence=0.5, iou=0.45):
    """Process video with improved parameters to reduce false positives"""
    print("🎥 Processing video with improved detection parameters")
    print("=" * 60)
    
    # Load model
    model = load_yolo_model(model_path)
    if model is None:
        return None
    
    print(f"\n🎬 Processing video: {video_path}")
    print(f"⚙️ Confidence threshold: {confidence} (higher = fewer false positives)")
    print(f"🔗 IoU threshold: {iou} (higher = better overlap detection)")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        # Process video with better parameters
        results = model.predict(
            video_path,
            conf=confidence,  # Higher confidence to reduce false positives
            iou=iou,          # IoU threshold for NMS
            save=True,
            project="improved_results",
            name="detections"
        )
        
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        # Analyze results with focus on accuracy
        total_detections = 0
        class_counts = {}
        frame_analysis = {}
        
        for i, result in enumerate(results):
            if result.boxes is not None:
                detections_in_frame = len(result.boxes)
                total_detections += detections_in_frame
                
                frame_detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    confidence_score = float(box.conf[0].cpu().numpy())
                    
                    # Count all detections
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Store frame-level detection
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence_score,
                        'bbox': box.xyxy[0].cpu().numpy().tolist()
                    })
                
                frame_analysis[i] = frame_detections
        
        print(f"🎯 Total detections: {total_detections:,}")
        
        # Show all detections with confidence analysis
        if class_counts:
            print(f"\n🏷️ All Objects Detected (with confidence analysis):")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                
                # Calculate average confidence for this class
                confidences = []
                for frame_data in frame_analysis.values():
                    for det in frame_data:
                        if det['class'] == class_name:
                            confidences.append(det['confidence'])
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                high_conf_count = len([c for c in confidences if c > 0.7])
                
                print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%) - Avg Conf: {avg_confidence:.3f}, High Conf: {high_conf_count}")
        
        # Identify potential false positives
        print(f"\n⚠️ Potential False Positives Analysis:")
        suspicious_classes = ['motorcycle', 'airplane', 'boat', 'train', 'bus']
        for class_name in suspicious_classes:
            if class_name in class_counts:
                count = class_counts[class_name]
                confidences = []
                for frame_data in frame_analysis.values():
                    for det in frame_data:
                        if det['class'] == class_name:
                            confidences.append(det['confidence'])
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                low_conf_count = len([c for c in confidences if c < 0.6])
                
                if avg_confidence < 0.6 or low_conf_count > count * 0.5:
                    print(f"   ⚠️ {class_name}: {count} detections, Avg Conf: {avg_confidence:.3f} (likely false positives)")
                else:
                    print(f"   ✅ {class_name}: {count} detections, Avg Conf: {avg_confidence:.3f} (likely real)")
        
        # Save detailed results
        results_summary = {
            'video_file': video_path,
            'model_used': model_path,
            'confidence_threshold': confidence,
            'iou_threshold': iou,
            'processing_time': processing_time,
            'total_detections': total_detections,
            'total_frames': len(results),
            'fps': len(results) / processing_time,
            'all_class_counts': class_counts,
            'frame_analysis': frame_analysis,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('improved_detection_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Processed Video: improved_results/detections/")
        print(f"   Results Summary: improved_detection_results.json")
        
        print("\n💡 Recommendations:")
        print("   - If you see false positives, increase confidence threshold")
        print("   - If you miss objects, decrease confidence threshold")
        print("   - Try larger models (yolov8m, yolov8l) for better accuracy")
        print("   - Check the processed video to verify detections")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

def test_multiple_models(video_path):
    """Test multiple models to find the best one"""
    print("🧪 Testing Multiple Models for Best Results")
    print("=" * 60)
    
    models_to_test = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🔬 Testing {model_name}...")
        try:
            result = process_video_improved(video_path, model_name, confidence=0.5, iou=0.45)
            if result:
                results[model_name] = result
                print(f"✅ {model_name} completed successfully")
            else:
                print(f"❌ {model_name} failed")
        except Exception as e:
            print(f"❌ {model_name} error: {e}")
    
    # Compare results
    if results:
        print(f"\n📊 Model Comparison:")
        print("-" * 50)
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"   Total Detections: {result['total_detections']:,}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")
            print(f"   FPS: {result['fps']:.1f}")
            print(f"   Classes Detected: {len(result['all_class_counts'])}")
            
            # Show top 3 classes
            top_classes = sorted(result['all_class_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
            for class_name, count in top_classes:
                print(f"     - {class_name}: {count:,}")
    
    return results

def main():
    """Main function"""
    print("🎯 Improved Object Detection System")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Show options
    print("\n🔧 Detection Options:")
    print("1. Single model with improved parameters")
    print("2. Test multiple models")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single model with improved parameters
        confidence = float(input("Enter confidence threshold (0.3-0.8, default 0.5): ") or "0.5")
        iou = float(input("Enter IoU threshold (0.3-0.7, default 0.45): ") or "0.45")
        
        print(f"\n🎯 Processing with improved parameters...")
        results = process_video_improved(video_file, "yolov8n.pt", confidence, iou)
        
        if results:
            print(f"\n✅ Processing completed successfully!")
            print(f"📊 Check 'improved_detection_results.json' for detailed analysis")
        else:
            print(f"\n❌ Processing failed!")
    
    elif choice == "2":
        # Test multiple models
        results = test_multiple_models(video_file)
        if results:
            print(f"\n✅ Multi-model testing completed!")
            print(f"📊 Check individual result files for detailed analysis")
        else:
            print(f"\n❌ Multi-model testing failed!")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main() 