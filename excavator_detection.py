#!/usr/bin/env python3
"""
Specialized Excavator Detection Script
Uses better models and custom class mapping for construction equipment
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

def get_construction_classes():
    """Get classes that might be relevant for construction equipment"""
    construction_related = {
        'truck': 'Truck/Dump Truck',
        'car': 'Car/Vehicle',
        'bus': 'Bus',
        'train': 'Train (might be excavator)',
        'boat': 'Boat',
        'airplane': 'Airplane',
        'motorcycle': 'Motorcycle',
        'bicycle': 'Bicycle',
        'person': 'Person',
        'dog': 'Dog',
        'horse': 'Horse',
        'sheep': 'Sheep',
        'cow': 'Cow',
        'elephant': 'Elephant',
        'bear': 'Bear',
        'zebra': 'Zebra',
        'giraffe': 'Giraffe',
        'backpack': 'Backpack',
        'umbrella': 'Umbrella',
        'handbag': 'Handbag',
        'tie': 'Tie',
        'suitcase': 'Suitcase',
        'frisbee': 'Frisbee',
        'skis': 'Skis',
        'snowboard': 'Snowboard',
        'sports ball': 'Sports Ball',
        'kite': 'Kite',
        'baseball bat': 'Baseball Bat',
        'baseball glove': 'Baseball Glove',
        'skateboard': 'Skateboard',
        'surfboard': 'Surfboard',
        'tennis racket': 'Tennis Racket',
        'bottle': 'Bottle',
        'wine glass': 'Wine Glass',
        'cup': 'Cup',
        'fork': 'Fork',
        'knife': 'Knife',
        'spoon': 'Spoon',
        'bowl': 'Bowl',
        'banana': 'Banana',
        'apple': 'Apple',
        'sandwich': 'Sandwich',
        'orange': 'Orange',
        'broccoli': 'Broccoli',
        'carrot': 'Carrot',
        'hot dog': 'Hot Dog',
        'pizza': 'Pizza',
        'donut': 'Donut',
        'cake': 'Cake',
        'chair': 'Chair',
        'couch': 'Couch',
        'potted plant': 'Potted Plant',
        'bed': 'Bed',
        'dining table': 'Dining Table',
        'toilet': 'Toilet',
        'tv': 'TV',
        'laptop': 'Laptop',
        'mouse': 'Mouse',
        'remote': 'Remote',
        'keyboard': 'Keyboard',
        'cell phone': 'Cell Phone',
        'microwave': 'Microwave',
        'oven': 'Oven',
        'toaster': 'Toaster',
        'sink': 'Sink',
        'refrigerator': 'Refrigerator',
        'book': 'Book',
        'clock': 'Clock',
        'vase': 'Vase',
        'scissors': 'Scissors',
        'teddy bear': 'Teddy Bear',
        'hair drier': 'Hair Drier',
        'toothbrush': 'Toothbrush'
    }
    return construction_related

def process_video_for_excavators(video_path, model_path="yolov8n.pt", confidence=0.3):
    """Process video with focus on construction equipment detection"""
    print("🎥 Processing video for construction equipment detection")
    print("=" * 60)
    
    # Load model
    model = load_yolo_model(model_path)
    if model is None:
        return None
    
    # Get construction-related classes
    construction_classes = get_construction_classes()
    
    print(f"\n📋 Available classes for detection:")
    print("Construction equipment might be detected as:")
    relevant_classes = ['truck', 'car', 'bus', 'train', 'boat', 'airplane']
    for class_name in relevant_classes:
        if class_name in construction_classes:
            print(f"   - {class_name}: {construction_classes[class_name]}")
    
    print(f"\n🎬 Processing video: {video_path}")
    print(f"⚙️ Confidence threshold: {confidence}")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    try:
        # Process video with lower confidence for better detection
        results = model.predict(
            video_path,
            conf=confidence,  # Lower confidence to catch more objects
            save=True,
            project="excavator_results",
            name="detections"
        )
        
        processing_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("📊 PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        # Analyze results with focus on construction equipment
        total_detections = 0
        class_counts = {}
        construction_detections = {}
        
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
                    
                    # Focus on construction-related detections
                    if class_name in ['truck', 'car', 'bus', 'train', 'boat', 'airplane']:
                        if class_name not in construction_detections:
                            construction_detections[class_name] = []
                        construction_detections[class_name].append({
                            'frame': i,
                            'confidence': confidence_score,
                            'description': construction_classes.get(class_name, class_name)
                        })
        
        print(f"🎯 Total detections: {total_detections:,}")
        
        # Show all detections
        if class_counts:
            print(f"\n🏷️ All Objects Detected:")
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%)")
        
        # Show construction equipment analysis
        if construction_detections:
            print(f"\n🚧 Construction Equipment Analysis:")
            print("-" * 50)
            for class_name, detections in construction_detections.items():
                print(f"\n🔍 {class_name.upper()} (Possible Excavator/Construction Equipment):")
                print(f"   Total detections: {len(detections)}")
                
                # Show high-confidence detections
                high_conf = [d for d in detections if d['confidence'] > 0.7]
                if high_conf:
                    print(f"   High confidence (>70%): {len(high_conf)}")
                    for det in high_conf[:5]:  # Show first 5
                        print(f"     Frame {det['frame']}: {det['confidence']:.3f}")
                
                # Show frame ranges
                frames = [d['frame'] for d in detections]
                if frames:
                    print(f"   Frame range: {min(frames)} - {max(frames)}")
                    print(f"   Description: {detections[0]['description']}")
        
        # Save detailed results
        results_summary = {
            'video_file': video_path,
            'model_used': model_path,
            'confidence_threshold': confidence,
            'processing_time': processing_time,
            'total_detections': total_detections,
            'total_frames': len(results),
            'fps': len(results) / processing_time,
            'all_class_counts': class_counts,
            'construction_detections': construction_detections,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('excavator_detection_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Processed Video: excavator_results/detections/")
        print(f"   Results Summary: excavator_detection_results.json")
        
        print("\n💡 Analysis:")
        print("   - 'train' detections might actually be excavators")
        print("   - 'truck' detections might be dump trucks or construction vehicles")
        print("   - Check high-confidence detections for better accuracy")
        print("   - Consider using a larger model for better classification")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        return None

def main():
    """Main function"""
    print("🚧 Excavator Detection System")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Show available models
    models = get_available_models()
    print("\n📋 Available YOLO Models:")
    for model_name, description in models.items():
        print(f"   {model_name}: {description}")
    
    # Process with different models
    print(f"\n🎯 Processing with YOLOv8 Nano (default)")
    results = process_video_for_excavators(video_file, "yolov8n.pt", confidence=0.3)
    
    if results:
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Check 'excavator_detection_results.json' for detailed analysis")
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    main() 