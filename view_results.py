#!/usr/bin/env python3
"""
View detection results from the JSON file
"""

import json
import pandas as pd
from collections import Counter

def analyze_detections():
    """Analyze the detection results"""
    try:
        with open('detection_results.json', 'r') as f:
            data = json.load(f)
        
        summary = data['summary']
        detection_history = data['detection_history']
        
        print("🎯 YOLO Detection Results Analysis")
        print("=" * 50)
        
        # Overall summary
        print(f"📊 Overall Statistics:")
        print(f"   Total frames processed: {summary['total_frames']}")
        print(f"   Total detections: {summary['total_detections']}")
        print(f"   Processing time: {summary['processing_time']:.2f} seconds")
        print(f"   Average FPS: {summary['total_frames'] / summary['processing_time']:.2f}")
        
        # Detection by class
        print(f"\n🏷️ Detections by Object Class:")
        for class_name, count in summary['class_counts'].items():
            percentage = (count / summary['total_detections']) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Frame-by-frame analysis
        print(f"\n📈 Frame Analysis:")
        frames_with_detections = sum(1 for frame in detection_history if frame['detections'])
        print(f"   Frames with detections: {frames_with_detections}")
        print(f"   Frames without detections: {summary['total_frames'] - frames_with_detections}")
        print(f"   Detection rate: {(frames_with_detections / summary['total_frames']) * 100:.1f}%")
        
        # Confidence analysis
        all_confidences = []
        for frame in detection_history:
            for detection in frame['detections']:
                all_confidences.append(detection['confidence'])
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)
            
            print(f"\n🎯 Confidence Analysis:")
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Minimum confidence: {min_confidence:.3f}")
            print(f"   Maximum confidence: {max_confidence:.3f}")
        
        # Sample detections from first few frames
        print(f"\n🔍 Sample Detections (First 5 frames):")
        for i, frame in enumerate(detection_history[:5]):
            if frame['detections']:
                print(f"   Frame {frame['frame']}:")
                for detection in frame['detections']:
                    print(f"     - {detection['class_name']}: {detection['confidence']:.3f}")
            else:
                print(f"   Frame {frame['frame']}: No detections")
        
        return True
        
    except FileNotFoundError:
        print("❌ detection_results.json not found!")
        print("Please run the video processing first.")
        return False
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False

if __name__ == "__main__":
    analyze_detections() 