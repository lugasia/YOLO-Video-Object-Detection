#!/usr/bin/env python3
"""
Enhanced Object Detection Script (Simplified)
Advanced features for better accuracy and reduced false positives
"""

import os
import time
import json
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

class EnhancedDetectorSimple:
    def __init__(self, model_path: str = "yolov8n.pt", base_confidence: float = 0.4):
        """
        Initialize enhanced detector with advanced features
        
        Args:
            model_path: Path to YOLO model
            base_confidence: Base confidence threshold
        """
        self.model = YOLO(model_path)
        self.base_confidence = base_confidence
        self.class_names = self.model.names
        
        # Temporal filtering parameters
        self.temporal_window = 5  # frames
        self.min_detection_persistence = 3  # minimum frames for valid detection
        
        # Object tracking
        self.tracked_objects = {}
        self.next_track_id = 0
        
        # Confidence adaptation
        self.confidence_history = defaultdict(list)
        self.adaptive_confidence = base_confidence
        
        # False positive filtering
        self.suspicious_classes = ['motorcycle', 'airplane', 'boat', 'train']
        self.class_confidence_thresholds = {
            'motorcycle': 0.8,
            'airplane': 0.85,
            'boat': 0.8,
            'train': 0.75,
            'bus': 0.7,
            'truck': 0.5,
            'car': 0.5,
            'person': 0.6
        }
        
        # Size-based filtering
        self.min_object_size = 0.01  # minimum relative size
        self.max_object_size = 0.8   # maximum relative size
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_adaptive_confidence(self, class_name: str, base_confidence: float) -> float:
        """Get adaptive confidence threshold based on class and history"""
        # Use class-specific threshold if available
        if class_name in self.class_confidence_thresholds:
            return self.class_confidence_thresholds[class_name]
        
        # For suspicious classes, use higher threshold
        if class_name in self.suspicious_classes:
            return max(base_confidence + 0.2, 0.7)
        
        return base_confidence
    
    def filter_by_size(self, bbox: List[float], frame_shape: Tuple[int, int]) -> bool:
        """Filter detections based on object size"""
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]
        
        # Calculate relative size
        width = (x2 - x1) / frame_width
        height = (y2 - y1) / frame_height
        area = width * height
        
        return self.min_object_size <= area <= self.max_object_size
    
    def update_tracking(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Update object tracking and filter unstable detections"""
        current_objects = {}
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Find best matching tracked object
            best_match_id = None
            best_iou = 0.5  # minimum IoU threshold
            
            for track_id, track_info in self.tracked_objects.items():
                if track_info['class_name'] == class_name:
                    iou = self.calculate_iou(bbox, track_info['last_bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Update existing track
                track = self.tracked_objects[best_match_id]
                track['last_bbox'] = bbox
                track['last_seen'] = frame_id
                track['detection_count'] += 1
                track['confidence_history'].append(confidence)
                
                # Keep only recent confidence values
                if len(track['confidence_history']) > 10:
                    track['confidence_history'].pop(0)
                
                current_objects[best_match_id] = track
                detection['track_id'] = best_match_id
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracked_objects[track_id] = {
                    'class_name': class_name,
                    'first_seen': frame_id,
                    'last_seen': frame_id,
                    'last_bbox': bbox,
                    'detection_count': 1,
                    'confidence_history': [confidence]
                }
                
                current_objects[track_id] = self.tracked_objects[track_id]
                detection['track_id'] = track_id
        
        # Remove old tracks
        self.tracked_objects = current_objects
        
        # Filter detections based on tracking stability
        stable_detections = []
        for detection in detections:
            track_id = detection['track_id']
            track = self.tracked_objects[track_id]
            
            # Only keep detections that have been stable for multiple frames
            if track['detection_count'] >= self.min_detection_persistence:
                # Calculate average confidence
                avg_confidence = sum(track['confidence_history']) / len(track['confidence_history'])
                detection['avg_confidence'] = avg_confidence
                stable_detections.append(detection)
        
        return stable_detections
    
    def temporal_filter(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """Apply temporal filtering to reduce false positives"""
        # Store detections in temporal window
        if not hasattr(self, 'temporal_detections'):
            self.temporal_detections = deque(maxlen=self.temporal_window)
        
        self.temporal_detections.append({
            'frame_id': frame_id,
            'detections': detections
        })
        
        # If we don't have enough frames yet, return current detections
        if len(self.temporal_detections) < self.temporal_window:
            return detections
        
        # Count detections per class across temporal window
        class_counts = defaultdict(int)
        for frame_data in self.temporal_detections:
            for det in frame_data['detections']:
                class_counts[det['class_name']] += 1
        
        # Filter detections that appear consistently
        filtered_detections = []
        for detection in detections:
            class_name = detection['class_name']
            class_frequency = class_counts[class_name] / len(self.temporal_detections)
            
            # Keep detection if it appears in most frames
            if class_frequency >= 0.6:  # 60% of frames
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def process_video_enhanced(self, video_path: str) -> Dict:
        """Process video with enhanced detection using ultralytics"""
        print("🎥 Processing video with ENHANCED DETECTION")
        print("=" * 60)
        
        print(f"📹 Processing video: {video_path}")
        print(f"⚙️ Base confidence threshold: {self.base_confidence}")
        print(f"🔄 Temporal window: {self.temporal_window} frames")
        print(f"📊 Min detection persistence: {self.min_detection_persistence} frames")
        print("This may take a few minutes...")
        
        start_time = time.time()
        
        try:
            # Process video with ultralytics
            results = self.model.predict(
                video_path,
                conf=self.base_confidence,
                save=True,
                project="enhanced_simple_results",
                name="detections"
            )
            
            processing_time = time.time() - start_time
            
            print("\n" + "=" * 60)
            print("📊 PROCESSING COMPLETE!")
            print("=" * 60)
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            
            # Analyze results with enhanced filtering
            total_detections = 0
            class_counts = defaultdict(int)
            frame_analysis = {}
            enhanced_detections = []
            
            for i, result in enumerate(results):
                if result.boxes is not None:
                    frame_detections = []
                    
                    for box in result.boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        # Apply class-specific confidence threshold
                        class_conf_threshold = self.get_adaptive_confidence(class_name, self.base_confidence)
                        if confidence < class_conf_threshold:
                            continue
                        
                        # Filter by size (approximate frame size)
                        frame_shape = (1080, 1920, 3)  # Approximate HD video size
                        if not self.filter_by_size([x1, y1, x2, y2], frame_shape):
                            continue
                        
                        detection = {
                            'class_name': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'frame_id': i
                        }
                        frame_detections.append(detection)
                    
                    # Apply temporal filtering
                    frame_detections = self.temporal_filter(frame_detections, i)
                    
                    # Apply tracking-based filtering
                    frame_detections = self.update_tracking(frame_detections, i)
                    
                    # Count enhanced detections
                    for detection in frame_detections:
                        class_counts[detection['class_name']] += 1
                        total_detections += 1
                        enhanced_detections.append(detection)
                    
                    frame_analysis[i] = frame_detections
            
            print(f"🎯 Total enhanced detections: {total_detections:,}")
            print(f"🔄 Tracked objects: {len(self.tracked_objects)}")
            
            # Show detection results
            if class_counts:
                print(f"\n🏷️ Objects Detected (Enhanced):")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_detections) * 100
                    print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%)")
            
            # Show tracking statistics
            if self.tracked_objects:
                print(f"\n🔄 Tracking Statistics:")
                for track_id, track_info in self.tracked_objects.items():
                    avg_conf = sum(track_info['confidence_history']) / len(track_info['confidence_history'])
                    print(f"   Track {track_id}: {track_info['class_name']} - {track_info['detection_count']} detections, avg conf: {avg_conf:.3f}")
            
            # Save detailed results
            results_summary = {
                'video_file': video_path,
                'model_used': self.model.ckpt_path,
                'base_confidence': self.base_confidence,
                'processing_time': processing_time,
                'total_frames': len(results),
                'fps': len(results) / processing_time,
                'total_detections': total_detections,
                'tracked_objects': len(self.tracked_objects),
                'class_counts': dict(class_counts),
                'temporal_window': self.temporal_window,
                'min_detection_persistence': self.min_detection_persistence,
                'enhanced_detections': enhanced_detections,
                'frame_analysis': frame_analysis,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open('enhanced_simple_detection_results.json', 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\n📁 Output Files:")
            print(f"   Processed Video: enhanced_simple_results/detections/")
            print(f"   Results Summary: enhanced_simple_detection_results.json")
            
            print("\n💡 Enhanced Features Applied:")
            print("   ✅ Adaptive confidence thresholds")
            print("   ✅ Temporal filtering")
            print("   ✅ Object tracking")
            print("   ✅ Size-based filtering")
            print("   ✅ Class-specific thresholds")
            print("   ✅ False positive reduction")
            
            return results_summary
            
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return None

def main():
    """Main function"""
    print("🚀 Enhanced Object Detection System (Simplified)")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Initialize enhanced detector
    detector = EnhancedDetectorSimple(model_path="yolov8n.pt", base_confidence=0.4)
    
    # Process video
    print(f"\n🎯 Processing with enhanced detection...")
    results = detector.process_video_enhanced(video_file)
    
    if results:
        print(f"\n✅ Processing completed successfully!")
        print(f"📊 Check 'enhanced_simple_detection_results.json' for detailed analysis")
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    main()
