#!/usr/bin/env python3
"""
Enhanced Object Detection Script
Advanced features for better accuracy and reduced false positives
"""

import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import tempfile
from typing import Dict, List, Tuple, Optional

class EnhancedDetector:
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
                avg_confidence = np.mean(track['confidence_history'])
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
    
    def process_frame_enhanced(self, frame: np.ndarray, frame_id: int) -> List[Dict]:
        """Process a single frame with enhanced filtering"""
        # Get adaptive confidence for this frame
        adaptive_conf = self.adaptive_confidence
        
        # Run YOLO detection
        results = self.model(frame, conf=adaptive_conf, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get detection info
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Apply class-specific confidence threshold
                    class_conf_threshold = self.get_adaptive_confidence(class_name, adaptive_conf)
                    if confidence < class_conf_threshold:
                        continue
                    
                    # Filter by size
                    if not self.filter_by_size([x1, y1, x2, y2], frame.shape):
                        continue
                    
                    detection = {
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'frame_id': frame_id
                    }
                    detections.append(detection)
        
        # Apply temporal filtering
        detections = self.temporal_filter(detections, frame_id)
        
        # Apply tracking-based filtering
        detections = self.update_tracking(detections, frame_id)
        
        return detections
    
    def process_video_enhanced(self, video_path: str, output_path: str = None) -> Dict:
        """Process video with enhanced detection"""
        print("🎥 Processing video with ENHANCED DETECTION")
        print("=" * 60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        start_time = time.time()
        all_detections = []
        class_counts = defaultdict(int)
        
        print("🔄 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with enhanced detection
            detections = self.process_frame_enhanced(frame, frame_count)
            
            # Count detections
            for detection in detections:
                class_counts[detection['class_name']] += 1
            
            all_detections.append({
                'frame_id': frame_count,
                'detections': detections
            })
            
            # Draw detections on frame
            annotated_frame = frame.copy()
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                class_name = detection['class_name']
                confidence = detection['confidence']
                track_id = detection.get('track_id', 'N/A')
                
                # Choose color based on class
                color = (0, 255, 0) if class_name == 'truck' else (255, 0, 0)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f} (ID: {track_id})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Progress update
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if writer:
            writer.release()
        
        processing_time = time.time() - start_time
        
        # Generate results summary
        results_summary = {
            'video_file': video_path,
            'model_used': self.model.ckpt_path,
            'base_confidence': self.base_confidence,
            'processing_time': processing_time,
            'total_frames': frame_count,
            'fps': frame_count / processing_time,
            'total_detections': sum(class_counts.values()),
            'class_counts': dict(class_counts),
            'tracked_objects': len(self.tracked_objects),
            'temporal_window': self.temporal_window,
            'min_detection_persistence': self.min_detection_persistence,
            'all_detections': all_detections,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results_summary

def main():
    """Main function"""
    print("🚀 Enhanced Object Detection System")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Initialize enhanced detector
    detector = EnhancedDetector(model_path="yolov8n.pt", base_confidence=0.4)
    
    # Process video
    print(f"\n🎯 Processing with enhanced detection...")
    results = detector.process_video_enhanced(video_file, "enhanced_output.mp4")
    
    if results:
        print("\n" + "=" * 60)
        print("📊 ENHANCED PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Processing time: {results['processing_time']:.2f} seconds")
        print(f"🎯 Total detections: {results['total_detections']:,}")
        print(f"📈 Processing FPS: {results['fps']:.1f}")
        print(f"🔄 Tracked objects: {results['tracked_objects']}")
        
        # Show detection results
        if results['class_counts']:
            print(f"\n🏷️ Objects Detected (Enhanced):")
            for class_name, count in sorted(results['class_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / results['total_detections']) * 100
                print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%)")
        
        # Save results
        with open('enhanced_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Enhanced Video: enhanced_output.mp4")
        print(f"   Results Summary: enhanced_detection_results.json")
        
        print("\n💡 Enhanced Features Applied:")
        print("   ✅ Adaptive confidence thresholds")
        print("   ✅ Temporal filtering")
        print("   ✅ Object tracking")
        print("   ✅ Size-based filtering")
        print("   ✅ Class-specific thresholds")
        print("   ✅ False positive reduction")
        
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    main()
