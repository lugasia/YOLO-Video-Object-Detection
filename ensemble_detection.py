#!/usr/bin/env python3
"""
Ensemble Object Detection Script
Combines multiple YOLO models for improved accuracy
"""

import os
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import tempfile

class EnsembleDetector:
    def __init__(self, model_paths: List[str] = None, base_confidence: float = 0.4):
        """
        Initialize ensemble detector with multiple models
        
        Args:
            model_paths: List of YOLO model paths
            base_confidence: Base confidence threshold
        """
        if model_paths is None:
            model_paths = ["yolov8n.pt", "yolov8s.pt"]
        
        self.models = {}
        self.model_weights = {}
        self.base_confidence = base_confidence
        
        # Load models with weights (based on model size/accuracy)
        model_weights_map = {
            'yolov8n.pt': 0.3,  # Nano - faster but less accurate
            'yolov8s.pt': 0.4,  # Small - balanced
            'yolov8m.pt': 0.5,  # Medium - better accuracy
            'yolov8l.pt': 0.6,  # Large - high accuracy
            'yolov8x.pt': 0.7   # XLarge - best accuracy
        }
        
        print("🔧 Loading ensemble models...")
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = YOLO(model_path)
                    self.models[model_path] = model
                    self.model_weights[model_path] = model_weights_map.get(model_path, 0.5)
                    print(f"✅ Loaded {model_path} (weight: {self.model_weights[model_path]})")
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
            else:
                print(f"⚠️ Model {model_path} not found, skipping...")
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
        
        # Ensemble parameters
        self.min_votes = max(1, len(self.models) // 2)  # At least half models must agree
        self.iou_threshold = 0.5  # IoU threshold for merging detections
        self.confidence_boost = 0.1  # Boost confidence for ensemble agreement
        
        # Class-specific parameters
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
        
        print(f"🎯 Ensemble initialized with {len(self.models)} models")
        print(f"📊 Minimum votes required: {self.min_votes}")
    
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
    
    def merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge overlapping detections from ensemble"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            # Find overlapping detections
            overlapping = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if same class and overlapping
                if (det1['class_name'] == det2['class_name'] and 
                    self.calculate_iou(det1['bbox'], det2['bbox']) > self.iou_threshold):
                    overlapping.append(det2)
                    used.add(j)
            
            if len(overlapping) >= self.min_votes:
                # Merge overlapping detections
                merged_det = self._merge_overlapping_detections(overlapping)
                merged.append(merged_det)
        
        return merged
    
    def _merge_overlapping_detections(self, detections: List[Dict]) -> Dict:
        """Merge multiple overlapping detections into one"""
        # Calculate weighted average confidence
        total_weight = 0
        weighted_conf = 0
        
        for det in detections:
            weight = det.get('model_weight', 1.0)
            total_weight += weight
            weighted_conf += det['confidence'] * weight
        
        avg_confidence = weighted_conf / total_weight if total_weight > 0 else 0
        
        # Boost confidence for ensemble agreement
        if len(detections) > self.min_votes:
            avg_confidence = min(1.0, avg_confidence + self.confidence_boost)
        
        # Calculate average bounding box
        avg_bbox = np.mean([det['bbox'] for det in detections], axis=0)
        
        # Use the most common class name
        class_names = [det['class_name'] for det in detections]
        class_name = Counter(class_names).most_common(1)[0][0]
        
        return {
            'class_name': class_name,
            'confidence': avg_confidence,
            'bbox': avg_bbox.tolist(),
            'ensemble_votes': len(detections),
            'model_agreement': len(detections) / len(self.models)
        }
    
    def get_class_confidence_threshold(self, class_name: str) -> float:
        """Get confidence threshold for specific class"""
        return self.class_confidence_thresholds.get(class_name, self.base_confidence)
    
    def process_frame_ensemble(self, frame: np.ndarray) -> List[Dict]:
        """Process frame with ensemble of models"""
        all_detections = []
        
        # Get detections from each model
        for model_path, model in self.models.items():
            try:
                results = model(frame, conf=self.base_confidence, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get detection info
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            class_name = model.names[class_id]
                            
                            # Apply class-specific threshold
                            class_threshold = self.get_class_confidence_threshold(class_name)
                            if confidence < class_threshold:
                                continue
                            
                            detection = {
                                'class_name': class_name,
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': [x1, y1, x2, y2],
                                'model_path': model_path,
                                'model_weight': self.model_weights[model_path]
                            }
                            all_detections.append(detection)
            
            except Exception as e:
                print(f"⚠️ Error processing with {model_path}: {e}")
                continue
        
        # Merge overlapping detections
        merged_detections = self.merge_detections(all_detections)
        
        return merged_detections
    
    def process_video_ensemble(self, video_path: str, output_path: str = None) -> Dict:
        """Process video with ensemble detection"""
        print("🎥 Processing video with ENSEMBLE DETECTION")
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
        print(f"🤖 Ensemble: {len(self.models)} models")
        
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
        ensemble_stats = {
            'total_votes': 0,
            'high_agreement_detections': 0,
            'model_agreements': []
        }
        
        print("🔄 Processing frames with ensemble...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with ensemble
            detections = self.process_frame_ensemble(frame)
            
            # Count detections and collect stats
            for detection in detections:
                class_counts[detection['class_name']] += 1
                ensemble_stats['total_votes'] += detection['ensemble_votes']
                
                if detection['model_agreement'] >= 0.8:  # 80% model agreement
                    ensemble_stats['high_agreement_detections'] += 1
                
                ensemble_stats['model_agreements'].append(detection['model_agreement'])
            
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
                votes = detection['ensemble_votes']
                agreement = detection['model_agreement']
                
                # Choose color based on agreement level
                if agreement >= 0.8:
                    color = (0, 255, 0)  # Green for high agreement
                elif agreement >= 0.6:
                    color = (0, 255, 255)  # Yellow for medium agreement
                else:
                    color = (0, 0, 255)  # Red for low agreement
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with ensemble info
                label = f"{class_name}: {confidence:.2f} (Votes: {votes})"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Models: {len(self.models)}", (10, 110),
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
        
        # Calculate ensemble statistics
        avg_agreement = np.mean(ensemble_stats['model_agreements']) if ensemble_stats['model_agreements'] else 0
        
        # Generate results summary
        results_summary = {
            'video_file': video_path,
            'models_used': list(self.models.keys()),
            'model_weights': self.model_weights,
            'base_confidence': self.base_confidence,
            'min_votes': self.min_votes,
            'processing_time': processing_time,
            'total_frames': frame_count,
            'fps': frame_count / processing_time,
            'total_detections': sum(class_counts.values()),
            'class_counts': dict(class_counts),
            'ensemble_stats': {
                'total_votes': ensemble_stats['total_votes'],
                'high_agreement_detections': ensemble_stats['high_agreement_detections'],
                'average_model_agreement': avg_agreement,
                'detection_confidence_boost': self.confidence_boost
            },
            'all_detections': all_detections,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results_summary

def main():
    """Main function"""
    print("🤖 Ensemble Object Detection System")
    print("=" * 50)
    
    # Check for video file
    video_file = "vid.mov"
    if not os.path.exists(video_file):
        print(f"❌ Video file '{video_file}' not found!")
        return
    
    # Check available models
    available_models = []
    model_candidates = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    for model in model_candidates:
        if os.path.exists(model):
            available_models.append(model)
    
    if not available_models:
        print("❌ No YOLO models found!")
        return
    
    print(f"📋 Available models: {available_models}")
    
    # Initialize ensemble detector
    detector = EnsembleDetector(model_paths=available_models[:2], base_confidence=0.4)
    
    # Process video
    print(f"\n🎯 Processing with ensemble detection...")
    results = detector.process_video_ensemble(video_file, "ensemble_output.mp4")
    
    if results:
        print("\n" + "=" * 60)
        print("📊 ENSEMBLE PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Processing time: {results['processing_time']:.2f} seconds")
        print(f"🎯 Total detections: {results['total_detections']:,}")
        print(f"📈 Processing FPS: {results['fps']:.1f}")
        print(f"🤖 Models used: {len(results['models_used'])}")
        
        # Show ensemble statistics
        ensemble_stats = results['ensemble_stats']
        print(f"\n📊 Ensemble Statistics:")
        print(f"   Total votes: {ensemble_stats['total_votes']:,}")
        print(f"   High agreement detections: {ensemble_stats['high_agreement_detections']:,}")
        print(f"   Average model agreement: {ensemble_stats['average_model_agreement']:.3f}")
        print(f"   Confidence boost: {ensemble_stats['detection_confidence_boost']}")
        
        # Show detection results
        if results['class_counts']:
            print(f"\n🏷️ Objects Detected (Ensemble):")
            for class_name, count in sorted(results['class_counts'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / results['total_detections']) * 100
                print(f"   {class_name:>15}: {count:>6,} ({percentage:>5.1f}%)")
        
        # Save results
        with open('ensemble_detection_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Output Files:")
        print(f"   Ensemble Video: ensemble_output.mp4")
        print(f"   Results Summary: ensemble_detection_results.json")
        
        print("\n💡 Ensemble Features Applied:")
        print("   ✅ Multiple model voting")
        print("   ✅ Detection merging")
        print("   ✅ Confidence boosting")
        print("   ✅ Model agreement tracking")
        print("   ✅ Class-specific thresholds")
        print("   ✅ False positive reduction")
        
    else:
        print(f"\n❌ Processing failed!")

if __name__ == "__main__":
    main()
