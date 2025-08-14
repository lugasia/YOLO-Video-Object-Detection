import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from typing import List, Tuple, Dict
import json

class YOLOVideoDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO video detector
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        
        # Color palette for different classes
        self.colors = self._generate_colors(len(self.class_names))
        
        # Detection history for tracking
        self.detection_history = []
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame and return annotated frame with detections
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (annotated_frame, detections_list)
        """
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names[class_id]
                    
                    # Get color for this class
                    color = self.colors[class_id]
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Create label
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Calculate label position
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    label_y = y1 - 10 if y1 - 25 > 0 else y1 + 25
                    
                    # Draw label background
                    cv2.rectangle(annotated_frame, 
                                (x1, label_y - label_size[1] - 10),
                                (x1 + label_size[0], label_y + 5),
                                color, -1)
                    
                    # Draw label text
                    cv2.putText(annotated_frame, label, (x1, label_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Store detection info
                    detection_info = {
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'color': color
                    }
                    detections.append(detection_info)
        
        # Add detection count to frame
        detection_count = len(detections)
        cv2.putText(annotated_frame, f"Detections: {detection_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, detections
    
    def process_video_file(self, video_path: str, output_path: str = None) -> None:
        """
        Process a video file and save the annotated version
        
        Args:
            video_path: Path to input video file
            output_path: Path to save annotated video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = self.process_frame(frame)
            
            # Store detection history
            self.detection_history.append({
                'frame': frame_count,
                'detections': detections,
                'timestamp': time.time() - start_time
            })
            
            # Write frame if output is specified
            if writer:
                writer.write(annotated_frame)
            
            # Display progress
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Display frame (optional - for debugging)
            # cv2.imshow('YOLO Detection', annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processing completed! Processed {frame_count} frames")
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
    
    def process_webcam(self, camera_id: int = 0) -> None:
        """
        Process real-time webcam feed
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            annotated_frame, detections = self.process_frame(frame)
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated_frame, timestamp, (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('YOLO Real-time Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"detection_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def get_detection_summary(self) -> Dict:
        """Get summary of all detections"""
        if not self.detection_history:
            return {}
        
        class_counts = {}
        total_detections = 0
        
        for frame_data in self.detection_history:
            for detection in frame_data['detections']:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_detections += 1
        
        return {
            'total_frames': len(self.detection_history),
            'total_detections': total_detections,
            'class_counts': class_counts,
            'processing_time': self.detection_history[-1]['timestamp'] if self.detection_history else 0
        }
    
    def save_detection_results(self, output_path: str) -> None:
        """Save detection results to JSON file"""
        results = {
            'detection_history': self.detection_history,
            'summary': self.get_detection_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detection results saved to {output_path}")

def main():
    """Main function to demonstrate usage"""
    # Initialize detector
    detector = YOLOVideoDetector(confidence_threshold=0.5)
    
    # Process the video file in the workspace
    video_file = "Screen Recording 2025-08-12 at 21.56.27.mov"
    
    if os.path.exists(video_file):
        print(f"Processing video file: {video_file}")
        output_video = "output_detected_video.mp4"
        detector.process_video_file(video_file, output_video)
        
        # Save detection results
        detector.save_detection_results("detection_results.json")
        
        # Print summary
        summary = detector.get_detection_summary()
        print("\nDetection Summary:")
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Total detections: {summary['total_detections']}")
        print("Detections by class:")
        for class_name, count in summary['class_counts'].items():
            print(f"  {class_name}: {count}")
    else:
        print(f"Video file {video_file} not found. Starting webcam mode...")
        detector.process_webcam()

if __name__ == "__main__":
    main() 