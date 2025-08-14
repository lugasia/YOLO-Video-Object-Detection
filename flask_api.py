from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import json
import os
from yolo_video_detector import YOLOVideoDetector

app = Flask(__name__)
CORS(app)

# Global variables for video processing
camera = None
detector = None
processing_thread = None
stop_processing = False
current_detections = []
frame_buffer = None

class VideoCamera:
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        return frame

def process_video_stream():
    """Background thread for processing video stream"""
    global camera, detector, current_detections, frame_buffer, stop_processing
    
    while not stop_processing:
        if camera is None:
            time.sleep(0.1)
            continue
            
        frame = camera.get_frame()
        if frame is None:
            continue
            
        # Process frame with YOLO
        annotated_frame, detections = detector.process_frame(frame)
        current_detections = detections
        
        # Store frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            frame_buffer = buffer.tobytes()
        
        time.sleep(0.03)  # ~30 FPS

def generate_frames():
    """Generate video frames for streaming"""
    global frame_buffer, stop_processing
    
    while not stop_processing:
        if frame_buffer is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        else:
            time.sleep(0.03)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera and detection"""
    global camera, detector, processing_thread, stop_processing
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id', 0)
        confidence = data.get('confidence', 0.5)
        
        # Stop existing processing
        stop_processing = True
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
        
        # Initialize camera and detector
        camera = VideoCamera(camera_id)
        detector = YOLOVideoDetector(confidence_threshold=confidence)
        
        # Start processing thread
        stop_processing = False
        processing_thread = threading.Thread(target=process_video_stream)
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({'status': 'success', 'message': 'Camera started successfully'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera and detection"""
    global camera, processing_thread, stop_processing
    
    try:
        stop_processing = True
        if processing_thread and processing_thread.is_alive():
            processing_thread.join()
        
        if camera:
            del camera
            camera = None
        
        return jsonify({'status': 'success', 'message': 'Camera stopped successfully'})
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/detections')
def get_detections():
    """Get current detections"""
    global current_detections
    
    return jsonify({
        'detections': current_detections,
        'count': len(current_detections),
        'timestamp': time.time()
    })

@app.route('/api/process_video', methods=['POST'])
def process_video():
    """Process uploaded video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
        
        video_file = request.files['video']
        confidence = float(request.form.get('confidence', 0.5))
        
        # Save uploaded file temporarily
        temp_path = f"temp_video_{int(time.time())}.mp4"
        video_file.save(temp_path)
        
        # Process video
        detector = YOLOVideoDetector(confidence_threshold=confidence)
        output_path = f"output_{int(time.time())}.mp4"
        detector.process_video_file(temp_path, output_path)
        
        # Get results
        summary = detector.get_detection_summary()
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'output_video': output_path
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/models')
def get_models():
    """Get available YOLO models"""
    models = [
        {'name': 'YOLOv8 Nano', 'path': 'yolov8n.pt', 'size': '6.7MB'},
        {'name': 'YOLOv8 Small', 'path': 'yolov8s.pt', 'size': '21.5MB'},
        {'name': 'YOLOv8 Medium', 'path': 'yolov8m.pt', 'size': '52.2MB'},
        {'name': 'YOLOv8 Large', 'path': 'yolov8l.pt', 'size': '87.7MB'},
        {'name': 'YOLOv8 XLarge', 'path': 'yolov8x.pt', 'size': '136.2MB'}
    ]
    return jsonify(models)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 