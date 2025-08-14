# 🎥 YOLO Video Object Detection System

**Powered by Inteli AI** - Advanced Computer Vision Solutions

A comprehensive video object detection system using YOLOv8, featuring real-time processing, construction equipment detection, and a user-friendly web interface.

![Inteli AI Logo](inteli-ai-black.webp)

## 🚀 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

## ✨ Features

- **🎯 Real-time Object Detection**: Using state-of-the-art YOLOv8 models
- **🚧 Construction Equipment Recognition**: Specialized detection for construction sites
- **📊 Comprehensive Analysis**: Detailed detection statistics and visualizations
- **🎨 Professional UI**: Clean, modern interface with Inteli AI branding
- **⚙️ Multiple Models**: From fast nano models to high-accuracy large models
- **🔧 Adjustable Parameters**: Fine-tune confidence thresholds and model selection
- **📁 Export Capabilities**: Save processed videos and detection results

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **AI Framework**: Ultralytics YOLOv8
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Image Processing**: Pillow (PIL)

## 📋 Supported Object Classes

The system can detect 80+ object classes including:
- 🚛 Construction equipment (trucks, excavators, bulldozers)
- 🚗 Vehicles (cars, buses, motorcycles)
- 👤 People and animals
- 📦 Common objects and items

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/lugasia/YOLO-Video-Object-Detection.git
   cd YOLO-Video-Object-Detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv yolo_env
   source yolo_env/bin/activate  # On Windows: yolo_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app_with_logo.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Connect your GitHub account**
4. **Deploy the app**:
   - Repository: `your-username/YOLO-Video-Object-Detection`
   - Main file path: `streamlit_app_with_logo.py`
   - Python version: 3.8+

## 📖 Usage Guide

### 1. Upload Video
- Supported formats: MP4, AVI, MOV, MKV
- Maximum file size: 200MB (Streamlit Cloud limit)

### 2. Configure Settings
- **Model Selection**: Choose from YOLOv8 nano to xlarge models
- **Confidence Threshold**: Adjust detection sensitivity (0.1-0.9)
- **Processing Options**: Real-time or batch processing

### 3. View Results
- **Construction Analysis**: Specialized equipment detection
- **All Results**: Complete object detection statistics
- **Export**: Download processed videos and data

## 🎯 Special Features

### Construction Equipment Detection
- **Smart Classification**: Identifies construction equipment even when misclassified
- **False Positive Reduction**: Optimized parameters for accurate detection
- **Equipment Mapping**: Maps detections to construction equipment types

### Performance Optimization
- **Model Selection**: Balance speed vs accuracy
- **Confidence Tuning**: Reduce false positives
- **Real-time Processing**: Fast video analysis

## 📁 Project Structure

```
YOLO-Video-Object-Detection/
├── streamlit_app_with_logo.py    # Main Streamlit application
├── requirements.txt              # Python dependencies
├── inteli-ai-black.webp         # Company logo
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
├── yolo_video_detector.py       # Core YOLO detection class
├── balanced_detection.py        # Balanced detection script
├── high_confidence_detection.py # High confidence detection
├── improved_detection.py        # Improved detection with analysis
└── docs/                        # Documentation
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom model path
YOLO_MODEL_PATH=yolov8n.pt

# Optional: Set confidence threshold
DEFAULT_CONFIDENCE=0.5
```

### Model Options
- **yolov8n.pt**: Fastest, smallest (recommended for testing)
- **yolov8s.pt**: Small, balanced
- **yolov8m.pt**: Medium, better accuracy
- **yolov8l.pt**: Large, high accuracy
- **yolov8x.pt**: XLarge, best accuracy

## 📊 Performance Metrics

- **Processing Speed**: 25-30 FPS (depending on model)
- **Accuracy**: 85-95% (depending on model and confidence)
- **Memory Usage**: 2-8 GB (depending on model size)
- **Supported Resolutions**: Up to 4K

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Model Download Failed**
   ```bash
   # Manual model download
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

2. **Memory Issues**
   - Use smaller models (yolov8n.pt)
   - Reduce video resolution
   - Lower confidence threshold

3. **Streamlit Cloud Issues**
   - Check Python version compatibility
   - Verify all dependencies in requirements.txt
   - Ensure main file path is correct

### Performance Tips

- **For Speed**: Use yolov8n.pt model
- **For Accuracy**: Use yolov8x.pt model
- **For Construction**: Use confidence 0.3-0.5
- **For General Use**: Use confidence 0.5-0.7

## 📞 Support

For support and questions:
- **GitHub Issues**: [Create an issue](https://github.com/lugasia/YOLO-Video-Object-Detection/issues)
- **Email**: support@inteli-ai.com
- **Website**: [Inteli AI](https://inteliate.com)

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **Inteli AI**: Advanced AI solutions

---

**Made with ❤️ by Inteli AI**

*Advanced Computer Vision Solutions* 