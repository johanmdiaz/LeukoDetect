# LeukoDetect Application 🔬

A real-time leukemia cell detection application powered by YOLO11 and Streamlit. Analyze blood smear images to detect various types of leukemia cells with high accuracy using state-of-the-art machine learning models.


## ✨ Features

- **🖼️ Multi-source Detection**: Support for images, videos, and webcam feeds
- **📸 Smart Camera Integration**: 
  - Real-time webcam analysis (local deployment)
  - Single photo capture (cloud deployment)
- **🎯 Real-time Inference**: Live detection with confidence scores and bounding boxes
- **🤖 Multiple AI Models**: Choose from different YOLO11 model variants
- **☁️ Cloud-Ready**: Optimized for both local and cloud deployments
- **📊 Interactive Results**: Visual results table with class names and confidence percentages
- **🎭 Object Tracking**: Advanced tracking for video and webcam sources
- **🔬 Example Images**: Built-in example blood smear images for testing
- **⚡ Performance Optimized**: Frame rate control and efficient processing

## 🎥 Webcam & Camera Features

### Local Deployment
- **Real-time Analysis**: Live webcam feed with instant detection
- **FPS Control**: Adjustable frame rates (5-30 FPS)
- **Live Metrics**: Real-time FPS counter and detection count
- **Smart Controls**: Start/stop webcam with session state management

### Cloud Deployment  
- **Single Photo Capture**: Use your device's camera to take photos
- **Instant Analysis**: Immediate processing of captured images
- **Full Results**: Complete detection results with confidence scores

## 🤖 AI Models

The application uses custom-trained YOLO11 models specifically designed for leukemia cell detection:

| Model | Description | Use Case |
|-------|-------------|----------|
| `yolo11l_v5_3` | Large model variant |  |
| `yolo11x_v5_3` | Extra-large model (v3) |  |
| `yolo11x_v5_4` | Extra-large model (v4) |  |

All models are automatically downloaded on first use and cached locally for faster subsequent runs.

## 🚀 Quick Start

### Try Online (Recommended)
Visit the live demo: [LeukoDetect on Streamlit Cloud](https://leukodetect.onrender.com) 

### Local Installation

1. **Clone the repository:**
```bash
git clone https://github.com/johanmdiaz/LeukoDetect.git
cd LeukoDetect
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Select Your Model**
Choose from the available YOLO11 models in the sidebar. Larger models provide higher accuracy but require more processing time.

### 2. **Choose Input Source**
- **📸 Image**: Upload single blood smear images (JPG, PNG)
- **🎬 Video**: Upload video files for frame-by-frame analysis  
- **🎥 Webcam**: Real-time analysis (local) or single photo capture (cloud)

### 3. **Try Example Images**
Click on the provided example images to test the application:
- **Myeloblasts**: Example showing myeloblasts in peripheral blood
- **Neutrophils**: Example showing neutrophils in blood smear

### 4. **Configure Detection**
- **Classes**: Select specific cell types to detect
- **Confidence**: Set minimum confidence threshold (0.0-1.0)
- **IoU**: Adjust intersection over union for overlapping detections
- **Tracking**: Enable object tracking for videos and webcam

### 5. **Analyze Results**
- View original and annotated images side by side
- Check the results table for detected classes and confidence scores
- Download or save results as needed

## 🌐 Deployment Options

### Streamlit Community Cloud
Perfect for sharing and demonstrations:

1. Fork this repository on GitHub
2. Connect to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Deploy by selecting your forked repository
4. Models and assets download automatically on first run

### Render
For more robust hosting:

1. Connect your GitHub repository to Render
2. Use the included configuration files
3. Deploy as a web service

### Local Development
Best for development and full webcam functionality:
- Full real-time webcam support
- Faster model loading (cached locally)
- No upload size limitations

## 🔬 Example Images

The application includes built-in example images for testing:

| Image | Description | Source |
|-------|-------------|---------|
| Myeloblasts | Myeloblasts on peripheral blood smear | https://commons.wikimedia.org/wiki/File:Myeloblasts_on_peripheral_bloodsmear.jpg |
| Neutrophils | Neutrophils in blood smear | https://commons.wikimedia.org/wiki/File:Neutrophils.jpg |

These images are automatically available in both local and cloud deployments.

## ⚙️ Technical Details

### Architecture
- **Frontend**: Streamlit for interactive web interface
- **Backend**: YOLO11 (Ultralytics) for object detection
- **Image Processing**: OpenCV and PIL for image manipulation
- **Cloud Storage**: GitHub for asset hosting and version control

### Performance Optimizations
- **Base64 Encoding**: Optimized image display for cloud deployments
- **Memory Management**: Automatic cleanup and garbage collection
- **Frame Skipping**: Intelligent frame rate control for webcam
- **Caching**: Model and asset caching for faster loading

### Browser Compatibility
- Chrome, Firefox, Safari, Edge
- Mobile browsers supported for photo capture
- Camera permissions required for webcam/photo features

## 🩺 About the Project

**LeukoDetect** was developed by a team of medical professionals:

- **Johan Diaz, MD** - Lead Developer
- **Arunima Deb, MD** -  
- **Alexandra Lyubimova, DO** - 
- **Cedric Nasnan, MD** - 

### Training Dataset

LeukoDetect models were trained using the **LeukemiaAttri Dataset** by Rehman et al., a comprehensive dataset designed for white blood cell detection with morphological attributes.

**Citation:**
```
Rehman A, Meraj T, Minhas AM, Imran A, Ali M, Sultani W. 
A large-scale multi-domain leukemia dataset for the white blood cells detection 
with morphological attributes for explainability. 
arXiv. Published May 17, 2024. doi:10.48550/arXiv.2405.10803
```

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space for models
- **Internet**: Required for initial model download
- **Browser**: Modern browser with JavaScript enabled

### For Webcam Features
- **Camera**: Built-in or USB webcam
- **Permissions**: Camera access permissions
- **OS**: Windows, macOS, or Linux

## 🤝 Contributing

We welcome contributions! 


## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/johanmdiaz/LeukoDetect/issues)
- **Documentation**: See this README and in-app help
- **Discussions**: [GitHub Discussions](https://github.com/johanmdiaz/LeukoDetect/discussions)

## ⚖️ License

This project is for educational and research purposes. Please ensure compliance with your local regulations when using medical AI tools.

## 🙏 Acknowledgments

- Ultralytics team for the excellent YOLO11 framework
- Streamlit team for the amazing web app framework  
- The research community for providing the LeukemiaAttri dataset
- All contributors and testers who helped improve this application

---

**⚠️ Medical Disclaimer**: This application is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions. 
