# LeukoDetect Application 🔬

A real-time leukemia cell detection application powered by YOLO11 and Streamlit.

## Features

- **Multi-source Detection**: Support for images, videos, and webcam feeds
- **Real-time Inference**: Live detection with confidence scores
- **Multiple Models**: Choose from different YOLO11 model variants
- **Automatic Downloads**: Models and assets are downloaded automatically from Dropbox
- **Interactive Results**: Visual results table with class names and confidence percentages
- **Tracking Support**: Object tracking for video and webcam sources

## Important Notes

⚠️ **Webcam Limitation**: Webcam functionality is only available when running the app locally. Streamlit Community Cloud doesn't have access to physical webcams.

**For webcam usage**: Download and run the app locally on your computer.

## Models

The application uses custom-trained YOLO11 models for leukemia cell detection:
- `yolo11l_v5_3`: Large model variant
- `yolo11x_v5_3`: Extra-large model variant  
- `yolo11x_v5_4`: Extra-large model variant (v4)

## Deployment

### Streamlit Community Cloud

1. Fork this repository
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Deploy the app by selecting this repository
4. The app will automatically install dependencies and download models on first run

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd leuko-detect
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Select Model**: Choose from available YOLO11 models
2. **Choose Source**: 
   - **Image**: Upload single images for analysis
   - **Video**: Upload video files for frame-by-frame analysis
   - **Webcam**: Real-time analysis (local deployment only)
3. **Configure Classes**: Expand the Classes section to select specific cell types
4. **Adjust Thresholds**: Set confidence and IoU thresholds
5. **Start Detection**: Click Start and upload your image/video or use webcam

## About

**LeukoDetect** was created by **Johan Diaz, MD**; **Arunima Deb, MD**; **Alexandra Lyubimova, DO**; **Cedric Nasnan, MD**

LeukoDetect was trained using the **LeukemiaAttri Dataset** by Rehman et al.

### Citation

Rehman A, Meraj T, Minhas AM, Imran A, Ali M, Sultani W. A large-scale multi-domain leukemia dataset for the white blood cells detection with morphological attributes for explainability. arXiv. Published May 17, 2024. doi:10.48550/arXiv.2405.10803

## License

This project is for educational and research purposes. 