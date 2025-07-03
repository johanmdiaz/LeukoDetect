import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import warnings
import os
import requests
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="LeukoDetect Application 🔬", layout="wide")

# Custom CSS for color scheme and button
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #ffffff;
}
[data-testid="stSidebar"] .stButton button {
    background-color: #FF3D3D;
    color: white;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #B22222;
}
.stButton button {
    background-color: #FF3D3D;
    color: white;
    width: 100%;
}
.stButton button:hover {
    background-color: #B22222;
}
</style>
""", unsafe_allow_html=True)

# Model configuration with Dropbox download links
model_config = {
    "yolo11l_v5_3": {
        "path": "models/yolo11l_v5_3_250_e.pt",
        "url": "https://www.dropbox.com/scl/fi/kfsbpfapbfoqbdsno5cr8/yolo11l_v5_3_250_e.pt?rlkey=pplolsevoxcx3ts9xzzwi7aqd&st=9ze18gmn&dl=1"
    },
    "yolo11x_v5_3": {
        "path": "models/yolo11x_v5_3_250.pt",
        "url": "https://www.dropbox.com/scl/fi/j8zywy4x6d3v151ke7zsx/yolo11x_v5_3_250.pt?rlkey=iix8397beki4nvnsccfttr7s3&st=wnbekxwr&dl=1"
    },
    "yolo11x_v5_4": {
        "path": "models/yolo11x_v5_4_250.pt",
        "url": "https://www.dropbox.com/scl/fi/uwqv435wxwdmf32d4qa12/yolo11x_v5_4_250.pt?rlkey=hfa84xgee2tlf8etrgijyxgt6&st=vyguyn2k&dl=1"
    }
}

# Logo configuration
logo_config = {
    "path": "assets/logov1.png",
    "url": "https://www.dropbox.com/scl/fi/5009uw9qyqusu1fqts6tu/logov1.png?rlkey=sfoxvloqldcnangk56d4hp4qi&st=kmoauecw&dl=1"
}

# Example images configuration
example_images_config = {
    "myeloblasts": {
        "path": "examples/myeloblasts.jpg",
        "url": "https://www.dropbox.com/scl/fi/0trtrtuftem2k0dmjswbz/Myeloblasts_on_peripheral_bloodsmear.jpg?rlkey=11hk7p4clzyz0vh5n360sbev4&st=fwivpczt&dl=1",
        "title": "Myeloblasts on Peripheral Blood Smear",
        "description": "Example showing myeloblasts in peripheral blood"
    },
    "neutrophils": {
        "path": "examples/neutrophils.jpg", 
        "url": "https://www.dropbox.com/scl/fi/xn3i8qjk2o6tccbrfh0w1/Neutrophils.jpg?rlkey=mpmahyuc5pi185ek7vogs0pni&st=4zol9vr6&dl=1",
        "title": "Neutrophils",
        "description": "Example showing neutrophils in blood smear"
    }
}

def download_file(url, filepath, file_type="file"):
    """Download file from URL with progress bar"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(filepath, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        if file_type == "model":
                            status_text.text(f"Downloading: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        else:
                            status_text.text(f"Downloading {file_type}...")
        
        progress_bar.empty()
        status_text.empty()
        return True
        
    except Exception as e:
        st.error(f"Error downloading {file_type}: {str(e)}")
        return False

def ensure_logo_exists():
    """Ensure logo exists locally, download if necessary"""
    filepath = logo_config["path"]
    
    if not os.path.exists(filepath):
        # Download silently without showing messages
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Download without progress bar or messages
            response = requests.get(logo_config["url"], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            return True
        except Exception:
            return False
    
    return True

def ensure_model_exists(model_name):
    """Ensure model exists locally, download if necessary"""
    config = model_config[model_name]
    filepath = config["path"]
    
    if not os.path.exists(filepath):
        st.warning(f"Model {model_name} not found locally. Downloading from Dropbox...")
        
        with st.spinner(f"Downloading {model_name}..."):
            success = download_file(config["url"], filepath, "model")
            
        if success:
            st.success(f"✅ Model {model_name} downloaded successfully!")
        else:
            st.error(f"❌ Failed to download model {model_name}")
            return False
    
    return True

def ensure_example_image_exists(image_name):
    """Ensure example image exists locally, download if necessary"""
    config = example_images_config[image_name]
    filepath = config["path"]
    
    if not os.path.exists(filepath):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Download without progress bar or messages
            response = requests.get(config["url"], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            return True
        except Exception:
            return False
    
    return True

# Sidebar: Logo (download if necessary)
with st.sidebar:
    if ensure_logo_exists():
        st.image(logo_config["path"], width=250)
    else:
        st.warning("Logo could not be loaded")
    st.title("User Configuration")

# Sidebar: Model selection
model_name = st.sidebar.selectbox("Model", list(model_config.keys()))

# Load model with better error handling
@st.cache_resource
def load_model(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Ensure model exists and load it
if ensure_model_exists(model_name):
    model = load_model(model_config[model_name]["path"])
    if model is None:
        st.stop()
else:
    st.stop()

# Add small space before title
st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

class_names = list(model.names.values())

# Sidebar: Class selection (more compact)
with st.sidebar.expander("Classes", expanded=False):
    select_all = st.checkbox("Select All Classes", value=True)
    
    # Always show the multiselect when expander is open
    if select_all:
        selected_classes = st.multiselect("All classes selected:", class_names, default=class_names, key="class_select", disabled=True)
        selected_inds = list(range(len(class_names)))
    else:
        selected_classes = st.multiselect("Choose specific classes:", class_names, default=class_names, key="class_select")
        selected_inds = [class_names.index(option) for option in selected_classes]

# Sidebar: Source selection
source = st.sidebar.selectbox("Source", ("Webcam", "Video", "Image"), index=2)

# Sidebar: Tracking and thresholds
if source in ("Webcam", "Video"):
    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No"))
else:
    enable_trk = "No"
conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

# Sidebar: Start button for Image source sets session state
if source == "Image":
    if st.sidebar.button("Start", key="start_image"):
        st.session_state["show_upload"] = True

# Sidebar: Start button at the end for other sources (optional, can be removed if not needed)
st.sidebar.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)  # Spacer

# Main title and subtitle
st.markdown("""
<div><h1 style="color:#00cfff; text-align:center; font-size:40px; margin-top:-50px; font-family: 'Archivo', sans-serif; margin-bottom:20px;">LeukoDetect Application 🔬 </h1></div>
""", unsafe_allow_html=True)
st.markdown("""
<div><h4 style="color:#155a5a; text-align:center; font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;"> Leukemia cells detection on your browser with the power of Ultralytic's YOLO11 🚀</h4></div>
""", unsafe_allow_html=True)

# Main area: Image upload and inference (only after Start is pressed)
if source == "Image" and st.session_state.get("show_upload", False):
    st.markdown("### Upload your Image")
    
    # Single file uploader that handles both regular uploads and example images
    uploaded_file = st.file_uploader(
        "Drag and drop your Image here or browse your computer",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="main_image_uploader"
    )
    
    # Check if we have an example image selected
    example_file = st.session_state.get("uploaded_example")
    
    # Determine which image to process (regular upload takes priority)
    if uploaded_file is not None:
        # Clear any example selection when user uploads a file
        if "uploaded_example" in st.session_state:
            del st.session_state["uploaded_example"]
        
        # Process regular upload
        st.write(f"**{uploaded_file.name}**  {uploaded_file.size/1024:.1f}KB")
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Inference
        results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
        result = results[0]
        annotated_frame = result.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(annotated_frame, caption="Inference Result", use_container_width=True)
        
        # Results table
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            results_md = """
            <div style='margin-top:30px;'>
            <h4 style='color:#155a5a;'>Inference Results</h4>
            <table style='width:60%;margin:auto;border-collapse:collapse;'>
                <tr style='background-color:#f2f2f2;'>
                    <th style='padding:8px;text-align:left;'>Class</th>
                    <th style='padding:8px;text-align:left;'>Confidence</th>
                </tr>
            """
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                class_name = result.names[cls]
                results_md += f"<tr><td style='padding:8px;'>{class_name}</td><td style='padding:8px;'>{conf_score*100:.2f}%</td></tr>"
            results_md += "</table></div>"
            st.markdown(results_md, unsafe_allow_html=True)
        else:
            st.info("No objects detected in this image.")
        
        # Add space before the clear button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Clear button to try another image
        if st.button("🔄 Clear and Try Another Image", key="clear_example"):
            if "uploaded_example" in st.session_state:
                del st.session_state["uploaded_example"]
            st.rerun()
    
    elif example_file is not None:
        # Process example image
        st.write(f"**{example_file['name']}** (Example Image)")
        image = example_file["image"]
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        # Inference
        results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
        result = results[0]
        annotated_frame = result.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.image(annotated_frame, caption="Inference Result", use_container_width=True)
        
        # Results table
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            results_md = """
            <div style='margin-top:30px;'>
            <h4 style='color:#155a5a;'>Inference Results</h4>
            <table style='width:60%;margin:auto;border-collapse:collapse;'>
                <tr style='background-color:#f2f2f2;'>
                    <th style='padding:8px;text-align:left;'>Class</th>
                    <th style='padding:8px;text-align:left;'>Confidence</th>
                </tr>
            """
            for box in result.boxes:
                cls = int(box.cls[0].cpu().numpy())
                conf_score = float(box.conf[0].cpu().numpy())
                class_name = result.names[cls]
                results_md += f"<tr><td style='padding:8px;'>{class_name}</td><td style='padding:8px;'>{conf_score*100:.2f}%</td></tr>"
            results_md += "</table></div>"
            st.markdown(results_md, unsafe_allow_html=True)
        else:
            st.info("No objects detected in this image.")
        
        # Add space before the clear button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Clear button to try another image
        if st.button("🔄 Clear and Try Another Image", key="clear_example"):
            if "uploaded_example" in st.session_state:
                del st.session_state["uploaded_example"]
            st.rerun()
    
    # Example Images Section - Clickable Thumbnails
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🔬 Try Example Images")
    st.markdown("Click on any example image below to load it for analysis:")
    
    # Create columns for example thumbnails - more compact and centered layout
    col1, col2, col3, col4, col5 = st.columns([0.5, 2, 1, 2, 0.5])
    
    # Myeloblasts example
    with col2:
        if ensure_example_image_exists("myeloblasts"):
            example_config = example_images_config["myeloblasts"]
            example_image = Image.open(example_config["path"])
            
            # Display title - centered
            st.markdown(f"<p style='text-align: center; font-size: 11px; font-weight: bold; color: #155a5a; margin-bottom: 2px; margin-top: 0px;'>{example_config['title']}</p>", unsafe_allow_html=True)
            
            # Center the image
            col_img1, col_img2, col_img3 = st.columns([0.5, 1, 0.5])
            with col_img2:
                st.image(example_image, width=160)
            
            # Compact button with reduced spacing and smaller size
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
            
            # Create a centered container for smaller button
            col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
            with col_btn2:
                if st.button("Load Image", key="click_myeloblasts", type="secondary"):
                    # Store the example image in session state to simulate file upload
                    st.session_state["uploaded_example"] = {
                        "image": example_image,
                        "name": f"{example_config['title']}.jpg",
                        "size": len(example_image.tobytes())
                    }
                    st.rerun()
    
    # Neutrophils example  
    with col4:
        if ensure_example_image_exists("neutrophils"):
            example_config = example_images_config["neutrophils"]
            example_image = Image.open(example_config["path"])
            
            # Display title - centered
            st.markdown(f"<p style='text-align: center; font-size: 11px; font-weight: bold; color: #155a5a; margin-bottom: 2px; margin-top: 0px;'>{example_config['title']}</p>", unsafe_allow_html=True)
            
            # Center the image
            col_img1, col_img2, col_img3 = st.columns([0.5, 1, 0.5])
            with col_img2:
                st.image(example_image, width=160)
            
            # Compact button with reduced spacing and smaller size
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
            
            # Create a centered container for smaller button
            col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
            with col_btn2:
                if st.button("Load Image", key="click_neutrophils", type="secondary"):
                    # Store the example image in session state to simulate file upload
                    st.session_state["uploaded_example"] = {
                        "image": example_image,
                        "name": f"{example_config['title']}.jpg", 
                        "size": len(example_image.tobytes())
                    }
                    st.rerun()

# Handle Video and Webcam sources
elif source == "Video":
    # Video file uploader in sidebar
    vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"], key="video_uploader")
    
    if vid_file is not None and st.sidebar.button("Start Video", key="start_video"):
        # Save uploaded video to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(vid_file.read())
            video_path = tmp_file.name
        
        # Create columns for video display
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Could not open video file.")
        else:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Process frame with model
                if enable_trk == "Yes":
                    results = model.track(
                        frame, conf=conf, iou=iou, classes=selected_inds, persist=True
                    )
                else:
                    results = model(frame, conf=conf, iou=iou, classes=selected_inds)
                
                # Get the first result
                result = results[0]
                annotated_frame = result.plot()
                
                # Display frames
                org_frame.image(frame, channels="BGR", caption="Original Video")
                ann_frame.image(annotated_frame, channels="BGR", caption="Inference Result")
            
            cap.release()
            # Clean up temporary file
            os.unlink(video_path)

elif source == "Webcam":
    # Check if running on Streamlit Cloud (no webcam available)
    if 'STREAMLIT_CLOUD' in os.environ or not os.path.exists('/dev/video0'):
        st.warning("⚠️ Webcam is not available on Streamlit Cloud")
        st.info("""
        **Alternative solutions:**
        1. **Use Image Upload**: Switch to 'Image' source for single image analysis
        2. **Use Video Upload**: Switch to 'Video' source to upload and analyze video files
        3. **Run Locally**: Download and run this app locally to access your webcam
        
        **To run locally:**
        ```bash
        git clone <repository-url>
        cd leukodetect
        pip install -r requirements.txt
        streamlit run app.py
        ```
        """)
    else:
        if st.sidebar.button("Start Webcam", key="start_webcam"):
            # Create columns for webcam display
            col1, col2 = st.columns(2)
            org_frame = col1.empty()
            ann_frame = col2.empty()
            
            # Create stop button
            stop_button = st.button("Stop Webcam")
            
            # Start webcam capture with error handling
            cap = cv2.VideoCapture(0)  # Use webcam index 0
            
            if not cap.isOpened():
                st.error("Could not open webcam. Please verify the webcam is connected properly.")
            else:
                try:
                    while cap.isOpened() and not stop_button:
                        success, frame = cap.read()
                        if not success:
                            st.warning("Failed to read frame from webcam.")
                            break
                        
                        # Process frame with model
                        if enable_trk == "Yes":
                            results = model.track(
                                frame, conf=conf, iou=iou, classes=selected_inds, persist=True
                            )
                        else:
                            results = model(frame, conf=conf, iou=iou, classes=selected_inds)
                        
                        # Get the first result
                        result = results[0]
                        annotated_frame = result.plot()
                        
                        # Display frames
                        org_frame.image(frame, channels="BGR", caption="Original Webcam")
                        ann_frame.image(annotated_frame, channels="BGR", caption="Inference Result")
                        
                        # Check for stop button
                        if stop_button:
                            break
                except Exception as e:
                    st.error(f"Webcam error: {str(e)}")
                finally:
                    cap.release()
                    cv2.destroyAllWindows()

# Footer with information about LeukoDetect
st.markdown("---")
st.markdown("""
<div style="text-align: left; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 50px;">
    <h4 style="color: #155a5a; margin-bottom: 15px;">About LeukoDetect</h4>
    <p style="color: #666; margin-bottom: 10px; font-size: 14px;">
        <strong>LeukoDetect</strong> was created by <strong>Johan Diaz, MD</strong>; <strong>Arunima Deb, MD</strong>; <strong>Alexandra Lyubimova, DO</strong>; <strong>Cedric Nasnan, MD</strong>
    </p>
    <p style="color: #666; margin-bottom: 10px; font-size: 14px;">
        LeukoDetect was trained using the <strong>LeukemiaAttri Dataset</strong> by Rehman et al.
    </p>
    <p style="color: #888; font-size: 12px; font-style: italic;">
        1. Rehman A, Meraj T, Minhas AM, Imran A, Ali M, Sultani W. A large-scale multi-domain leukemia dataset for the white blood cells detection with morphological attributes for explainability. arXiv. Published May 17, 2024. doi:10.48550/arXiv.2405.10803
    </p>
</div>
""", unsafe_allow_html=True) 
