import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import requests
import os
import time
from datetime import datetime
import tempfile
from pathlib import Path
import threading
import io
import base64
import gc
import warnings

# Cloud deployment configuration  
def is_cloud_deployment():
    """Detect if running on cloud platform"""
    cloud_indicators = [
        'RENDER',
        'HEROKU', 
        'VERCEL',
        'NETLIFY',
        'Railway',
        'STREAMLIT_SHARING',
        'DYNO'  # Heroku
    ]
    return any(os.getenv(key) for key in cloud_indicators) or 'streamlit.io' in os.getenv('SERVER_NAME', '')

# Configure for cloud deployment
IS_CLOUD = is_cloud_deployment()

# Force cloud mode to ensure base64 encoding (set to True for guaranteed cloud compatibility)
force_cloud_mode = True

# Cloud-specific Streamlit configuration
if IS_CLOUD or force_cloud_mode:
    # Early detection for enhanced file handling
    import streamlit as st
    # Configuration will be handled by .streamlit/config.toml
    pass

# Utility functions for robust image handling
def convert_image_to_base64(image):
    """Convert numpy array or PIL image to base64 string for reliable display"""
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Ensure the image is in the correct format (0-255, uint8)
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            # Handle different channel orders
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB format
                    image = Image.fromarray(image)
                elif image.shape[2] == 4:
                    # RGBA format
                    image = Image.fromarray(image, 'RGBA')
            else:
                # Grayscale
                image = Image.fromarray(image, 'L')
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        # Use JPEG for smaller file sizes, PNG for transparency
        format_type = "PNG" if hasattr(image, 'mode') and 'A' in str(image.mode) else "JPEG"
        if format_type == "JPEG" and image.mode in ('RGBA', 'LA'):
            # Convert RGBA to RGB for JPEG
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = bg
        
        image.save(buffered, format=format_type, quality=95 if format_type == "JPEG" else None)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/{format_type.lower()};base64,{img_str}"
    except Exception as e:
        st.error(f"Error converting image to base64: {e}")
        return None

def optimize_image_for_display(image, max_width=800, max_height=600):
    """Optimize image size for display to reduce memory usage"""
    try:
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            if w > max_width or h > max_height:
                # Calculate new dimensions maintaining aspect ratio
                ratio = min(max_width/w, max_height/h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif isinstance(image, Image.Image):
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        st.warning(f"Could not optimize image: {e}")
        return image

def display_image_safe(image, caption="", use_container_width=True, **kwargs):
    """Safely display image, using base64 on cloud platforms to avoid MediaFileHandler issues"""
    try:
        # Optimize image first to reduce memory usage
        image = optimize_image_for_display(image)
        
        # On cloud platforms or when forced, use base64 encoding
        if IS_CLOUD or force_cloud_mode:
            base64_image = convert_image_to_base64(image)
            if base64_image:
                # Display using markdown with base64 data URI
                width_style = "width: 100%;" if use_container_width else ""
                st.markdown(
                    f'<div style="text-align: center;"><img src="{base64_image}" style="{width_style} max-width: 100%; height: auto;"><br><small>{caption}</small></div>',
                    unsafe_allow_html=True
                )
                return True
            else:
                st.error("Failed to process image for display")
                return False
        else:
            # On local development, try normal Streamlit display first
            try:
                st.image(image, caption=caption, use_container_width=use_container_width, **kwargs)
                return True
            except Exception as local_error:
                st.warning(f"Normal display failed, using base64 fallback: {local_error}")
                # Fallback to base64 even on local if normal display fails
                base64_image = convert_image_to_base64(image)
                if base64_image:
                    width_style = "width: 100%;" if use_container_width else ""
                    st.markdown(
                        f'<div style="text-align: center;"><img src="{base64_image}" style="{width_style} max-width: 100%; height: auto;"><br><small>{caption}</small></div>',
                        unsafe_allow_html=True
                    )
                    return True
                return False
    except Exception as e:
        st.error(f"Error displaying image: {e}")
        return False

def cleanup_memory():
    """Perform garbage collection to free memory"""
    try:
        gc.collect()
        # Clear some session state if it gets too large
        if hasattr(st.session_state, 'keys'):
            # Keep important session state, clear temporary data
            temp_keys = [k for k in st.session_state.keys() if k.startswith('temp_') or k.startswith('cache_')]
            for key in temp_keys:
                if key in st.session_state:
                    del st.session_state[key]
    except Exception:
        pass

def clear_upload_session():
    """Clear all upload-related session state to prevent conflicts"""
    try:
        # Clear all file uploader related session state
        upload_keys = [k for k in st.session_state.keys() if 'upload' in k.lower() or 'file' in k.lower()]
        for key in upload_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear any cached file data
        if hasattr(st, '_file_uploader_state_cache'):
            st._file_uploader_state_cache.clear()
            
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        # Silent fail - don't break the app if cleanup fails
        pass

def reset_file_uploader_widget(key):
    """Reset specific file uploader widget state"""
    try:
        # Clear the specific widget's session state
        widget_keys = [k for k in st.session_state.keys() if key in k]
        for widget_key in widget_keys:
            if widget_key in st.session_state:
                del st.session_state[widget_key]
                
        # Clear any retry counters for this widget
        retry_key = f"{key}_retry_count"
        if retry_key in st.session_state:
            st.session_state[retry_key] = 0
            
    except Exception:
        pass

def validate_uploaded_file(uploaded_file, max_size_mb=3):
    """Validate uploaded file size and type for cloud compatibility"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size (cloud platforms typically have smaller limits)
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large ({file_size_mb:.1f}MB). Maximum size allowed: {max_size_mb}MB"
    
    # Check file type
    allowed_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_types:
        return False, f"Invalid file type: {file_extension}. Allowed types: {', '.join(allowed_types)}"
    
    return True, "File is valid"

def safe_file_uploader(label, type=None, accept_multiple_files=False, key=None, help=None, 
                      on_change=None, args=None, kwargs=None, disabled=False, 
                      label_visibility="visible", max_size_mb=3):
    """Cloud-safe file uploader with enhanced error handling and retry logic"""
    
    # Set cloud-optimized default file types if not specified
    if type is None:
        type = ["jpg", "jpeg", "png"]
    
    # Create a unique key for retry attempts
    retry_key = f"{key}_retry_count" if key else "upload_retry_count"
    
    # Initialize retry counter in session state
    if retry_key not in st.session_state:
        st.session_state[retry_key] = 0
    
    # Proactive cleanup before first attempt to prevent session conflicts
    if st.session_state[retry_key] == 0:
        clear_upload_session()
        reset_file_uploader_widget(key)
    
    # Maximum retry attempts
    max_retries = 3
    
    try:
        # Show retry information if needed
        if st.session_state[retry_key] > 0:
            st.info(f"🔄 Upload attempt {st.session_state[retry_key] + 1} of {max_retries + 1}")
        
        # Use Streamlit's file uploader with cloud-optimized settings
        uploaded_file = st.file_uploader(
            label=label,
            type=type,
            accept_multiple_files=accept_multiple_files,
            key=f"{key}_{st.session_state[retry_key]}" if key else None,  # Unique key for each retry
            help=help,
            on_change=on_change,
            args=args,
            kwargs=kwargs,
            disabled=disabled,
            label_visibility=label_visibility
        )
        
        if uploaded_file is not None:
            # Validate the uploaded file
            is_valid, message = validate_uploaded_file(uploaded_file, max_size_mb)
            
            if not is_valid:
                st.error(f"Upload Error: {message}")
                return None
            
            # Reset retry counter on successful upload
            st.session_state[retry_key] = 0
            
            # Cleanup after successful upload to prevent future conflicts
            cleanup_memory()
            
            # Show success message for valid uploads
            if not accept_multiple_files:  # Single file
                st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({uploaded_file.size/1024:.1f}KB)")
            
            return uploaded_file
        
        return None
        
    except Exception as e:
        # Handle upload errors gracefully with retry logic
        error_msg = str(e).lower()
        
        if any(code in error_msg for code in ["400", "bad request", "network error", "timeout"]):
            st.session_state[retry_key] += 1
            
            if st.session_state[retry_key] <= max_retries:
                # Show retry option
                st.error(f"❌ Upload failed (attempt {st.session_state[retry_key]}): Network error or file too large.")
                st.warning("💡 **Possible solutions:**")
                st.markdown("- Try a **smaller image** (under 1MB)")
                st.markdown("- **Refresh the page** and try again")
                st.markdown("- **Compress your image** before uploading")
                
                # Auto-retry button
                if st.button("🔄 Retry Upload", key=f"retry_{key}_{st.session_state[retry_key]}"):
                    st.rerun()
                
                return None
            else:
                # Max retries reached - offer alternative solution
                st.error("❌ **Upload failed after multiple attempts.** This is a known issue with cloud deployments.")
                st.warning("🛠️ **Alternative Solution:** Try using a smaller image (under 1MB) or contact support.")
                
                # Reset retry counter
                st.session_state[retry_key] = 0
                return None
        
        elif "413" in error_msg:
            st.error("❌ Upload failed: File is too large. Please try a smaller file (under 1MB).")
        else:
            st.error(f"❌ Upload failed: {e}")
        
        return None

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
    "path": "logov1.png",  # Logo is now in main directory
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
        display_image_safe(Image.open(logo_config["path"]), caption="", use_container_width=False)
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

# SAHI AutoDetection Model
@st.cache_resource
def load_sahi_model(path):
    """Load SAHI AutoDetection model"""
    try:
        from ultralytics.utils.torch_utils import select_device
        return AutoDetectionModel.from_pretrained(
            model_type="ultralytics", 
            model_path=path, 
            device=select_device('')
        )
    except Exception as e:
        st.error(f"Error loading SAHI model: {e}")
        return None

def perform_sahi_inference(image_np, sahi_model, conf, slice_width, slice_height, overlap_ratio):
    """Perform SAHI inference on image"""
    try:
        # Convert BGR to RGB for SAHI
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Perform sliced prediction
        result = get_sliced_prediction(
            image_rgb,
            sahi_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMS",  # Use NMS for post-processing
            postprocess_match_threshold=0.5,
            postprocess_class_agnostic=False
        )
        
        return result
    except Exception as e:
        st.error(f"Error during SAHI inference: {e}")
        return None

def safe_extract_tensor_value(tensor_like_obj):
    """Safely extract value from tensor or non-tensor object"""
    if hasattr(tensor_like_obj, 'cpu'):
        return tensor_like_obj.cpu().numpy()
    else:
        return tensor_like_obj

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

# SAHI Configuration (only for images)
sahi_model = None
if source == "Image":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 SAHI Configuration")
    st.sidebar.markdown("*For better detection of small objects*")
    
    enable_sahi = st.sidebar.toggle("Enable SAHI", value=False, help="Use SAHI for sliced inference to detect small objects better")
    
    if enable_sahi:
        slice_width = st.sidebar.slider("Slice Width", 256, 1024, 512, 64, help="Width of each slice for SAHI inference")
        slice_height = st.sidebar.slider("Slice Height", 256, 1024, 512, 64, help="Height of each slice for SAHI inference")
        overlap_ratio = st.sidebar.slider("Overlap Ratio", 0.1, 0.9, 0.2, 0.1, help="Overlap ratio between slices")
        
        # Load SAHI model when enabled
        sahi_model = load_sahi_model(model_config[model_name]["path"])
        if sahi_model is None:
            st.warning("Failed to load SAHI model. Falling back to regular inference.")
            enable_sahi = False
    else:
        slice_width = 512
        slice_height = 512
        overlap_ratio = 0.2
else:
    enable_sahi = False
    slice_width = 512
    slice_height = 512
    overlap_ratio = 0.2

# Sidebar: Start button for Image source sets session state
if source == "Image":
    if st.sidebar.button("Start", key="start_image"):
        st.session_state["show_upload"] = True

# Sidebar: Start button at the end for other sources (optional, can be removed if not needed)
st.sidebar.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)  # Spacer

# Initialize clean session state on app start
if "app_initialized" not in st.session_state:
    clear_upload_session()
    cleanup_memory()
    st.session_state["app_initialized"] = True

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
    uploaded_file = safe_file_uploader(
        "Drag and drop your Image here or browse your computer",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="main_image_uploader",
        max_size_mb=3
    )
    
    # Check if we have an example image selected
    example_file = st.session_state.get("uploaded_example")
    
    # Determine which image to process (regular upload takes priority)
    if uploaded_file is not None:
        # Clear any example selection when user uploads a file
        if "uploaded_example" in st.session_state:
            del st.session_state["uploaded_example"]
        
        # Process regular upload - FIRST OCCURRENCE
        st.write(f"**{uploaded_file.name}**  {uploaded_file.size/1024:.1f}KB")
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            display_image_safe(image, caption="Original Image", use_container_width=True)
        
        # Inference with SAHI or regular YOLO
        if enable_sahi and sahi_model is not None:
            st.info("🔍 Using SAHI for enhanced small object detection...")
            sahi_result = perform_sahi_inference(image_np, sahi_model, conf, slice_width, slice_height, overlap_ratio)
            
            if sahi_result is not None:
                # Convert SAHI result to display format with bounding boxes
                import tempfile
                import os
                
                # Create temporary file to save the annotated image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                # Export the visualization with bounding boxes
                sahi_result.export_visuals(export_dir=os.path.dirname(temp_path), 
                                         file_name=os.path.basename(temp_path).replace('.png', ''),
                                         hide_labels=False,
                                         hide_conf=False)
                
                # Read the annotated image
                annotated_frame = cv2.imread(temp_path)
                if annotated_frame is not None:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to original image if visualization fails
                    annotated_frame = sahi_result.image
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                # Create a mock result object for consistency with the rest of the code
                import torch
                class MockResult:
                    def __init__(self, sahi_result, class_names):
                        self.boxes = []
                        self.names = {i: name for i, name in enumerate(class_names)}
                        
                        # Convert SAHI detections to boxes format
                        for detection in sahi_result.object_prediction_list:
                            box_info = type('Box', (), {})()
                            # Create tensor-like objects that have .cpu().numpy() methods
                            box_info.cls = [torch.tensor(detection.category.id)]
                            box_info.conf = [torch.tensor(detection.score.value)]
                            self.boxes.append(box_info)
                
                result = MockResult(sahi_result, class_names)
            else:
                st.warning("SAHI inference failed. Using regular inference...")
                results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
                result = results[0]
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
            result = results[0]
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            display_image_safe(annotated_frame, caption="Inference Result", use_container_width=True)
        
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
                cls = int(safe_extract_tensor_value(box.cls[0]))
                conf_score = float(safe_extract_tensor_value(box.conf[0]))
                class_name = result.names[cls]
                results_md += f"<tr><td style='padding:8px;'>{class_name}</td><td style='padding:8px;'>{conf_score*100:.2f}%</td></tr>"
            results_md += "</table></div>"
            st.markdown(results_md, unsafe_allow_html=True)
        else:
            st.info("No objects detected in this image.")
        
        # Add space before the clear button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Clear button to try another image
        if st.button("🔄 Clear and Try Another Image", key="clear_upload_1"):
            if "uploaded_example" in st.session_state:
                del st.session_state["uploaded_example"]
            # Comprehensive cleanup before next upload
            clear_upload_session()
            reset_file_uploader_widget("main_image_uploader")
            cleanup_memory()
            st.rerun()
    
    elif example_file is not None:
        # Process example image
        st.write(f"**{example_file['name']}** (Example Image)")
        image = example_file["image"]
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        with col1:
            display_image_safe(image, caption="Original Image", use_container_width=True)
        
        # Inference with SAHI or regular YOLO
        if enable_sahi and sahi_model is not None:
            st.info("🔍 Using SAHI for enhanced small object detection...")
            sahi_result = perform_sahi_inference(image_np, sahi_model, conf, slice_width, slice_height, overlap_ratio)
            
            if sahi_result is not None:
                # Convert SAHI result to display format with bounding boxes
                import tempfile
                import os
                
                # Create temporary file to save the annotated image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                
                # Export the visualization with bounding boxes
                sahi_result.export_visuals(export_dir=os.path.dirname(temp_path), 
                                         file_name=os.path.basename(temp_path).replace('.png', ''),
                                         hide_labels=False,
                                         hide_conf=False)
                
                # Read the annotated image
                annotated_frame = cv2.imread(temp_path)
                if annotated_frame is not None:
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                else:
                    # Fallback to original image if visualization fails
                    annotated_frame = sahi_result.image
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                # Create a mock result object for consistency with the rest of the code
                import torch
                class MockResult:
                    def __init__(self, sahi_result, class_names):
                        self.boxes = []
                        self.names = {i: name for i, name in enumerate(class_names)}
                        
                        # Convert SAHI detections to boxes format
                        for detection in sahi_result.object_prediction_list:
                            box_info = type('Box', (), {})()
                            # Create tensor-like objects that have .cpu().numpy() methods
                            box_info.cls = [torch.tensor(detection.category.id)]
                            box_info.conf = [torch.tensor(detection.score.value)]
                            self.boxes.append(box_info)
                
                result = MockResult(sahi_result, class_names)
            else:
                st.warning("SAHI inference failed. Using regular inference...")
                results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
                result = results[0]
                annotated_frame = result.plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        else:
            results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
            result = results[0]
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            display_image_safe(annotated_frame, caption="Inference Result", use_container_width=True)
        
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
                cls = int(safe_extract_tensor_value(box.cls[0]))
                conf_score = float(safe_extract_tensor_value(box.conf[0]))
                class_name = result.names[cls]
                results_md += f"<tr><td style='padding:8px;'>{class_name}</td><td style='padding:8px;'>{conf_score*100:.2f}%</td></tr>"
            results_md += "</table></div>"
            st.markdown(results_md, unsafe_allow_html=True)
        else:
            st.info("No objects detected in this image.")
        
        # Add space before the clear button
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        
        # Clear button to try another image
        if st.button("🔄 Clear and Try Another Image", key="clear_upload_2"):
            if "uploaded_example" in st.session_state:
                del st.session_state["uploaded_example"]
            # Comprehensive cleanup before next upload
            clear_upload_session()
            reset_file_uploader_widget("main_image_uploader")
            cleanup_memory()
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
                    # Clear any existing upload state first
                    clear_upload_session()
                    reset_file_uploader_widget("main_image_uploader")
                    
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
                    # Clear any existing upload state first
                    clear_upload_session()
                    reset_file_uploader_widget("main_image_uploader")
                    
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
