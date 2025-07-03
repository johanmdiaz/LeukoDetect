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

# Fix Ultralytics config directory warning for cloud deployments
if not os.environ.get('YOLO_CONFIG_DIR'):
    # Set to a writable directory (temp dir or current working directory)
    config_dir = os.path.join(tempfile.gettempdir(), 'ultralytics_config')
    os.makedirs(config_dir, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = config_dir

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
        
        # Safety check: Ensure image isn't too large before base64 conversion
        # Prevent memory issues that could cause 400 errors
        if hasattr(image, 'size'):
            width, height = image.size
            pixel_count = width * height
            # Limit to ~2MP to prevent memory issues
            if pixel_count > 2_000_000:
                st.warning(f"Large image detected ({width}x{height}). Optimizing for display...")
                # Resize while maintaining aspect ratio
                max_dimension = 1400  # Conservative limit
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        # Use JPEG for smaller file sizes, PNG for transparency
        format_type = "PNG" if hasattr(image, 'mode') and 'A' in str(image.mode) else "JPEG"
        if format_type == "JPEG" and image.mode in ('RGBA', 'LA'):
            # Convert RGBA to RGB for JPEG
            bg = Image.new('RGB', image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = bg
        
        # Save with quality settings to prevent large base64 strings
        save_kwargs = {"format": format_type}
        if format_type == "JPEG":
            save_kwargs["quality"] = 85  # Good quality while keeping size manageable
            save_kwargs["optimize"] = True
        
        image.save(buffered, **save_kwargs)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Final safety check on base64 string size
        if len(img_str) > 10_000_000:  # ~10MB base64 limit
            st.warning("Image too large after processing. Try uploading a smaller image.")
            return None
            
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

def display_image_safe(image, caption="", use_container_width=True, width=None, **kwargs):
    """Safely display image, using base64 on cloud platforms to avoid MediaFileHandler issues"""
    try:
        # Optimize image first to reduce memory usage
        image = optimize_image_for_display(image)
        
        # On cloud platforms or when forced, use base64 encoding
        if IS_CLOUD or force_cloud_mode:
            base64_image = convert_image_to_base64(image)
            if base64_image:
                # Display using markdown with base64 data URI
                if width is not None:
                    width_style = f"width: {width}px;"
                elif use_container_width:
                    width_style = "width: 100%;"
                else:
                    width_style = "width: 160px;"  # Default for thumbnails
                
                caption_html = f"<br><small>{caption}</small>" if caption else ""
                st.markdown(
                    f'<div style="text-align: center;"><img src="{base64_image}" style="{width_style} max-width: 100%; height: auto;">{caption_html}</div>',
                    unsafe_allow_html=True
                )
                return True
            else:
                st.error("Failed to process image for display")
                return False
        else:
            # On local development, try normal Streamlit display first
            try:
                # Force base64 for all deployments to avoid MediaFileHandler issues
                base64_image = convert_image_to_base64(image)
                if base64_image:
                    if width is not None:
                        width_style = f"width: {width}px;"
                    elif use_container_width:
                        width_style = "width: 100%;"
                    else:
                        width_style = "width: 160px;"  # Default for thumbnails
                    
                    caption_html = f"<br><small>{caption}</small>" if caption else ""
                    st.markdown(
                        f'<div style="text-align: center;"><img src="{base64_image}" style="{width_style} max-width: 100%; height: auto;">{caption_html}</div>',
                        unsafe_allow_html=True
                    )
                    return True
                else:
                    # Last resort fallback to regular st.image
                    st.image(image, caption=caption, use_container_width=use_container_width, **kwargs)
                    return True
            except Exception as local_error:
                st.error(f"Image display failed: {local_error}")
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
        # Clear all file uploader related session state, but preserve important UI state
        upload_keys = [k for k in st.session_state.keys() if 'upload' in k.lower() or 'file' in k.lower()]
        # Preserve important session state keys
        preserve_keys = ['show_upload', 'uploaded_example']
        for key in upload_keys:
            if key in st.session_state and key not in preserve_keys:
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
    
    # Check file type - MUST match the uploader's accepted types exactly
    allowed_types = ['jpg', 'jpeg', 'png']  # Match file_uploader type parameter
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_types:
        return False, f"Invalid file type: {file_extension}. Allowed types: {', '.join(allowed_types)}"
    
    # Additional validation: Check MIME type if available
    if hasattr(uploaded_file, 'type') and uploaded_file.type:
        allowed_mime_types = ['image/jpeg', 'image/jpg', 'image/png']
        if not any(mime_type in uploaded_file.type.lower() for mime_type in allowed_mime_types):
            return False, f"Invalid file format. Expected image file, got: {uploaded_file.type}"
    
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
    
    # Initialize retry counter in session state - with better error handling
    try:
        if retry_key not in st.session_state:
            st.session_state[retry_key] = 0
    except Exception:
        # Fallback initialization if session state access fails
        st.session_state[retry_key] = 0
    
    # Only cleanup on retry attempts, not first attempt
    if st.session_state.get(retry_key, 0) > 0:
        clear_upload_session()
        reset_file_uploader_widget(key)
    
    # Maximum retry attempts
    max_retries = 3
    
    try:
        # Show retry information if needed
        retry_count = st.session_state.get(retry_key, 0)
        if retry_count > 0:
            st.info(f"🔄 Upload attempt {retry_count + 1} of {max_retries + 1}")
        
        # Use Streamlit's file uploader with cloud-optimized settings
        uploaded_file = st.file_uploader(
            label=label,
            type=type,
            accept_multiple_files=accept_multiple_files,
            key=f"{key}_{retry_count}" if key else None,  # Unique key for each retry
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
            current_retry_count = st.session_state.get(retry_key, 0)
            st.session_state[retry_key] = current_retry_count + 1
            
            if st.session_state[retry_key] <= max_retries:
                # Show retry option
                current_retry_count = st.session_state.get(retry_key, 0)
                st.error(f"❌ Upload failed (attempt {current_retry_count}): Network error or file too large.")
                st.warning("💡 **Possible solutions:**")
                st.markdown("- Try a **smaller image** (under 1MB)")
                st.markdown("- **Refresh the page** and try again")
                st.markdown("- **Compress your image** before uploading")
                
                # Auto-retry button
                if st.button("🔄 Retry Upload", key=f"retry_{key}_{current_retry_count}"):
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

def safe_extract_tensor_value(tensor_like_obj):
    """Safely extract value from tensor or non-tensor object"""
    if hasattr(tensor_like_obj, 'cpu'):
        return tensor_like_obj.cpu().numpy()
    else:
        return tensor_like_obj

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
    "url": "https://raw.githubusercontent.com/johanmdiaz/LeukoDetect/main/logov1.png"
}

# Example images configuration
example_images_config = {
    "myeloblasts": {
        "path": "examples/myeloblasts.jpg",
        "url": "https://raw.githubusercontent.com/johanmdiaz/LeukoDetect/main/Myeloblasts_on_peripheral_bloodsmear.jpg",
        "title": "Myeloblasts on Peripheral Blood Smear",
        "description": "Example showing myeloblasts in peripheral blood"
    },
    "neutrophils": {
        "path": "examples/neutrophils.jpg", 
        "url": "https://raw.githubusercontent.com/johanmdiaz/LeukoDetect/main/Neutrophils.jpg",
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
        st.sidebar.success("✅ Upload interface activated!")

# Sidebar: Start button at the end for other sources (optional, can be removed if not needed)
st.sidebar.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)  # Spacer

# Initialize clean session state on app start
if "app_initialized" not in st.session_state:
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
    
    # Debug: Show what files are detected
    if uploaded_file is not None:
        st.info(f"🔄 Processing uploaded file: {uploaded_file.name}")
    elif example_file is not None:
        st.info(f"🔄 Processing example image: {example_file['name']}")
    
    # Determine which image to process (regular upload takes priority)
    if uploaded_file is not None:
        # Clear any example selection when user uploads a file
        if "uploaded_example" in st.session_state:
            del st.session_state["uploaded_example"]
        
        # Process regular upload - FIRST OCCURRENCE
        st.write(f"**{uploaded_file.name}**  {uploaded_file.size/1024:.1f}KB")
        
        # Safe image loading with error handling
        try:
            image = Image.open(uploaded_file)
            # Verify the image is valid by attempting to load it
            image.verify()
            # Reopen since verify() closes the file
            uploaded_file.seek(0)  # Reset file pointer
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary to ensure compatibility
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            st.error(f"❌ Invalid or corrupted image file: {e}")
            st.info("💡 Please try uploading a different image file.")
            st.stop()  # Use st.stop() instead of return since we're not in a function
        
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        col1, col2 = st.columns(2)
        with col1:
            display_image_safe(image, caption="Original Image", use_container_width=True)
        
        # Inference with regular YOLO
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
        
        # Inference with regular YOLO
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
            
            # Center the image - using safe display to avoid MediaFileHandler issues
            col_img1, col_img2, col_img3 = st.columns([0.5, 1, 0.5])
            with col_img2:
                # Use safe base64 display for cloud compatibility
                display_image_safe(example_image, caption="", use_container_width=False, width=160)
            
            # Compact button with reduced spacing and smaller size
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
            
            # Create a centered container for smaller button
            col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
            with col_btn2:
                if st.button("Load Image", key="click_myeloblasts", type="secondary"):
                    # Safe cleanup that preserves essential session state
                    if "uploaded_example" in st.session_state:
                        del st.session_state["uploaded_example"]
                    
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
            
            # Center the image - using safe display to avoid MediaFileHandler issues
            col_img1, col_img2, col_img3 = st.columns([0.5, 1, 0.5])
            with col_img2:
                # Use safe base64 display for cloud compatibility
                display_image_safe(example_image, caption="", use_container_width=False, width=160)
            
            # Compact button with reduced spacing and smaller size
            st.markdown("<div style='margin-top: -10px;'></div>", unsafe_allow_html=True)
            
            # Create a centered container for smaller button
            col_btn1, col_btn2, col_btn3 = st.columns([0.5, 1, 0.5])
            with col_btn2:
                if st.button("Load Image", key="click_neutrophils", type="secondary"):
                    # Safe cleanup that preserves essential session state
                    if "uploaded_example" in st.session_state:
                        del st.session_state["uploaded_example"]
                    
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
    # Improved cloud detection
    is_cloud_deployment = any([
        'STREAMLIT_CLOUD' in os.environ,
        'STREAMLIT_SHARING' in os.environ,
        'RENDER' in os.environ,
        'HEROKU' in os.environ,
        'VERCEL' in os.environ,
        'streamlit.io' in os.getenv('SERVER_NAME', ''),
        IS_CLOUD or force_cloud_mode
    ])
    
    if is_cloud_deployment:
        st.warning("⚠️ Webcam is not available on cloud deployments")
        st.info("""
        **📸 Single Photo Analysis Available**
        
        Real-time webcam is only available when running locally. However, you can still analyze photos using the camera input below!
        
        **Other options:**
        • Switch to **'Image'** source to upload saved photos
        • Switch to **'Video'** source to upload video files
        • Run locally for real-time webcam access
        """)
        
        # Alternative: Camera input for single photos
        st.markdown("### 📷 Single Photo Capture")
        picture = st.camera_input("Take a picture for analysis")
        
        if picture is not None:
            # Process the captured image
            image = Image.open(picture)
            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            with col1:
                display_image_safe(image, caption="Captured Image", use_container_width=True)
            
            # Inference
            results = model(image_np, conf=conf, iou=iou, classes=selected_inds)
            result = results[0]
            annotated_frame = result.plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            with col2:
                display_image_safe(annotated_frame, caption="Analysis Result", use_container_width=True)
            
            # Results table
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                results_md = """
                <div style='margin-top:30px;'>
                <h4 style='color:#155a5a;'>Detection Results</h4>
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
    else:
        # Local deployment - full webcam functionality
        st.markdown("### 🎥 Real-time Webcam Analysis")
        
        # Initialize webcam session state
        if "webcam_running" not in st.session_state:
            st.session_state.webcam_running = False
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Webcam", key="start_webcam", disabled=st.session_state.webcam_running):
                st.session_state.webcam_running = True
                st.rerun()
        
        with col2:
            if st.button("⏸️ Stop Webcam", key="stop_webcam", disabled=not st.session_state.webcam_running):
                st.session_state.webcam_running = False
                st.rerun()
        
        with col3:
            # Frame rate control
            fps_limit = st.selectbox("FPS Limit", [5, 10, 15, 20, 30], index=1, help="Lower FPS reduces CPU usage")
        
        # Display webcam feed
        if st.session_state.webcam_running:
            # Create placeholders for frames
            col1, col2 = st.columns(2)
            org_frame_placeholder = col1.empty()
            ann_frame_placeholder = col2.empty()
            
            # Status indicators
            status_col1, status_col2 = st.columns(2)
            fps_display = status_col1.empty()
            detection_count = status_col2.empty()
            
            # Try to initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Set webcam properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, fps_limit)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check:")
                st.markdown("""
                - Webcam is connected and not being used by another application
                - Camera permissions are granted to your browser/system
                - Try refreshing the page and starting again
                """)
                st.session_state.webcam_running = False
            else:
                st.success("✅ Webcam connected successfully!")
                
                # Frame processing variables
                frame_count = 0
                import time
                start_time = time.time()
                frame_skip = max(1, 30 // fps_limit)  # Skip frames to achieve target FPS
                
                try:
                    # Main webcam loop
                    while st.session_state.webcam_running:
                        success, frame = cap.read()
                        
                        if not success:
                            st.warning("⚠️ Failed to read frame from webcam.")
                            time.sleep(0.1)  # Brief pause before retrying
                            continue
                        
                        frame_count += 1
                        
                        # Skip frames for FPS control
                        if frame_count % frame_skip != 0:
                            continue
                        
                        # Process frame with model
                        try:
                            if enable_trk == "Yes":
                                results = model.track(
                                    frame, conf=conf, iou=iou, classes=selected_inds, persist=True
                                )
                            else:
                                results = model(frame, conf=conf, iou=iou, classes=selected_inds)
                            
                            # Get the first result
                            result = results[0]
                            annotated_frame = result.plot()
                            
                            # Convert frames for display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display frames using safe display method
                            org_frame_placeholder.image(frame_rgb, caption="📹 Live Webcam Feed", use_container_width=True)
                            ann_frame_placeholder.image(annotated_frame_rgb, caption="🔍 Detection Results", use_container_width=True)
                            
                            # Calculate and display FPS
                            current_time = time.time()
                            elapsed_time = current_time - start_time
                            if elapsed_time > 0:
                                current_fps = frame_count / elapsed_time
                                fps_display.metric("FPS", f"{current_fps:.1f}")
                            
                            # Count detections
                            if hasattr(result, 'boxes') and result.boxes is not None:
                                detection_count.metric("Detections", len(result.boxes))
                            else:
                                detection_count.metric("Detections", 0)
                            
                        except Exception as model_error:
                            st.error(f"Model processing error: {model_error}")
                            continue
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(1.0 / fps_limit)
                        
                        # Check if stop was pressed (session state updated)
                        if not st.session_state.webcam_running:
                            break
                
                except Exception as e:
                    st.error(f"🚨 Webcam error: {str(e)}")
                    st.info("💡 Try restarting the webcam or refreshing the page.")
                
                finally:
                    # Always release the webcam
                    cap.release()
                    cv2.destroyAllWindows()
                    st.session_state.webcam_running = False
                    st.info("📴 Webcam stopped.")
        
        else:
            st.info("🎥 Click '▶️ Start Webcam' to begin real-time analysis.")
            st.markdown("""
            **💡 Tips for best performance:**
            - Close other applications using the webcam
            - Use lower FPS settings for better performance
            - Ensure good lighting for better detection
            - Position the webcam steadily for tracking
            """)

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
