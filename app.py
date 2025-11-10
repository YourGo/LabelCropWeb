import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from label_processor import PDFLabelProcessor

st.set_page_config(
    page_title="PDF Label Cropper",
    page_icon="📄",
    layout="wide"
)

# Hide Streamlit's default header (top-right buttons)
st.markdown("""
<style>
.stAppHeader {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("📄 PDF Label Cropper")
st.markdown("Automatically detect and crop label regions from PDF documents")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None
if 'preview_img' not in st.session_state:
    st.session_state.preview_img = None
if 'crop_info' not in st.session_state:
    st.session_state.crop_info = None

# Sidebar for parameters
st.sidebar.header("⚙️ Processing Parameters")

threshold = st.sidebar.slider(
    "Threshold",
    min_value=150,
    max_value=250,
    value=200,
    help="Adjust detection sensitivity"
)

padding_percent = st.sidebar.slider(
    "Padding %",
    min_value=0,
    max_value=10,
    value=2,
    help="Add padding around detected label"
)

dpi = st.sidebar.select_slider(
    "DPI",
    options=[150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
    value=300,
    help="Resolution for PDF rendering"
)

aspect_lock = st.sidebar.checkbox(
    "Lock 6:4/4:6 Aspect (auto)",
    value=True,
    help="Automatically adjust to standard label aspect ratio"
)

# Check barcode availability
barcode_available = PDFLabelProcessor.barcode_available()
if not barcode_available:
    st.sidebar.warning("⚠️ ZBar not available. Barcode-based detection disabled.")

# File upload
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=['pdf'],
    help="Select a PDF file containing a label"
)

# Process button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    process_button = st.button(
        "🔍 Detect & Crop",
        disabled=(uploaded_file is None),
        use_container_width=True
    )

with col2:
    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )

if reset_button:
    st.session_state.processed = False
    st.session_state.cropped_img = None
    st.session_state.preview_img = None
    st.session_state.crop_info = None
    st.rerun()

# Process the PDF
if process_button and uploaded_file:
    with st.spinner("Processing PDF..."):
        try:
            # Read PDF bytes
            pdf_bytes = uploaded_file.read()
            
            # Initialize processor
            processor = PDFLabelProcessor(
                threshold=threshold,
                padding_percent=padding_percent,
                dpi=dpi,
                aspect_lock=aspect_lock
            )
            
            # Process PDF
            cropped_img, preview_img, crop_info = processor.process_pdf(pdf_bytes)
            
            # Store in session state
            st.session_state.processed = True
            st.session_state.cropped_img = cropped_img
            st.session_state.preview_img = preview_img
            st.session_state.crop_info = crop_info
            
            st.success(f"✅ Label detected: {crop_info[2]}x{crop_info[3]}px at ({crop_info[0]}, {crop_info[1]})")
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.exception(e)

# Display results
if st.session_state.processed and st.session_state.preview_img is not None:
    st.markdown("---")
    
    # Create tabs for preview and cropped result
    tab1, tab2 = st.tabs(["📋 Preview with Detection", "✂️ Cropped Label"])
    
    with tab1:
        st.subheader("Detection Preview")
        preview_rgb = cv2.cvtColor(st.session_state.preview_img, cv2.COLOR_BGR2RGB)
        st.image(preview_rgb, use_container_width=True)
    
    with tab2:
        st.subheader("Cropped Label")
        if st.session_state.cropped_img is not None:
            cropped_rgb = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
            st.image(cropped_rgb, use_container_width=True)
            
            # Download button
            is_success, buffer = cv2.imencode(".png", st.session_state.cropped_img)
            if is_success:
                st.download_button(
                    label="💾 Download Cropped Label",
                    data=buffer.tobytes(),
                    file_name="cropped_label.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # Show crop info
    if st.session_state.crop_info:
        x, y, w, h = st.session_state.crop_info
        st.info(f"📐 Crop Info: Position ({x}, {y}) | Size: {w}×{h}px | Aspect Ratio: {w/h:.2f}")

# Instructions
if not uploaded_file:
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use
    
    1. **Upload** a PDF file containing a label
    2. **Adjust** processing parameters in the sidebar if needed
    3. **Click** "Detect & Crop" to process the document
    4. **Download** the cropped label image
    
    ### 🎯 Features
    
    - Automatic border detection using computer vision
    - Configurable threshold and padding
    - Multiple DPI options for quality control
    - Optional aspect ratio locking (6:4 or 4:6)
    - Barcode-based orientation detection (when available)
    
    ### ⚡ Tips
    
    - If detection fails, try adjusting the **Threshold** slider
    - Use higher **DPI** for better quality (but slower processing)
    - Add **Padding** to include more area around the label
    - Enable **Aspect Lock** for standard shipping labels
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "PDF Label Cropper | Built with Streamlit & OpenCV"
    "</div>",
    unsafe_allow_html=True
)
