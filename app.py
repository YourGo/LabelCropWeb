import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import fitz  # PyMuPDF for PDF preview
from label_processor import PDFLabelProcessor

# Helper function for Font Awesome icons (more reliable than Lucide in Streamlit)
def fa_icon(name, style="solid", size="sm", color="currentColor"):
    return f'<i class="fa-{style} fa-{name} fa-{size}" style="color: {color};"></i>'

st.set_page_config(
    page_title="PDF Label Cropper",
    page_icon="📄",
    layout="wide"
)

# Add Font Awesome CSS
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Hide Streamlit's default header (top-right buttons)
st.markdown("""
<style>
.stAppHeader {visibility: hidden;}

/* Enhanced CTA button styling */
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #4CAF50 !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #45a049 !important;
    box-shadow: 0 6px 12px rgba(76, 175, 80, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .stColumn > div > div > div {
        flex-direction: column !important;
    }
    .stButton > button {
        width: 100% !important;
        margin-bottom: 8px !important;
    }
}

/* Better file uploader styling */
.stFileUploader > div > div > div > div {
    border: 2px dashed #4CAF50 !important;
    border-radius: 8px !important;
    padding: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize settings in session state
if 'threshold' not in st.session_state:
    st.session_state.threshold = 200
if 'padding_percent' not in st.session_state:
    st.session_state.padding_percent = 2
if 'dpi' not in st.session_state:
    st.session_state.dpi = 300
if 'aspect_lock' not in st.session_state:
    st.session_state.aspect_lock = True

# Settings modal
@st.dialog("Processing Settings")
def settings_modal():
    st.header(f"{fa_icon('cog')} Processing Parameters")
    
    # Threshold with help
    col_thresh, col_help = st.columns([3, 1])
    with col_thresh:
        st.session_state.threshold = st.slider(
            "Threshold",
            min_value=150,
            max_value=250,
            value=st.session_state.threshold,
            help="Adjust detection sensitivity"
        )
    with col_help:
        with st.popover(fa_icon("info-circle")):
            st.markdown("""
            **Threshold**: Controls how sensitive the label detection is.
            - Lower values (150-180): More sensitive, may detect noise
            - Higher values (220-250): Less sensitive, may miss faint labels
            - Default 200: Good balance for most shipping labels
            """)
    
    # Padding with help
    col_pad, col_help = st.columns([3, 1])
    with col_pad:
        st.session_state.padding_percent = st.slider(
            "Padding %",
            min_value=0,
            max_value=10,
            value=st.session_state.padding_percent,
            help="Add padding around detected label"
        )
    with col_help:
        with st.popover(fa_icon("info-circle")):
            st.markdown("""
            **Padding**: Adds extra space around the detected label.
            - 0%: Exact crop (may cut off edges)
            - 2%: Small buffer (recommended)
            - 5-10%: Large buffer for safety
            """)
    
    # DPI with help
    col_dpi, col_help = st.columns([3, 1])
    with col_dpi:
        st.session_state.dpi = st.select_slider(
            "DPI",
            options=[150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
            value=st.session_state.dpi,
            help="Resolution for PDF rendering"
        )
    with col_help:
        with st.popover(fa_icon("info-circle")):
            st.markdown("""
            **DPI**: Image resolution for processing.
            - 150-200: Fast, good for simple labels
            - 300: Standard, balanced quality/speed
            - 400-600: High quality, slower processing
            """)
    
    # Aspect lock with help
    col_aspect, col_help = st.columns([3, 1])
    with col_aspect:
        st.session_state.aspect_lock = st.checkbox(
            "Lock 6:4/4:6 Aspect (auto)",
            value=st.session_state.aspect_lock,
            help="Automatically adjust to standard label aspect ratio"
        )
    with col_help:
        with st.popover(fa_icon("info-circle")):
            st.markdown("""
            **Aspect Lock**: Forces standard shipping label ratios.
            - 6:4 (1.5:1): Horizontal labels
            - 4:6 (1:1.5): Vertical labels
            - Auto-adjusts detected region to fit
            """)
    
    # Check barcode availability
    barcode_available = PDFLabelProcessor.barcode_available()
    if not barcode_available:
        st.warning(f"{fa_icon('exclamation-triangle')} ZBar not available. Barcode-based detection disabled.")
    
    if st.button(f"{fa_icon('check-circle')} Apply Settings", use_container_width=True):
        st.rerun()

# Header with settings button
col_title, col_settings = st.columns([4, 1])
with col_title:
    st.title(f"{fa_icon('file-alt')} PDF Label Cropper")
with col_settings:
    if st.button(f"{fa_icon('cog')} Settings", use_container_width=True):
        settings_modal()

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



# File upload with validation and preview
uploaded_file = st.file_uploader(
    "Upload PDF",
    type=['pdf'],
    help="Select a PDF file containing a label (max 50MB)"
)

# File validation and preview
if uploaded_file is not None:
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
    
    if file_size > 50:
        st.error(f"{fa_icon('times-circle')} File too large: {file_size:.1f}MB (max 50MB)")
        uploaded_file = None
    elif file_size > 10:
        st.warning(f"{fa_icon('exclamation-triangle')} Large file: {file_size:.1f}MB - processing may be slow")
    else:
        pdf_bytes = uploaded_file.getvalue()  # Define at top level
        
        # Show file info and preview
        col_info, col_preview = st.columns([1, 2])
        
        with col_info:
            st.success(f"{fa_icon('check-circle')} PDF uploaded: {file_size:.1f}MB")
            
            # Get PDF info
            try:
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_count = len(pdf_doc)
                st.info(f"{fa_icon('file-alt')} {page_count} page{'s' if page_count != 1 else ''}")
                pdf_doc.close()
            except Exception as e:
                st.warning(f"{fa_icon('exclamation-triangle')} Could not read PDF info")
        
        with col_preview:
            # Show first page thumbnail
            try:
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page = pdf_doc.load_page(0)
                pix = page.get_pixmap(matrix=fitz.Matrix(0.3, 0.3))  # 30% scale for thumbnail
                img_data = pix.tobytes("png")
                st.image(img_data, caption="PDF Preview (Page 1)", width=200)
                pdf_doc.close()
            except Exception as e:
                st.info(f"{fa_icon('file-alt')} PDF preview not available")

# Process button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    process_button = st.button(
        f"{fa_icon('search')} Detect & Crop",
        disabled=(uploaded_file is None),
        use_container_width=True,
        type="primary"  # Make it prominent
    )

with col2:
    reset_button = st.button(
        f"{fa_icon('undo')} Reset",
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Reading PDF
        status_text.text(f"{fa_icon('book')} Reading PDF file...")
        progress_bar.progress(10)
        
        pdf_bytes = uploaded_file.getvalue()
        
        # Step 2: Initializing processor
        status_text.text(f"{fa_icon('cog')} Setting up processing parameters...")
        progress_bar.progress(20)
        
        processor = PDFLabelProcessor(
            threshold=st.session_state.threshold,
            padding_percent=st.session_state.padding_percent,
            dpi=st.session_state.dpi,
            aspect_lock=st.session_state.aspect_lock
        )
        
        # Step 3: Processing PDF
        status_text.text(f"{fa_icon('search')} Detecting label regions...")
        progress_bar.progress(50)
        
        cropped_img, preview_img, crop_info = processor.process_pdf(pdf_bytes)
        
        # Step 4: Finalizing
        status_text.text(f"{fa_icon('check-circle')} Processing complete!")
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.processed = True
        st.session_state.cropped_img = cropped_img
        st.session_state.preview_img = preview_img
        st.session_state.crop_info = crop_info
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"{fa_icon('check-circle')} Label detected: {crop_info[2]}x{crop_info[3]}px at ({crop_info[0]}, {crop_info[1]})")
        
    except Exception as e:
        # Clear progress indicators on error
        progress_bar.empty()
        status_text.empty()
        st.error(f"{fa_icon('times-circle')} Processing failed: {str(e)}")
        st.exception(e)

# Display results
if st.session_state.processed and st.session_state.preview_img is not None:
    st.markdown("---")
    
    # Create tabs for preview and cropped result
    tab1, tab2 = st.tabs([f"{fa_icon('clipboard-list')} Preview with Detection", f"{fa_icon('cut')} Cropped Label"])
    
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
                    label=f"{fa_icon('download')} Download Cropped Label",
                    data=buffer.tobytes(),
                    file_name="cropped_label.png",
                    mime="image/png",
                    use_container_width=True
                )
    
    # Show crop info
    if st.session_state.crop_info:
        x, y, w, h = st.session_state.crop_info
        st.info(f"{fa_icon('ruler')} Crop Info: Position ({x}, {y}) | Size: {w}×{h}px | Aspect Ratio: {w/h:.2f}")

# Instructions
if not uploaded_file:
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use
    
    1. **Upload** a PDF file (max 50MB) - you'll see a preview and page count
    2. **Click** """ + fa_icon('cog') + """ Settings to adjust processing parameters if needed
    3. **Click** the green "Detect & Crop" button to process the document
    4. **Download** the cropped label image
    
    ### 🎯 Features
    
    - """ + fa_icon('file-alt') + """ **PDF Preview**: See thumbnail and page count before processing
    - """ + fa_icon('search') + """ **Smart Detection**: Automatic border detection using computer vision
    - """ + fa_icon('cog') + """ **Configurable Settings**: Threshold, padding, DPI, and aspect ratio options
    - """ + fa_icon('ruler') + """ **Progress Tracking**: Real-time progress during processing
    - """ + fa_icon('mobile-alt') + """ **Mobile Friendly**: Responsive design that works on all devices
    - """ + fa_icon('download') + """ **Easy Download**: One-click download of cropped labels
    
    ### ⚡ Tips
    
    - If detection fails, try adjusting the **Threshold** slider in settings
    - Use higher **DPI** for better quality (but slower processing)
    - Add **Padding** to include more area around the label
    - Enable **Aspect Lock** for standard shipping labels
    - Large PDFs (>10MB) may take longer to process
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "PDF Label Cropper | Built with Streamlit & OpenCV"
    "</div>",
    unsafe_allow_html=True
)
