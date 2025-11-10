print("Starting app.py")
try:
    import streamlit as st
    import cv2
    import numpy as np
    from PIL import Image
    import io
    from label_processor import PDFLabelProcessor
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    raise

print("Setting page config")
st.set_page_config(
    page_title="PDF Label Cropper",
    page_icon="📄",
    layout="wide"
)

print("Page config set")
st.title("📄 PDF Label Cropper")
st.markdown("Upload a PDF label document to automatically detect and crop the label area.")

if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.cropped_img = None
    st.session_state.preview_img = None
    st.session_state.crop_info = None

uploaded_file = st.file_uploader("Choose a PDF or image", type=['pdf','png','jpg','jpeg'])

if uploaded_file is not None:
    with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.slider("Detection Threshold", 150, 250, 200, 
                                help="Adjust if label detection fails")
        with col2:
            padding = st.slider("Padding %", 0, 10, 2,
                              help="Extra space around detected label")
        with col3:
            dpi = st.selectbox("DPI (PDF only)", [150, 200, 300, 400, 600], index=2,
                             help="Higher DPI = better quality, larger file")
    
    if st.button("🔍 Process Label", type="primary", width='stretch'):
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                pdf_bytes = uploaded_file.read()
                
                processor = PDFLabelProcessor(
                    threshold=threshold,
                    padding_percent=padding,
                    dpi=dpi,
                    aspect_lock=True
                )
                
                cropped_img, preview_img, crop_info = processor.process_pdf(pdf_bytes)
                
                st.session_state.processed = True
                st.session_state.cropped_img = cropped_img
                st.session_state.preview_img = preview_img
                st.session_state.crop_info = crop_info
                st.session_state.processor = processor
                
                st.success(f"✅ Label detected successfully! Size: {crop_info[2]}x{crop_info[3]} pixels")
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
                st.info("Try adjusting the detection threshold in Advanced Settings.")

if st.session_state.processed:
    st.divider()
    
    tab1, tab2 = st.tabs(["📋 Cropped Label", "🔍 Detection Preview"])
    
    with tab1:
        st.subheader("Cropped Label Result")
        rgb_cropped = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_cropped, width='stretch')
    
    with tab2:
        st.subheader("Detection Preview")
        st.caption("Green rectangle shows the detected label area")
        rgb_preview = cv2.cvtColor(st.session_state.preview_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_preview, width='stretch')
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        rgb_img = cv2.cvtColor(st.session_state.cropped_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format='PNG')
        png_buffer.seek(0)
        
        st.download_button(
            label="⬇️ Download as PNG",
            data=png_buffer,
            file_name="cropped_label.png",
            mime="image/png",
            width='stretch'
        )
    
    with col2:
        pdf_buffer = io.BytesIO()
        pil_img.save(pdf_buffer, format='PDF', resolution=100.0)
        pdf_buffer.seek(0)
        
        st.download_button(
            label="⬇️ Download as PDF",
            data=pdf_buffer,
            file_name="cropped_label.pdf",
            mime="application/pdf",
            width='stretch'
        )

st.divider()

with st.expander("ℹ️ How it works"):
    st.markdown("""
    This application uses advanced computer vision techniques to automatically detect and crop shipping labels:
    
    1. **Border Detection**: Identifies printed rectangular borders around labels
    2. **Cut-line Detection**: Detects horizontal dashed or solid cut lines
    3. **Barcode Analysis**: Uses barcode position and orientation to refine the crop area
    4. **Multi-pass Refinement**: Runs a second pass to optimize the crop boundaries
    
    The system automatically determines the best approach based on the label style.
    """)

