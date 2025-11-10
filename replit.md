# PDF Label Cropper

## Overview

PDF Label Cropper is a Streamlit-based web application that automatically detects and crops label areas from PDF documents. The application uses computer vision techniques to identify rectangular label borders within PDFs, allowing users to extract clean label images with configurable quality and padding settings. The tool is designed for processing shipping labels, product labels, or any PDF containing a bordered rectangular region of interest.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Problem**: Need an accessible, web-based interface for PDF label processing without requiring local installation.

**Solution**: Streamlit framework for rapid web UI development.

**Rationale**: Streamlit provides a Python-native approach to building interactive web applications with minimal code. It handles session state management, file uploads, and real-time UI updates automatically, making it ideal for image processing tools that require immediate visual feedback.

**Key Features**:
- File upload handling for PDF documents
- Adjustable processing parameters (threshold, padding, DPI)
- Real-time preview of detection results
- Download capability for processed images

### Backend Architecture

**Problem**: Need to extract and process images from PDFs, detect label boundaries using computer vision, and crop with precision.

**Solution**: Modular processor class (`PDFLabelProcessor`) combining PDF rendering with OpenCV-based contour detection.

**Components**:
1. **PDF Rendering**: PyMuPDF (fitz) converts PDF pages to high-resolution images
2. **Border Detection**: OpenCV Canny edge detection and contour approximation to find rectangular borders
3. **Cropping Logic**: Bounding box calculation with configurable padding
4. **Aspect Ratio Control**: Optional aspect ratio locking to maintain label proportions

**Design Patterns**:
- Single Responsibility: Separate UI (`app.py`) from processing logic (`label_processor.py`)
- Configurable Parameters: Threshold, padding, and DPI exposed as constructor parameters
- Stateful Processing: Maintains original and processed images for comparison

### Image Processing Pipeline

**Problem**: Accurately identify label borders that may vary in thickness, contrast, and position.

**Solution**: Multi-stage computer vision pipeline:

1. **Edge Detection**: Canny algorithm identifies borders in grayscale conversion
2. **Morphological Operations**: Dilation strengthens weak edges for better contour detection
3. **Contour Filtering**: Validates contours by area ratio (15-92% of page), aspect ratio (0.4-2.6), and rectangularity (4 vertices)
4. **Border Analysis**: Checks border thickness consistency to distinguish true label borders from content

**Alternatives Considered**:
- Template matching: Rejected due to label variation
- ML-based object detection: Overkill for structured documents with clear borders

**Pros**: Lightweight, no training data required, works on varied label formats
**Cons**: May struggle with very faint borders or complex backgrounds

### Session State Management

**Problem**: Preserve processing results across user interactions without re-processing.

**Solution**: Streamlit session state stores cropped images, preview images, and crop metadata.

**Implementation**:
- `st.session_state.processed`: Processing completion flag
- `st.session_state.cropped_img`: Final cropped image
- `st.session_state.preview_img`: Annotated preview with detection overlay
- `st.session_state.crop_info`: Bounding box coordinates and dimensions

## External Dependencies

### Core Libraries

**PyMuPDF (fitz)**: PDF rendering and rasterization
- Purpose: Convert PDF pages to high-resolution pixel arrays
- Version: Latest stable
- Critical for: Initial image extraction from PDF format

**OpenCV (cv2)**: Computer vision and image processing
- Purpose: Edge detection, contour finding, image manipulation
- Functions used: Canny, findContours, approxPolyDP, boundingRect, dilate
- Critical for: Label border detection algorithm

**NumPy**: Numerical array operations
- Purpose: Image array manipulation and calculations
- Critical for: OpenCV compatibility and array transformations

**Pillow (PIL)**: Image format conversion
- Purpose: Convert between NumPy arrays and image formats for display/export
- Critical for: Streamlit image rendering and file downloads

**Streamlit**: Web application framework
- Purpose: UI rendering, session management, file handling
- Critical for: Entire user interface

### Optional Dependencies

**pyzbar**: Barcode/QR code detection (gracefully degraded if unavailable)
- Purpose: Potential future feature for barcode-based label identification
- Current status: Imported with fallback if not installed
- Impact: Application functions fully without it

### Development Tools

The repository includes a legacy Tkinter UI implementation (`attached_assets/label_processor_ui_*.py`) which appears to be a previous desktop application version. The current production version uses Streamlit exclusively.

### Data Flow

1. User uploads PDF via Streamlit file uploader
2. PDF bytes passed to `PDFLabelProcessor.process_pdf()`
3. PyMuPDF renders first page at specified DPI
4. OpenCV processes image through detection pipeline
5. Cropped result stored in session state
6. Streamlit displays preview and provides download option