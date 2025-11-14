# PDF Label Cropper - React + FastAPI

A modern web application for automatically detecting and cropping label regions from PDF documents.

## Architecture

- **Frontend**: React.js with modern UI components
- **Backend**: FastAPI with computer vision processing
- **Processing**: OpenCV and PyMuPDF for PDF/image analysis

## Setup Instructions

### Backend Setup
```bash
cd React-Label/backend
pip install -r requirements.txt
python run.py
```

### Frontend Setup
```bash
cd React-Label/frontend
npm install
npm start
```

### Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Features

- 📄 **PDF Upload & Preview**: Drag & drop interface with file validation
- 🔍 **Smart Detection**: Computer vision algorithms for label detection
- ⚙️ **Configurable Settings**: Threshold, padding, DPI, and aspect ratio controls
- 📐 **Real-time Processing**: Progress indicators and instant results
- 💾 **Easy Download**: One-click download of cropped labels
- 📱 **Responsive Design**: Works on desktop and mobile devices

## API Endpoints

- `POST /api/process-pdf`: Process a PDF file and return cropped label
- `GET /api/health`: Health check endpoint

## Deployment

### Free Options
- **Frontend**: Vercel or Netlify
- **Backend**: Railway or Render

### Production Deployment
1. Deploy backend to Railway/Render
2. Update frontend API URLs
3. Deploy frontend to Vercel/Netlify

## Development

The backend reuses the proven PDF processing logic from the original Streamlit app, ensuring reliable label detection and cropping.