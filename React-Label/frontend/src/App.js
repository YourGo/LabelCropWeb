import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import SettingsModal from './components/SettingsModal';
import ProcessingProgress from './components/ProcessingProgress';
import ImagePreview from './components/ImagePreview';
import CroppedResult from './components/CroppedResult';
import Instructions from './components/Instructions';
import './styles.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [settings, setSettings] = useState({
    threshold: 200,
    paddingPercent: 2,
    dpi: 300,
    aspectLock: true,
    borderPaddingPercent: 10
  });
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  const handleProcess = async () => {
    if (!uploadedFile) return;

    setProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('threshold', settings.threshold);
      formData.append('padding_percent', settings.paddingPercent);
      formData.append('dpi', settings.dpi);
      formData.append('aspect_lock', settings.aspectLock);
      formData.append('border_padding_percent', settings.borderPaddingPercent);

      const response = await fetch('http://localhost:8001/api/process-pdf', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Processing failed');

      const data = await response.json();
      setResults(data);
    } catch (error) {
      alert('Error processing PDF: ' + error.message);
    } finally {
      setProcessing(false);
    }
  };

  const handleReset = () => {
    setUploadedFile(null);
    setResults(null);
  };

  return (
    <div className="app">
      <header>
        <h1>📄 PDF Label Cropper</h1>
        <button onClick={() => setShowSettings(true)}>⚙️ Settings</button>
      </header>

      <main>
        <FileUpload onFileSelect={setUploadedFile} file={uploadedFile} />

        <div className="actions">
          <button
            onClick={handleProcess}
            disabled={!uploadedFile || processing}
            className="primary"
          >
            🔍 Detect & Crop
          </button>
          <button onClick={handleReset}>🔄 Reset</button>
        </div>

        {processing && <ProcessingProgress />}

        {results && (
          <div className="results">
            <ImagePreview image={results.preview_image} title="Detection Preview" />
            <CroppedResult
              image={results.cropped_image}
              cropInfo={results.crop_info}
            />
          </div>
        )}

        {!uploadedFile && <Instructions />}
      </main>

      {showSettings && (
        <SettingsModal
          settings={settings}
          onSettingsChange={setSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </div>
  );
}

export default App;