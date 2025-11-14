import React, { useState, useRef, useEffect } from 'react';

function FileUpload({ onFileSelect, file }) {
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Clear the file input when file is reset
  useEffect(() => {
    if (!file && fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [file]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="file-upload">
      <div
        className={`upload-area ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileSelect}
          id="file-input"
        />
        <label htmlFor="file-input">
          {file ? (
            <div>
              <p>✅ {file.name}</p>
              <p>{(file.size / 1024 / 1024).toFixed(1)} MB</p>
            </div>
          ) : (
            <div>
              <p>📄 Drop PDF here or click to browse</p>
              <p>Max 50MB</p>
            </div>
          )}
        </label>
      </div>
    </div>
  );
}

export default FileUpload;