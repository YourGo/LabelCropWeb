import React from 'react';

function Instructions() {
  return (
    <div className="instructions">
      <h2>📖 How to Use</h2>

      <ol>
        <li><strong>Upload</strong> a PDF file (max 50MB) - you'll see a preview</li>
        <li><strong>Click</strong> ⚙️ Settings to adjust processing parameters if needed</li>
        <li><strong>Click</strong> the green "Detect & Crop" button to process the document</li>
        <li><strong>Download</strong> the cropped label image</li>
      </ol>

      <h3>🎯 Features</h3>
      <ul>
        <li>📄 <strong>PDF Preview</strong>: See thumbnail before processing</li>
        <li>🔍 <strong>Smart Detection</strong>: Automatic border detection using computer vision</li>
        <li>⚙️ <strong>Configurable Settings</strong>: Threshold, padding, DPI, and aspect ratio options</li>
        <li>📐 <strong>Progress Tracking</strong>: Real-time processing updates</li>
        <li>📱 <strong>Mobile Friendly</strong>: Responsive design that works on all devices</li>
        <li>💾 <strong>Easy Download</strong>: One-click download of cropped labels</li>
      </ul>

      <h3>⚡ Tips</h3>
      <ul>
        <li>If detection fails, try adjusting the <strong>Threshold</strong> slider in settings</li>
        <li>Use higher <strong>DPI</strong> for better quality (but slower processing)</li>
        <li>Add <strong>Padding</strong> to include more area around the label</li>
        <li>Enable <strong>Aspect Lock</strong> for standard shipping labels</li>
        <li>Large PDFs (&gt;10MB) may take longer to process</li>
      </ul>
    </div>
  );
}

export default Instructions;