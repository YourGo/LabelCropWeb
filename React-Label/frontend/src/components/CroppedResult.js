import React from 'react';

function CroppedResult({ image, cropInfo }) {
  const imageUrl = image ? `data:image/png;base64,${image}` : null;

  const handleDownload = () => {
    if (imageUrl) {
      const link = document.createElement('a');
      link.href = imageUrl;
      link.download = 'cropped_label.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  return (
    <div className="cropped-result">
      <h3>✂️ Cropped Label</h3>
      {imageUrl ? (
        <div>
          <img src={imageUrl} alt="Cropped label" style={{ maxWidth: '100%', height: 'auto' }} />
          <div className="crop-info">
            <p>📐 Crop Info: Position ({cropInfo.x}, {cropInfo.y}) | Size: {cropInfo.width}×{cropInfo.height}px</p>
            <button onClick={handleDownload} className="primary">💾 Download</button>
          </div>
        </div>
      ) : (
        <p>No cropped image available</p>
      )}
    </div>
  );
}

export default CroppedResult;