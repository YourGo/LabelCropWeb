import React from 'react';

function ProcessingProgress() {
  return (
    <div className="processing">
      <div className="spinner"></div>
      <p>🔍 Processing PDF...</p>
    </div>
  );
}

export default ProcessingProgress;