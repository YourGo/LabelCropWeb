import React from 'react';

function ImagePreview({ image, title }) {
  const imageUrl = image ? `data:image/png;base64,${image}` : null;

  return (
    <div className="image-preview">
      <h3>{title}</h3>
      {imageUrl ? (
        <img src={imageUrl} alt={title} style={{ maxWidth: '100%', height: 'auto' }} />
      ) : (
        <p>No image available</p>
      )}
    </div>
  );
}

export default ImagePreview;