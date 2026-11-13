import React, { useRef } from 'react';

export default function ImageUploader({ label, accept, value, onChange }) {
  const inputRef = useRef(null);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) onChange(file);
  };

  const handleRemove = () => {
    onChange(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const previewURL = value ? URL.createObjectURL(value) : null;

  return (
    <div className="image-uploader">
      <label>{label} <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>(optional)</span></label>
      <div className="upload-area">
        <input
          ref={inputRef}
          type="file"
          accept={accept || 'image/*'}
          onChange={handleChange}
        />
      </div>
      {previewURL && (
        <div className="preview-container">
          <img src={previewURL} alt={`${label} preview`} />
          <button className="remove-btn" onClick={handleRemove} title="Remove">✕</button>
        </div>
      )}
    </div>
  );
}

