import React, { useState } from 'react';

export default function CorrectionForm({ maskReady, onSubmit, onCancel }) {
  const [prompt, setPrompt] = useState('');
  const [strength, setStrength] = useState(0.85);

  const canSubmit = maskReady && prompt.trim().length > 0;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmit) return;
    onSubmit({
      prompt: prompt.trim(),
      strength,
    });
  };

  return (
    <form className="correction-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Correction Prompt <span style={{ color: 'var(--error)' }}>*</span></label>
        <textarea
          rows="2"
          placeholder="Describe the correction for the masked region..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
      </div>

      <div className="setting-item">
        <div className="setting-header">
          <span>Strength</span>
          <span>{strength}</span>
        </div>
        <input
          type="range" min="0" max="1" step="0.05"
          value={strength}
          onChange={(e) => setStrength(Number(e.target.value))}
        />
      </div>

      <div className="correction-actions">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!canSubmit}
        >
          Apply Correction
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onCancel}
        >
          Cancel
        </button>
      </div>

      {!maskReady && (
        <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 8 }}>
          Draw a mask on the image above before applying a correction.
        </p>
      )}
    </form>
  );
}

