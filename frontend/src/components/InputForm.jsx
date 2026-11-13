import React, { useState } from 'react';
import ImageUploader from './ImageUploader';
import AdvancedSettings from './AdvancedSettings';

const DEFAULT_SETTINGS = {
  inferenceSteps: 30,
  guidanceScale: 7.5,
  sketchScale: 0.8,
  depthScale: 0.5,
  ipAdapterScale: 0.6,
  seed: '',
  negativePrompt: '',
};

export default function InputForm({ onGenerate, disabled }) {
  const [prompt, setPrompt] = useState('');
  const [photo, setPhoto] = useState(null);
  const [sketch, setSketch] = useState(null);
  const [depthMap, setDepthMap] = useState(null);
  const [estimateDepth, setEstimateDepth] = useState(true);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

  const canSubmit = prompt.trim().length > 0 && !disabled;

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmit) return;

    const params = {
      prompt: prompt.trim(),
      photo: photo || undefined,
      sketch: sketch || undefined,
      depthMap: depthMap || undefined,
      estimateDepth: photo && !depthMap ? estimateDepth : false,
      inferenceSteps: settings.inferenceSteps,
      guidanceScale: settings.guidanceScale,
      sketchScale: settings.sketchScale,
      depthScale: settings.depthScale,
      ipAdapterScale: settings.ipAdapterScale,
      seed: settings.seed !== '' ? Number(settings.seed) : undefined,
      negativePrompt: settings.negativePrompt || undefined,
    };

    onGenerate(params);
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Text Prompt */}
      <div className="form-group">
        <label>Text Prompt <span style={{ color: 'var(--error)' }}>*</span></label>
        <textarea
          rows="3"
          placeholder="Describe the face to generate..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          required
        />
      </div>

      {/* Image Inputs */}
      <ImageUploader
        label="Reference Photo"
        accept="image/png,image/jpeg,image/webp"
        value={photo}
        onChange={setPhoto}
      />

      <ImageUploader
        label="Sketch"
        accept="image/png,image/jpeg,image/webp"
        value={sketch}
        onChange={setSketch}
      />

      <ImageUploader
        label="Depth Map"
        accept="image/png,image/jpeg,image/webp"
        value={depthMap}
        onChange={setDepthMap}
      />

      {/* Estimate Depth checkbox — only when photo is uploaded and no depth map */}
      {photo && !depthMap && (
        <div className="form-group">
          <label style={{ cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={estimateDepth}
              onChange={(e) => setEstimateDepth(e.target.checked)}
            />
            Estimate depth from reference photo
          </label>
        </div>
      )}

      {/* Advanced Settings */}
      <AdvancedSettings values={settings} onChange={setSettings} />

      {/* Submit */}
      <button
        type="submit"
        className="btn btn-primary"
        disabled={!canSubmit}
        style={{ width: '100%' }}
      >
        Generate Face
      </button>
    </form>
  );
}

