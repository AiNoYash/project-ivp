import React, { useState } from 'react';

export default function AdvancedSettings({ values, onChange }) {
  const [open, setOpen] = useState(false);

  const update = (key, val) => {
    onChange({ ...values, [key]: val });
  };

  return (
    <div className="advanced-settings">
      <button className="toggle-btn" onClick={() => setOpen(!open)}>
        <span>Advanced Settings</span>
        <span>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="settings-body">
          {/* Inference Steps */}
          <div className="setting-item">
            <div className="setting-header">
              <span>Inference Steps</span>
              <span>{values.inferenceSteps}</span>
            </div>
            <input
              type="range" min="1" max="100" step="1"
              value={values.inferenceSteps}
              onChange={(e) => update('inferenceSteps', Number(e.target.value))}
            />
          </div>

          {/* Guidance Scale */}
          <div className="setting-item">
            <div className="setting-header">
              <span>Guidance Scale</span>
              <span>{values.guidanceScale}</span>
            </div>
            <input
              type="range" min="1" max="30" step="0.5"
              value={values.guidanceScale}
              onChange={(e) => update('guidanceScale', Number(e.target.value))}
            />
          </div>

          {/* Sketch Control Scale */}
          <div className="setting-item">
            <div className="setting-header">
              <span>Sketch Control Scale</span>
              <span>{values.sketchScale}</span>
            </div>
            <input
              type="range" min="0" max="1.5" step="0.05"
              value={values.sketchScale}
              onChange={(e) => update('sketchScale', Number(e.target.value))}
            />
          </div>

          {/* Depth Control Scale */}
          <div className="setting-item">
            <div className="setting-header">
              <span>Depth Control Scale</span>
              <span>{values.depthScale}</span>
            </div>
            <input
              type="range" min="0" max="1.5" step="0.05"
              value={values.depthScale}
              onChange={(e) => update('depthScale', Number(e.target.value))}
            />
          </div>

          {/* IP-Adapter Scale */}
          <div className="setting-item">
            <div className="setting-header">
              <span>IP-Adapter Scale</span>
              <span>{values.ipAdapterScale}</span>
            </div>
            <input
              type="range" min="0" max="1.5" step="0.05"
              value={values.ipAdapterScale}
              onChange={(e) => update('ipAdapterScale', Number(e.target.value))}
            />
          </div>

          {/* Seed */}
          <div className="setting-item">
            <label>Seed <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>(empty = random)</span></label>
            <input
              type="number" min="0"
              placeholder="Random"
              value={values.seed}
              onChange={(e) => update('seed', e.target.value)}
            />
          </div>

          {/* Negative Prompt */}
          <div className="setting-item">
            <label>Negative Prompt</label>
            <textarea
              rows="2"
              placeholder="Things to avoid in generation..."
              value={values.negativePrompt}
              onChange={(e) => update('negativePrompt', e.target.value)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

