import React from 'react';

export default function GenerationViewer({ image, metadata, onRefine, onNewGeneration, onDownload }) {
  return (
    <div className="generation-viewer">
      <h2>Result</h2>
      <img className="result-image" src={image} alt="Generated face" />

      {metadata && (
        <div className="result-meta">
          {metadata.prompt && <p><strong>Prompt:</strong> {metadata.prompt}</p>}
          {metadata.seed != null && <p><strong>Seed:</strong> {metadata.seed}</p>}
          {metadata.active_modalities && (
            <p><strong>Modalities:</strong> {metadata.active_modalities.join(', ')}</p>
          )}
          {metadata.inference_steps && (
            <p><strong>Steps:</strong> {metadata.inference_steps}</p>
          )}
          {metadata.inference_time != null && (
            <p><strong>Inference:</strong> {metadata.inference_time.toFixed(1)}s</p>
          )}
          {metadata.total_time != null && (
            <p><strong>Total:</strong> {metadata.total_time.toFixed(1)}s</p>
          )}
          {(metadata.width && metadata.height) && (
            <p><strong>Resolution:</strong> {metadata.width}×{metadata.height}</p>
          )}
        </div>
      )}

      <div className="action-bar">
        <button className="btn btn-primary" onClick={onRefine}>
          Refine Image
        </button>
        <button className="btn btn-secondary" onClick={onNewGeneration}>
          New Generation
        </button>
        <button className="btn btn-secondary" onClick={onDownload}>
          Download
        </button>
      </div>
    </div>
  );
}

