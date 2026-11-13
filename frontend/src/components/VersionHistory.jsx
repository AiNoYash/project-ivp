import React from 'react';

export default function VersionHistory({ history, currentId, onSelect, onDownload }) {
  return (
    <div className="version-history">
      <h3>Version History ({history.length})</h3>
      <div className="version-list">
        {history.map((entry, idx) => (
          <div
            key={entry.id}
            className={`version-entry${entry.id === currentId ? ' active' : ''}`}
            onClick={() => onSelect(entry)}
          >
            <img className="thumb" src={entry.image} alt={`v${idx + 1}`} />
            <div className="version-info">
              <div className="version-label">
                v{idx + 1} — {entry.type === 'inpaint' ? 'Inpaint' : 'Generation'}
              </div>
              <div className="version-prompt" title={entry.prompt}>
                {entry.prompt}
              </div>
            </div>
            <button
              className="version-download"
              onClick={(e) => { e.stopPropagation(); onDownload(entry); }}
              title="Download"
            >
              ⬇
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

