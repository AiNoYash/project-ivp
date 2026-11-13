import React, { useState, useCallback } from 'react';
import InputForm from './components/InputForm';
import GenerationViewer from './components/GenerationViewer';
import MaskEditor from './components/MaskEditor';
import CorrectionForm from './components/CorrectionForm';
import VersionHistory from './components/VersionHistory';
import { generateImage, inpaintImage, dataURLtoBlob } from './services/api';

export default function App() {
  const [mode, setMode] = useState('input');       // input | view | edit
  const [currentImage, setCurrentImage] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [maskDataURL, setMaskDataURL] = useState(null);

  // ── Generate ──────────────────────────────────────────────
  const handleGenerate = useCallback(async (params) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await generateImage(params);
      const entry = {
        id: Date.now(),
        image: res.image,
        metadata: res.metadata,
        type: 'generation',
        prompt: params.prompt,
        timestamp: new Date().toISOString(),
        parentId: null,
      };
      setCurrentImage(res.image);
      setMetadata(res.metadata);
      setHistory(prev => [...prev, entry]);
      setMode('view');
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // ── Inpaint ───────────────────────────────────────────────
  const handleInpaint = useCallback(async (params) => {
    if (!currentImage) return;
    setIsLoading(true);
    setError(null);
    try {
      const imageBlob = dataURLtoBlob(currentImage);
      const maskBlob = dataURLtoBlob(params.maskDataURL);
      const res = await inpaintImage({
        image: imageBlob,
        mask: maskBlob,
        prompt: params.prompt,
        negativePrompt: params.negativePrompt,
        inferenceSteps: params.inferenceSteps,
        guidanceScale: params.guidanceScale,
        strength: params.strength,
        seed: params.seed,
      });
      const parentId = history.length > 0 ? history[history.length - 1].id : null;
      const entry = {
        id: Date.now(),
        image: res.image,
        metadata: res.metadata,
        type: 'inpaint',
        prompt: params.prompt,
        timestamp: new Date().toISOString(),
        parentId,
      };
      setCurrentImage(res.image);
      setMetadata(res.metadata);
      setHistory(prev => [...prev, entry]);
      setMaskDataURL(null);
      setMode('view');
    } catch (e) {
      setError(e.message);
    } finally {
      setIsLoading(false);
    }
  }, [currentImage, history]);

  // ── History select ────────────────────────────────────────
  const handleSelectVersion = useCallback((entry) => {
    setCurrentImage(entry.image);
    setMetadata(entry.metadata);
    setMode('view');
  }, []);

  // ── Download ──────────────────────────────────────────────
  const handleDownload = useCallback((dataURL, filename) => {
    const a = document.createElement('a');
    a.href = dataURL;
    a.download = filename || 'generated_face.png';
    a.click();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Multimodal Face Generation</h1>
        <p className="subtitle">Depth-Aware Suspect Face Generation &amp; Iterative Editing</p>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {isLoading && (
        <div className="loading-overlay">
          <div className="spinner" />
          <p>{mode === 'edit' ? 'Applying correction…' : 'Generating image…'}</p>
          <p className="loading-hint">This may take 15–30 seconds on first request (cold start).</p>
        </div>
      )}

      <main className="app-main">
        {/* ── Left: Input / Edit ── */}
        <section className="panel panel-left">
          {mode === 'edit' ? (
            <>
              <h2>Iterative Refinement</h2>
              <MaskEditor
                imageDataURL={currentImage}
                onMaskChange={setMaskDataURL}
              />
              <CorrectionForm
                maskReady={!!maskDataURL}
                onSubmit={(params) => handleInpaint({ ...params, maskDataURL })}
                onCancel={() => { setMode('view'); setMaskDataURL(null); }}
              />
            </>
          ) : (
            <>
              <h2>Input Configuration</h2>
              <InputForm onGenerate={handleGenerate} disabled={isLoading} />
            </>
          )}
        </section>

        {/* ── Right: Result / History ── */}
        <section className="panel panel-right">
          {currentImage && mode !== 'input' ? (
            <GenerationViewer
              image={currentImage}
              metadata={metadata}
              onRefine={() => setMode('edit')}
              onNewGeneration={() => setMode('input')}
              onDownload={() => handleDownload(currentImage, `face_v${history.length}.png`)}
            />
          ) : (
            <div className="placeholder">
              <p>Generated image will appear here</p>
            </div>
          )}

          {history.length > 0 && (
            <VersionHistory
              history={history}
              currentId={history[history.length - 1]?.id}
              onSelect={handleSelectVersion}
              onDownload={(entry) => handleDownload(entry.image, `face_v${history.indexOf(entry) + 1}.png`)}
            />
          )}
        </section>
      </main>

      <footer className="app-footer">
        <p>Academic prototype — all generated faces are synthetic visualisations, not identification evidence.</p>
      </footer>
    </div>
  );
}

