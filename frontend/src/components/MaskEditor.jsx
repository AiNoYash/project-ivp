import React, { useRef, useState, useEffect, useCallback } from 'react';

const MAX_UNDO = 20;

export default function MaskEditor({ imageDataURL, onMaskChange }) {
  const containerRef = useRef(null);
  const bgCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);

  const [brushSize, setBrushSize] = useState(30);
  const [isDrawing, setIsDrawing] = useState(false);
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });

  // ── Load image onto background canvas ──────────────
  useEffect(() => {
    if (!imageDataURL) return;
    const img = new Image();
    img.onload = () => {
      const w = img.width;
      const h = img.height;
      setImgSize({ w, h });

      const bgCanvas = bgCanvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      bgCanvas.width = w;
      bgCanvas.height = h;
      maskCanvas.width = w;
      maskCanvas.height = h;

      const bgCtx = bgCanvas.getContext('2d');
      bgCtx.drawImage(img, 0, 0);

      const maskCtx = maskCanvas.getContext('2d');
      maskCtx.clearRect(0, 0, w, h);

      setUndoStack([]);
      setRedoStack([]);
      onMaskChange(null);
    };
    img.src = imageDataURL;
  }, [imageDataURL]);

  // ── Coordinate scaling ─────────────────────────────
  const getCanvasCoords = useCallback((e) => {
    const canvas = maskCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  }, []);

  // ── Drawing ────────────────────────────────────────
  const drawDot = useCallback((ctx, x, y) => {
    ctx.fillStyle = 'rgba(255, 0, 0, 0.45)';
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
  }, [brushSize]);

  const handlePointerDown = useCallback((e) => {
    e.preventDefault();
    setIsDrawing(true);
    const { x, y } = getCanvasCoords(e);
    const ctx = maskCanvasRef.current.getContext('2d');
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.45)';
    ctx.beginPath();
    ctx.moveTo(x, y);
    drawDot(ctx, x, y);
  }, [getCanvasCoords, brushSize, drawDot]);

  const handlePointerMove = useCallback((e) => {
    if (!isDrawing) return;
    e.preventDefault();
    const { x, y } = getCanvasCoords(e);
    const ctx = maskCanvasRef.current.getContext('2d');
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.45)';
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  }, [isDrawing, getCanvasCoords, brushSize]);

  const exportMask = useCallback(() => {
    const canvas = maskCanvasRef.current;
    const w = canvas.width;
    const h = canvas.height;
    const ctx = canvas.getContext('2d');
    const srcData = ctx.getImageData(0, 0, w, h);

    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const outCtx = offscreen.getContext('2d');
    const outData = outCtx.createImageData(w, h);

    for (let i = 0; i < srcData.data.length; i += 4) {
      const alpha = srcData.data[i + 3];
      const val = alpha > 0 ? 255 : 0;
      outData.data[i] = val;
      outData.data[i + 1] = val;
      outData.data[i + 2] = val;
      outData.data[i + 3] = 255;
    }

    outCtx.putImageData(outData, 0, 0);
    return offscreen.toDataURL('image/png');
  }, []);

  const handlePointerUp = useCallback(() => {
    if (!isDrawing) return;
    setIsDrawing(false);

    // Save undo snapshot
    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext('2d');
    const snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setUndoStack((prev) => {
      const next = [...prev, snapshot];
      if (next.length > MAX_UNDO) next.shift();
      return next;
    });
    setRedoStack([]);

    // Export mask
    const dataURL = exportMask();
    onMaskChange(dataURL);
  }, [isDrawing, exportMask, onMaskChange]);

  // ── Undo / Redo ────────────────────────────────────
  const handleUndo = useCallback(() => {
    if (undoStack.length === 0) return;
    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext('2d');

    // Save current state to redo
    const current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setRedoStack((prev) => [...prev, current]);

    const newUndo = [...undoStack];
    const prev = newUndo.pop();
    setUndoStack(newUndo);

    if (newUndo.length === 0) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      onMaskChange(null);
    } else {
      ctx.putImageData(prev, 0, 0);
      onMaskChange(exportMask());
    }
  }, [undoStack, exportMask, onMaskChange]);

  const handleRedo = useCallback(() => {
    if (redoStack.length === 0) return;
    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext('2d');

    // Save current to undo
    const current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setUndoStack((prev) => [...prev, current]);

    const newRedo = [...redoStack];
    const snapshot = newRedo.pop();
    setRedoStack(newRedo);

    ctx.putImageData(snapshot, 0, 0);
    onMaskChange(exportMask());
  }, [redoStack, exportMask, onMaskChange]);

  const handleClear = useCallback(() => {
    const canvas = maskCanvasRef.current;
    const ctx = canvas.getContext('2d');

    // Save snapshot before clearing
    const current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setUndoStack((prev) => [...prev, current]);
    setRedoStack([]);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    onMaskChange(null);
  }, [onMaskChange]);

  return (
    <div className="mask-editor">
      <div className="mask-actions">
        <button type="button" className="btn btn-secondary" onClick={handleUndo} disabled={undoStack.length === 0}>
          Undo
        </button>
        <button type="button" className="btn btn-secondary" onClick={handleRedo} disabled={redoStack.length === 0}>
          Redo
        </button>
        <button type="button" className="btn btn-danger" onClick={handleClear}>
          Clear
        </button>
      </div>

      <div
        className="canvas-container"
        ref={containerRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        <canvas ref={bgCanvasRef} />
        <canvas ref={maskCanvasRef} className="overlay-canvas" />
      </div>

      <div className="brush-controls">
        <span>Brush: {brushSize}px</span>
        <input
          type="range" min="5" max="100" step="1"
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
        />
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
        Paint over the region you want to correct. The masked area (red) will be regenerated.
      </p>
    </div>
  );
}

