const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      msg = body.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export async function healthCheck() {
  return request('/health');
}

export async function getModels() {
  return request('/api/models');
}

export async function generateImage({
  prompt, negativePrompt, photo, sketch, depthMap,
  estimateDepth, sketchScale, depthScale, ipAdapterScale,
  inferenceSteps, guidanceScale, seed, width, height,
}) {
  const form = new FormData();
  form.append('prompt', prompt);
  if (negativePrompt) form.append('negative_prompt', negativePrompt);
  if (photo) form.append('photo', photo);
  if (sketch) form.append('sketch', sketch);
  if (depthMap) form.append('depth_map', depthMap);
  form.append('estimate_depth', String(!!estimateDepth));
  if (sketchScale != null) form.append('sketch_control_scale', String(sketchScale));
  if (depthScale != null) form.append('depth_control_scale', String(depthScale));
  if (ipAdapterScale != null) form.append('ip_adapter_scale', String(ipAdapterScale));
  if (inferenceSteps != null) form.append('inference_steps', String(inferenceSteps));
  if (guidanceScale != null) form.append('guidance_scale', String(guidanceScale));
  if (seed != null && seed !== '' && seed >= 0) form.append('seed', String(seed));
  if (width) form.append('width', String(width));
  if (height) form.append('height', String(height));

  return request('/api/generate', { method: 'POST', body: form });
}

export async function inpaintImage({
  image, mask, prompt, negativePrompt,
  inferenceSteps, guidanceScale, strength, seed,
}) {
  const form = new FormData();
  form.append('image', image);
  form.append('mask', mask);
  form.append('prompt', prompt);
  if (negativePrompt) form.append('negative_prompt', negativePrompt);
  if (inferenceSteps != null) form.append('inference_steps', String(inferenceSteps));
  if (guidanceScale != null) form.append('guidance_scale', String(guidanceScale));
  if (strength != null) form.append('strength', String(strength));
  if (seed != null && seed !== '' && seed >= 0) form.append('seed', String(seed));

  return request('/api/inpaint', { method: 'POST', body: form });
}

/** Convert a base64 data URL to a Blob for upload */
export function dataURLtoBlob(dataURL) {
  const [header, data] = dataURL.split(',');
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

