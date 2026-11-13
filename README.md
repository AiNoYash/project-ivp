# Multimodal Depth-Aware Suspect Face Generation & Iterative Editing

B.Tech Final Year Project — Academic prototype for multimodal facial image synthesis and iterative refinement using pretrained diffusion models.

> **Disclaimer:** All generated faces are synthetic visualisations, not identification evidence.

## Architecture

```
┌─────────────────┐         ┌──────────────────────────────┐
│  React Frontend │  HTTPS  │  FastAPI Backend (Modal GPU) │
│  localhost:5173  │ ──────► │  NVIDIA L40S / A100          │
└─────────────────┘         │                              │
                            │  • SDXL 1.0 Base             │
                            │  • Scribble ControlNet       │
                            │  • Depth ControlNet          │
                            │  • IP-Adapter Plus Face      │
                            │  • Depth Anything V2 Small   │
                            │  • SDXL Inpainting           │
                            └──────────────────────────────┘
```

## Supported Input Modalities

| Modality | Description |
|----------|-------------|
| **Text** | Descriptive prompt (required) |
| **Sketch** | Hand-drawn facial sketch → Scribble ControlNet |
| **Depth Map** | Explicit depth map → Depth ControlNet |
| **Reference Photo** | Facial photograph → IP-Adapter Plus Face |
| **Estimated Depth** | Auto-estimated from photo via Depth Anything V2 |

Any combination of modalities can be used simultaneously.

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Modal](https://modal.com) account with GPU access
- Hugging Face account with access to SDXL

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   python -m venv .venv
   .venv/Scripts/activate    # Windows
   pip install modal
   ```

2. **Configure Modal:**
   ```bash
   modal token new
   ```

3. **Create Modal secrets:**
   ```bash
   modal secret create hf-auth HF_TOKEN=<your_hf_token>
   ```

4. **Create Modal volume (first time only):**
   ```bash
   modal volume create hf-model-cache
   ```

5. **Start the API server (development):**
   ```bash
   modal serve backend/modal_serve.py
   ```
   This starts an ephemeral GPU container with hot-reload. The terminal will show the API URL.

### Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Configure the API URL:**
   ```bash
   cp .env.example .env
   # Set VITE_API_BASE_URL to the URL from modal serve output
   ```

3. **Start the dev server:**
   ```bash
   npm run dev
   ```
   Opens at `http://localhost:5173`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health, GPU info, loaded models |
| GET | `/api/models` | Model identifiers and capabilities |
| POST | `/api/generate` | Generate face from multimodal inputs |
| POST | `/api/inpaint` | Iterative refinement via mask + correction prompt |

### POST /api/generate (multipart form)

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `prompt` | string | ✅ | — |
| `negative_prompt` | string | | Auto |
| `photo` | file | | — |
| `sketch` | file | | — |
| `depth_map` | file | | — |
| `estimate_depth` | bool | | false |
| `inference_steps` | int | | 30 |
| `guidance_scale` | float | | 7.5 |
| `sketch_control_scale` | float | | 0.8 |
| `depth_control_scale` | float | | 0.5 |
| `ip_adapter_scale` | float | | 0.6 |
| `seed` | int | | random |
| `width` | int | | 1024 |
| `height` | int | | 1024 |

### POST /api/inpaint (multipart form)

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `image` | file | ✅ | — |
| `mask` | file | ✅ | — |
| `prompt` | string | ✅ | — |
| `negative_prompt` | string | | Auto |
| `inference_steps` | int | | 30 |
| `guidance_scale` | float | | 7.5 |
| `strength` | float | | 0.85 |
| `seed` | int | | random |

## Project Structure

```
├── backend/
│   ├── config.py                    # Model IDs, defaults, Modal config
│   ├── modal_serve.py               # Modal ASGI serving entry point
│   ├── modal_app.py                 # Stage 2 integration tests
│   ├── test_stage3_4.py             # Stage 3 & 4 tests
│   ├── api/
│   │   └── routes.py                # FastAPI endpoints
│   └── services/
│       ├── model_registry.py        # Lazy model loading & pipeline composition
│       ├── depth_service.py         # Depth Anything V2 estimation
│       ├── preprocessing.py         # Image validation & preprocessing
│       └── inpainting_service.py    # SDXL inpainting
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                  # Main app with mode state management
│       ├── services/api.js          # API client
│       ├── styles/global.css        # Dark theme
│       └── components/
│           ├── InputForm.jsx        # Text + image upload + settings
│           ├── ImageUploader.jsx    # Reusable upload with preview
│           ├── AdvancedSettings.jsx # Collapsible parameter controls
│           ├── GenerationViewer.jsx # Result display + actions
│           ├── MaskEditor.jsx       # Canvas-based mask drawing
│           ├── CorrectionForm.jsx   # Inpainting prompt + strength
│           └── VersionHistory.jsx   # Version list with thumbnails
└── README.md
```

## Models

| Component | Model | ID |
|-----------|-------|----|
| Base Generator | SDXL 1.0 | stabilityai/stable-diffusion-xl-base-1.0 |
| Sketch ControlNet | Xinsir Scribble | xinsir/controlnet-scribble-sdxl-1.0 |
| Depth ControlNet | Xinsir Depth | xinsir/controlnet-depth-sdxl-1.0 |
| Face Adapter | IP-Adapter Plus Face | h94/IP-Adapter |
| Depth Estimation | Depth Anything V2 Small | depth-anything/Depth-Anything-V2-Small-hf |
| Inpainting | SDXL Inpainting | diffusers/stable-diffusion-xl-1.0-inpainting-0.1 |

## Performance (NVIDIA L40S, FP16)

| Operation | Steps | Time |
|-----------|-------|------|
| Text-only generation | 20 | ~4s |
| Text + ControlNet | 20 | ~5s |
| Text + IP-Adapter | 20 | ~4.5s |
| All modalities | 20 | ~5.5s |
| Inpainting | 30 | ~3s |
| Depth estimation | — | ~3s |

## Known Limitations

- Cold start takes ~60-90s on first request (model loading from cache)
- One GPU inference at a time (serialized via asyncio lock)
- Maximum image upload size: 20 MB
- Dimensions must be multiples of 8, between 512-2048px

## License

Academic use only.
