"""
Modal serving application – exposes the FastAPI backend on a GPU container.

Usage:
    Development (ephemeral, hot-reload):
        modal serve backend/modal_serve.py

    Deployment (persistent URL – requires approval):
        modal deploy backend/modal_serve.py

The SDXL base pipeline is loaded once when the container starts.
ControlNets, depth estimator and inpainting model are loaded lazily
on first request that needs them.
"""

import modal

from backend.config import (
    MODAL_GPU,
    MODAL_VOLUME_NAME,
    MODAL_HF_CACHE_PATH,
    MODAL_SECRET_NAME,
    MODAL_APP_NAME,
    MODAL_CONTAINER_TIMEOUT,
)

hf_cache_vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
app = modal.App(MODAL_APP_NAME + "-api")

container_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "Pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        "huggingface-hub==0.24.7",
        "ftfy==6.2.3",
        "fastapi[standard]==0.112.4",
        "python-multipart==0.0.9",
        "uvicorn==0.30.6",
    )
    .env({"HF_HOME": MODAL_HF_CACHE_PATH})
    .add_local_python_source("backend")
)


@app.cls(
    gpu=MODAL_GPU,
    image=container_image,
    volumes={MODAL_HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=MODAL_CONTAINER_TIMEOUT,
)
class InferenceServer:
    """GPU inference server – loads models once, serves FastAPI app."""

    @modal.enter()
    def setup(self):
        import logging
        logging.basicConfig(level=logging.INFO)

        from backend.services.model_registry import ModelRegistry
        from backend.services.depth_service import DepthService
        from backend.services.inpainting_service import InpaintingService

        self.registry = ModelRegistry()
        self.registry.load_base_pipeline()

        self.depth_service = DepthService(self.registry)
        self.inpainting_service = InpaintingService(self.registry)

        print("✅ InferenceServer ready")

    @modal.asgi_app()
    def web(self):
        from backend.api.routes import create_app
        return create_app(
            registry=self.registry,
            depth_service=self.depth_service,
            inpainting_service=self.inpainting_service,
        )

