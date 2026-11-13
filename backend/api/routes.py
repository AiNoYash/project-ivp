"""
FastAPI application factory for the face generation API.

Creates the FastAPI app with all endpoints, wired to the provided
model registry and services. Uses an asyncio lock to serialize
GPU inference operations.
"""

import asyncio
import base64
import io
import logging
import random
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

logger = logging.getLogger(__name__)


def _img_to_b64(img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_app(registry, depth_service, inpainting_service):
    """
    Factory: build the FastAPI app with model dependencies injected.

    Args:
        registry: Initialized ModelRegistry instance.
        depth_service: DepthService instance.
        inpainting_service: InpaintingService instance.
    """
    app = FastAPI(
        title="Multimodal Depth-Aware Face Generation API",
        version="1.0.0",
        description="Academic prototype – synthetic facial visualisation only.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serialize GPU inference – one request at a time
    inference_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health():
        gpu = registry.gpu_info()
        return {
            "status": "ok",
            "version": "1.0.0",
            "stage": "Stage 6 – Full API",
            "gpu_available": gpu.get("gpu_available", False),
            "gpu_name": gpu.get("gpu_name", "N/A"),
            "models_loaded": {
                "base_pipeline": registry._base_pipe is not None,
                "scribble_controlnet": registry._cn_scribble is not None,
                "depth_controlnet": registry._cn_depth is not None,
                "depth_estimator": registry._depth_estimator is not None,
                "inpainting_pipeline": registry._inpaint_pipe is not None,
            },
        }

    # ------------------------------------------------------------------
    # GET /api/models
    # ------------------------------------------------------------------
    @app.get("/api/models")
    async def models_info():
        from backend.config import (
            SDXL_MODEL_ID, CONTROLNET_SCRIBBLE_ID, CONTROLNET_DEPTH_ID,
            IP_ADAPTER_WEIGHT_NAME, DEPTH_ESTIMATOR_ID, INPAINTING_MODEL_ID,
        )
        return {
            "base_model": "Stable Diffusion XL 1.0",
            "base_model_id": SDXL_MODEL_ID,
            "controlnets": ["scribble", "depth"],
            "controlnet_ids": {
                "scribble": CONTROLNET_SCRIBBLE_ID,
                "depth": CONTROLNET_DEPTH_ID,
            },
            "image_adapter": "IP-Adapter Plus Face",
            "image_adapter_weight": IP_ADAPTER_WEIGHT_NAME,
            "depth_estimator": "Depth Anything V2 Small",
            "depth_estimator_id": DEPTH_ESTIMATOR_ID,
            "inpainting": "SDXL Inpainting",
            "inpainting_id": INPAINTING_MODEL_ID,
            "supported_inputs": ["text", "photo", "sketch", "depth"],
        }

    # ------------------------------------------------------------------
    # POST /api/generate
    # ------------------------------------------------------------------
    @app.post("/api/generate")
    async def generate(
        prompt: str = Form(...),
        negative_prompt: Optional[str] = Form(None),
        photo: Optional[UploadFile] = File(None),
        sketch: Optional[UploadFile] = File(None),
        depth_map: Optional[UploadFile] = File(None),
        estimate_depth: bool = Form(False),
        sketch_control_scale: float = Form(0.8),
        depth_control_scale: float = Form(0.5),
        ip_adapter_scale: float = Form(0.6),
        inference_steps: int = Form(30),
        guidance_scale: float = Form(7.5),
        seed: Optional[int] = Form(None),
        width: int = Form(1024),
        height: int = Form(1024),
    ):
        # Validate prompt
        if not prompt or not prompt.strip():
            raise HTTPException(400, "A text prompt is required.")
        prompt = prompt.strip()

        # Validate dimensions
        if width % 8 != 0 or height % 8 != 0:
            raise HTTPException(400, "Width and height must be multiples of 8.")
        if width < 512 or height < 512 or width > 2048 or height > 2048:
            raise HTTPException(400, "Dimensions must be between 512 and 2048.")

        # Validate steps / guidance
        inference_steps = max(1, min(inference_steps, 100))
        guidance_scale = max(1.0, min(guidance_scale, 30.0))

        # Read uploaded file bytes (async)
        photo_bytes = await photo.read() if photo else None
        sketch_bytes = await sketch.read() if sketch else None
        depth_bytes = await depth_map.read() if depth_map else None

        # Resolve seed
        actual_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**32 - 1)

        try:
            async with inference_lock:
                result = await asyncio.to_thread(
                    _sync_generate,
                    registry, depth_service,
                    prompt, negative_prompt,
                    photo_bytes, sketch_bytes, depth_bytes,
                    estimate_depth,
                    sketch_control_scale, depth_control_scale, ip_adapter_scale,
                    inference_steps, guidance_scale, actual_seed, width, height,
                )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(500, f"Generation failed: {type(e).__name__}")

        return JSONResponse({
            "image": f"data:image/png;base64,{_img_to_b64(result['image'])}",
            "metadata": {
                "prompt": prompt,
                "negative_prompt": result["negative_prompt"],
                "seed": actual_seed,
                "active_modalities": result["active_modalities"],
                "conditioning_scales": result["scales"],
                "inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "inference_time": result["inference_time"],
                "total_time": result["total_time"],
            },
        })

    # ------------------------------------------------------------------
    # POST /api/inpaint
    # ------------------------------------------------------------------
    @app.post("/api/inpaint")
    async def inpaint(
        image: UploadFile = File(...),
        mask: UploadFile = File(...),
        prompt: str = Form(...),
        negative_prompt: Optional[str] = Form(None),
        inference_steps: int = Form(30),
        guidance_scale: float = Form(7.5),
        strength: float = Form(0.85),
        seed: Optional[int] = Form(None),
    ):
        if not prompt or not prompt.strip():
            raise HTTPException(400, "A correction prompt is required.")
        prompt = prompt.strip()

        image_bytes = await image.read()
        mask_bytes = await mask.read()

        if not image_bytes:
            raise HTTPException(400, "Original image is required.")
        if not mask_bytes:
            raise HTTPException(400, "Inpainting mask is required.")

        inference_steps = max(1, min(inference_steps, 100))
        guidance_scale = max(1.0, min(guidance_scale, 30.0))
        strength = max(0.0, min(strength, 1.0))
        actual_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**32 - 1)

        try:
            async with inference_lock:
                result = await asyncio.to_thread(
                    _sync_inpaint,
                    registry, inpainting_service,
                    image_bytes, mask_bytes, prompt, negative_prompt,
                    inference_steps, guidance_scale, strength, actual_seed,
                )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            logger.exception("Inpainting failed")
            raise HTTPException(500, f"Inpainting failed: {type(e).__name__}")

        return JSONResponse({
            "image": f"data:image/png;base64,{_img_to_b64(result['image'])}",
            "metadata": result["metadata"],
        })

    return app


# ======================================================================
# Synchronous inference functions (run in thread pool)
# ======================================================================

def _sync_generate(
    registry, depth_service,
    prompt, negative_prompt,
    photo_bytes, sketch_bytes, depth_bytes,
    estimate_depth,
    sketch_scale, depth_scale, ip_scale,
    steps, guidance, seed, width, height,
):
    """Run the full generation pipeline synchronously."""
    import torch
    from backend.services import preprocessing
    from backend.services.depth_service import DepthService
    from backend.config import DEFAULT_NEGATIVE_PROMPT

    t_start = time.time()
    neg = negative_prompt if negative_prompt else DEFAULT_NEGATIVE_PROMPT

    use_sketch = False
    use_depth = False
    use_ip_adapter = False
    active = ["text"]
    scales = {}

    sketch_img = None
    depth_img = None
    photo_img = None

    # --- Process sketch ---
    if sketch_bytes:
        raw = preprocessing.validate_image(sketch_bytes)
        sketch_img = preprocessing.preprocess_sketch(raw, width, height)
        use_sketch = True
        active.append("sketch")
        scales["sketch"] = sketch_scale

    # --- Process photograph (preserve original for depth estimation) ---
    photo_original = None
    if photo_bytes:
        raw = preprocessing.validate_image(photo_bytes)
        preprocessing.validate_image_dimensions(raw)
        photo_original = raw.copy()
        photo_img = preprocessing.preprocess_photo_for_ip_adapter(raw)
        use_ip_adapter = True
        active.append("photo")
        scales["ip_adapter"] = ip_scale

    # --- Depth logic ---
    if depth_bytes:
        raw = preprocessing.validate_image(depth_bytes)
        depth_img = preprocessing.preprocess_depth_map(raw, width, height)
        use_depth = True
        active.append("depth")
        scales["depth"] = depth_scale
    elif photo_original is not None and estimate_depth:
        estimated = depth_service.estimate_depth(photo_original)
        depth_img = DepthService.prepare_depth_for_controlnet(
            estimated, width, height
        )
        use_depth = True
        active.append("depth (estimated)")
        scales["depth"] = depth_scale

    # --- Get pipeline ---
    pipe, cn_count = registry.get_generation_pipeline(
        use_sketch=use_sketch,
        use_depth=use_depth,
        use_ip_adapter=use_ip_adapter,
        ip_adapter_scale=ip_scale,
    )

    gen_kwargs = {
        "prompt": prompt,
        "negative_prompt": neg,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "generator": torch.Generator(device=registry.device).manual_seed(seed),
    }

    # Add conditioning images
    if cn_count == 1:
        if use_sketch:
            gen_kwargs["image"] = sketch_img
            gen_kwargs["controlnet_conditioning_scale"] = sketch_scale
        elif use_depth:
            gen_kwargs["image"] = depth_img
            gen_kwargs["controlnet_conditioning_scale"] = depth_scale
    elif cn_count == 2:
        gen_kwargs["image"] = [sketch_img, depth_img]
        gen_kwargs["controlnet_conditioning_scale"] = [sketch_scale, depth_scale]

    if use_ip_adapter:
        gen_kwargs["ip_adapter_image"] = photo_img

    # --- Run inference ---
    t_infer = time.time()
    result = pipe(**gen_kwargs)
    inference_time = round(time.time() - t_infer, 2)
    total_time = round(time.time() - t_start, 2)

    logger.info(
        "Generation complete: modalities=%s, inference=%.1fs, total=%.1fs",
        active, inference_time, total_time,
    )

    return {
        "image": result.images[0],
        "negative_prompt": neg,
        "active_modalities": active,
        "scales": scales,
        "inference_time": inference_time,
        "total_time": total_time,
    }


def _sync_inpaint(
    registry, inpainting_service,
    image_bytes, mask_bytes, prompt, negative_prompt,
    steps, guidance, strength, seed,
):
    """Run the inpainting pipeline synchronously."""
    from backend.services import preprocessing

    # Parse images
    orig = preprocessing.validate_image(image_bytes)
    mask_raw = Image.open(io.BytesIO(mask_bytes)).convert("L")

    # Resize mask to match image if needed
    w, h = orig.size
    mask_processed = preprocessing.validate_mask(mask_raw, w, h)

    result = inpainting_service.inpaint(
        image=orig,
        mask=mask_processed,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        seed=seed,
    )

    result["metadata"]["seed"] = seed
    return result

