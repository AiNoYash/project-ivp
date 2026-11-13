"""
Inpainting service using SDXL Inpainting 0.1.

Model: diffusers/stable-diffusion-xl-1.0-inpainting-0.1

Provides mask-based iterative editing of generated facial images:
- Accept an original image, a grayscale mask, and a correction prompt.
- Regenerate the masked region while preserving the rest.
- White mask pixels = regenerate, black = preserve.
"""

import logging
import time
from typing import Optional

import torch
from PIL import Image

from backend.config import (
    DEFAULT_INPAINT_STEPS,
    DEFAULT_INPAINT_GUIDANCE_SCALE,
    DEFAULT_INPAINT_STRENGTH,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)

logger = logging.getLogger(__name__)


class InpaintingService:
    """Handles iterative mask-based inpainting on generated facial images."""

    def __init__(self, model_registry):
        """
        Args:
            model_registry: A ModelRegistry instance for lazy model access.
        """
        self._registry = model_registry

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = DEFAULT_INPAINT_STEPS,
        guidance_scale: float = DEFAULT_INPAINT_GUIDANCE_SCALE,
        strength: float = DEFAULT_INPAINT_STRENGTH,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Inpaint a region of the image based on the mask and correction prompt.

        Args:
            image: The original generated image (RGB, typically 1024×1024).
            mask: A grayscale mask (white=regenerate, black=preserve).
            prompt: The correction/editing prompt describing the desired change.
            negative_prompt: Optional negative prompt.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            strength: Denoising strength (0.0 = no change, 1.0 = full regeneration).
            seed: Optional random seed for reproducibility.

        Returns:
            A dict with:
                - "image": The edited PIL Image.
                - "metadata": Dict of generation parameters.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image.")
        if not isinstance(mask, Image.Image):
            raise ValueError("mask must be a PIL Image.")
        if not prompt or not prompt.strip():
            raise ValueError("A correction prompt is required for inpainting.")

        # Ensure correct formats
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Mask must be grayscale for the inpainting pipeline
        if mask.mode != "L":
            mask = mask.convert("L")

        # Ensure dimensions match
        img_w, img_h = image.size
        mask_w, mask_h = mask.size
        if (img_w, img_h) != (mask_w, mask_h):
            logger.warning(
                "Mask size %dx%d does not match image size %dx%d, resizing mask.",
                mask_w, mask_h, img_w, img_h,
            )
            mask = mask.resize((img_w, img_h), Image.LANCZOS)

        if negative_prompt is None:
            negative_prompt = DEFAULT_NEGATIVE_PROMPT

        # Get inpainting pipeline (lazy-loaded)
        pipe = self._registry.get_inpainting_pipeline()

        # Setup generator for reproducibility
        device = self._registry.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        logger.info(
            "Starting inpainting: %dx%d, steps=%d, guidance=%.1f, strength=%.2f",
            img_w, img_h, num_inference_steps, guidance_scale, strength,
        )
        t0 = time.time()

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=img_w,
            height=img_h,
            generator=generator,
        )

        elapsed = time.time() - t0
        edited_image = result.images[0]

        logger.info("Inpainting complete in %.1fs", elapsed)

        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": seed,
            "image_size": list(edited_image.size),
            "time_seconds": round(elapsed, 1),
        }

        return {"image": edited_image, "metadata": metadata}

