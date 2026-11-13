"""
Depth estimation service using Depth Anything V2 Small.

Provides:
- Automatic depth estimation from an RGB photograph.
- Depth map preparation for the Xinsir Depth ControlNet.

The Xinsir depth ControlNet expects a 3-channel RGB image where:
  brighter (higher value) = closer to camera
  darker  (lower value)   = farther from camera

Depth Anything V2 outputs `predicted_depth` where larger values = closer,
which aligns with the ControlNet's convention (no inversion needed).
"""

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DepthService:
    """Estimates depth from photographs and prepares depth maps for ControlNet."""

    def __init__(self, model_registry):
        """
        Args:
            model_registry: A ModelRegistry instance for lazy model access.
        """
        self._registry = model_registry

    def estimate_depth(self, image: Image.Image) -> Image.Image:
        """
        Estimate a depth map from an RGB photograph.

        Args:
            image: An RGB PIL Image (typically a facial photograph).

        Returns:
            An RGB PIL Image depth map sized to the original image dimensions,
            with brighter pixels indicating closer regions.

        Raises:
            ValueError: If the input image is invalid.
        """
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image.")
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_w, original_h = image.size
        logger.info(
            "Running depth estimation on %dx%d image", original_w, original_h
        )

        # Get the HF depth-estimation pipeline (lazy-loaded)
        estimator = self._registry.get_depth_estimator()

        # Run inference — returns dict with 'predicted_depth' and 'depth'
        result = estimator(image)

        # Use the numerical predicted_depth tensor (not the visualisation)
        predicted_depth = result["predicted_depth"]  # torch.Tensor

        # predicted_depth shape is typically (1, H_model, W_model) or (H, W)
        if predicted_depth.dim() == 2:
            predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
        elif predicted_depth.dim() == 3:
            predicted_depth = predicted_depth.unsqueeze(0)

        # Resize to original image dimensions using bicubic interpolation
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.float(),
            size=(original_h, original_w),
            mode="bicubic",
            align_corners=False,
        )

        # Squeeze back to 2D
        depth_np = predicted_depth.squeeze().cpu().numpy()

        # Normalize to 0–255
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_normalized = (depth_np - d_min) / (d_max - d_min) * 255.0
        else:
            depth_normalized = np.zeros_like(depth_np)

        depth_uint8 = depth_normalized.clip(0, 255).astype(np.uint8)

        # Convert to 3-channel RGB for ControlNet
        depth_map = Image.fromarray(depth_uint8, mode="L").convert("RGB")

        logger.info(
            "Depth estimation complete: output %dx%d, range [%.1f, %.1f]",
            depth_map.size[0], depth_map.size[1], d_min, d_max,
        )
        return depth_map

    @staticmethod
    def prepare_depth_for_controlnet(
        depth_map: Image.Image,
        target_width: int = 1024,
        target_height: int = 1024,
    ) -> Image.Image:
        """
        Prepare a user-supplied or estimated depth map for ControlNet input.

        Ensures the depth map is the correct size and format.

        Args:
            depth_map: A PIL Image depth map (grayscale or RGB).
            target_width: Target width for ControlNet.
            target_height: Target height for ControlNet.

        Returns:
            An RGB PIL Image resized to (target_width, target_height).

        Raises:
            ValueError: If the depth map is invalid.
        """
        if not isinstance(depth_map, Image.Image):
            raise ValueError("Depth map must be a PIL Image.")

        # Ensure RGB
        if depth_map.mode != "RGB":
            depth_map = depth_map.convert("RGB")

        # Resize if needed
        if depth_map.size != (target_width, target_height):
            depth_map = depth_map.resize(
                (target_width, target_height), Image.LANCZOS
            )

        return depth_map

