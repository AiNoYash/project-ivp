"""
Image preprocessing utilities for the face generation pipeline.

All functions are stateless and operate on PIL Images.
Each function validates its input and raises ValueError with a
descriptive message on failure.
"""

import io
import logging
from typing import Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def validate_image(image_bytes: bytes, max_size_mb: float = 20.0) -> Image.Image:
    """
    Validate raw image bytes: check file size, open with PIL, convert to RGB.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        ValueError: If validation fails.
    """
    # File size check
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(
            f"Image file is {size_mb:.1f} MB, exceeds limit of {max_size_mb} MB."
        )

    # Open with PIL
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # force full decode
    except Exception as e:
        raise ValueError(f"Cannot open image file: {e}")

    # Convert to RGB
    image = _ensure_rgb(image)
    logger.debug("Validated image: %dx%d, %.1f MB", image.width, image.height, size_mb)
    return image


def validate_image_dimensions(
    image: Image.Image, max_dimension: int = 4096
) -> Image.Image:
    """
    Check that neither dimension exceeds max_dimension.

    Args:
        image: A PIL Image.
        max_dimension: Maximum allowed width or height.

    Returns:
        The same image (unmodified) if valid.

    Raises:
        ValueError: If any dimension exceeds the limit.
    """
    w, h = image.size
    if w > max_dimension or h > max_dimension:
        raise ValueError(
            f"Image dimensions {w}x{h} exceed the maximum of "
            f"{max_dimension}x{max_dimension}. Please resize before uploading."
        )
    return image


# ---------------------------------------------------------------------------
# Resizing and cropping
# ---------------------------------------------------------------------------

def resize_for_generation(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 1024,
) -> Image.Image:
    """
    Resize to target dimensions with aspect-ratio-preserving center crop.

    Steps:
      1. Scale so the *shorter* side matches the target (cover strategy).
      2. Center-crop to exact target dimensions.

    Args:
        image: An RGB PIL Image.
        target_width: Desired output width.
        target_height: Desired output height.

    Returns:
        An RGB PIL Image of exactly (target_width, target_height).
    """
    image = _ensure_rgb(image)
    w, h = image.size

    # Already exact size
    if w == target_width and h == target_height:
        return image

    # Scale so the image covers the target rectangle
    scale = max(target_width / w, target_height / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    image = image.crop((left, top, left + target_width, top + target_height))

    logger.debug("Resized %dx%d → %dx%d", w, h, target_width, target_height)
    return image


# ---------------------------------------------------------------------------
# Sketch preprocessing (for Xinsir Scribble ControlNet)
# ---------------------------------------------------------------------------

def preprocess_sketch(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 1024,
) -> Image.Image:
    """
    Prepare a hand-drawn sketch for the Xinsir Scribble ControlNet.

    The ControlNet expects white lines on a black background (RGB, 3-channel).

    Steps:
      1. Convert to grayscale.
      2. Apply binary threshold (> 128 → 255, else 0).
      3. Resize to target dimensions.
      4. Convert back to 3-channel RGB.

    Args:
        image: A PIL Image of a hand-drawn sketch.
        target_width: Target width.
        target_height: Target height.

    Returns:
        A processed RGB PIL Image ready for ControlNet.
    """
    # Convert to grayscale
    gray = image.convert("L")

    # Binary threshold for clean line art
    arr = np.array(gray)
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)
    binary = Image.fromarray(arr, mode="L")

    # Resize
    if binary.size != (target_width, target_height):
        binary = binary.resize((target_width, target_height), Image.LANCZOS)
        # Re-threshold after resize (LANCZOS may introduce intermediate values)
        arr = np.array(binary)
        arr = np.where(arr > 128, 255, 0).astype(np.uint8)
        binary = Image.fromarray(arr, mode="L")

    # Convert to 3-channel RGB
    sketch_rgb = binary.convert("RGB")
    logger.debug("Sketch preprocessed: %dx%d", target_width, target_height)
    return sketch_rgb


# ---------------------------------------------------------------------------
# Photo preprocessing (for IP-Adapter Plus Face)
# ---------------------------------------------------------------------------

def preprocess_photo_for_ip_adapter(
    image: Image.Image, max_side: int = 1024
) -> Image.Image:
    """
    Prepare a facial photograph for IP-Adapter Plus Face.

    The pipeline's CLIPImageProcessor handles the final 224×224 resize and
    normalization internally, so we only ensure the image is RGB and a
    reasonable resolution.

    Args:
        image: A PIL Image of a face.
        max_side: If any side exceeds this, resize proportionally.

    Returns:
        An RGB PIL Image ready for ip_adapter_image input.
    """
    image = _ensure_rgb(image)
    w, h = image.size

    # Down-scale very large images to avoid excessive memory use
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Photo resized for IP-Adapter: %dx%d → %dx%d", w, h, new_w, new_h)

    return image


# ---------------------------------------------------------------------------
# Depth map preprocessing
# ---------------------------------------------------------------------------

def preprocess_depth_map(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 1024,
) -> Image.Image:
    """
    Prepare a user-supplied depth map for the Depth ControlNet.

    Ensures the image is RGB and resized to the generation target dimensions.

    Args:
        image: A PIL Image depth map.
        target_width: Target width.
        target_height: Target height.

    Returns:
        An RGB PIL Image of the depth map.
    """
    image = _ensure_rgb(image)

    if image.size != (target_width, target_height):
        image = image.resize((target_width, target_height), Image.LANCZOS)
        logger.debug("Depth map resized to %dx%d", target_width, target_height)

    return image


# ---------------------------------------------------------------------------
# Mask preprocessing (for inpainting)
# ---------------------------------------------------------------------------

def validate_mask(
    image: Image.Image,
    target_width: int = 1024,
    target_height: int = 1024,
) -> Image.Image:
    """
    Validate and preprocess an inpainting mask.

    Convention: white (255) = area to regenerate, black (0) = preserve.

    Steps:
      1. Convert to grayscale.
      2. Resize to target dimensions.
      3. Apply binary threshold at 128.

    Args:
        image: A PIL Image mask.
        target_width: Target width (must match the generated image).
        target_height: Target height.

    Returns:
        A grayscale ("L" mode) PIL Image with binary values.

    Raises:
        ValueError: If the mask contains no white region.
    """
    gray = image.convert("L")

    # Resize
    if gray.size != (target_width, target_height):
        gray = gray.resize((target_width, target_height), Image.LANCZOS)

    # Binarize
    arr = np.array(gray)
    arr = np.where(arr > 128, 255, 0).astype(np.uint8)

    # Sanity check: mask must have some white pixels
    if arr.max() == 0:
        raise ValueError(
            "Mask is entirely black — no region selected for inpainting. "
            "Draw white over the area you want to regenerate."
        )

    mask = Image.fromarray(arr, mode="L")
    white_pct = (arr == 255).sum() / arr.size * 100
    logger.debug(
        "Mask preprocessed: %dx%d, %.1f%% white (regenerate)",
        target_width, target_height, white_pct,
    )
    return mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL Image to RGB, handling RGBA, L, P, etc."""
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        # Composite onto white background
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        return bg
    return image.convert("RGB")

