"""
Configuration for the Multimodal Depth-Aware Face Generation backend.
All model identifiers, default parameters, and infrastructure settings.
"""

# ---------------------------------------------------------------------------
# Model identifiers (fixed – do not replace without explicit approval)
# ---------------------------------------------------------------------------
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNET_SCRIBBLE_ID = "xinsir/controlnet-scribble-sdxl-1.0"
CONTROLNET_DEPTH_ID = "xinsir/controlnet-depth-sdxl-1.0"

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT_NAME = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"


DEPTH_ESTIMATOR_ID = "depth-anything/Depth-Anything-V2-Small-hf"

INPAINTING_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# ---------------------------------------------------------------------------
# Default generation parameters
# ---------------------------------------------------------------------------
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, bad anatomy, deformed, extra fingers, blurry, "
    "watermark, text, logo, worst quality"
)

DEFAULT_SKETCH_CONTROL_SCALE = 0.8
DEFAULT_DEPTH_CONTROL_SCALE = 0.5
DEFAULT_IP_ADAPTER_SCALE = 0.6

# ---------------------------------------------------------------------------
# Default inpainting parameters
# ---------------------------------------------------------------------------
DEFAULT_INPAINT_STEPS = 30
DEFAULT_INPAINT_GUIDANCE_SCALE = 7.5
DEFAULT_INPAINT_STRENGTH = 0.85

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------
MODAL_GPU = "L40S"  # Change to "A100" if needed
MODAL_VOLUME_NAME = "hf-model-cache"
MODAL_HF_CACHE_PATH = "/root/.cache/huggingface"
MODAL_SECRET_NAME = "hf-auth"
MODAL_APP_NAME = "face-gen"
MODAL_CONTAINER_TIMEOUT = 600  # seconds

# ---------------------------------------------------------------------------
# Supported image formats
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_FORMATS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_DIMENSION = 4096

