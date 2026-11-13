"""
Model Registry – lazy-loading and caching of all pipeline components.

Responsible for:
- Loading SDXL base pipeline (text-only, no ControlNet)
- Loading individual ControlNet models
- Composing Multi-ControlNet when both are needed
- Loading / unloading IP-Adapter weights
- Providing the correct pipeline variant for each modality combination
- Loading inpainting and depth estimation models on demand

All models are loaded in FP16 on GPU.  Nothing is loaded until first use.
"""

import gc
import time
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import heavy libraries so the module can be imported at definition time
# ---------------------------------------------------------------------------


def _import_diffusers():
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLInpaintPipeline,
        ControlNetModel,
    )
    return (
        StableDiffusionXLPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLInpaintPipeline,
        ControlNetModel,
    )


class ModelRegistry:
    """
    Singleton-style registry that lazily loads and caches model components.

    Design decisions:
    - The *base* SDXL pipeline is always loaded first (text-only).
    - ControlNet models are loaded once and kept in memory.
    - When ControlNets are needed, a *new* pipeline is created that shares
      the base model's components (UNet, VAE, tokenizer, text encoders) so
      weights are not duplicated in VRAM.
    - IP-Adapter weights are loaded/unloaded on the base pipeline or the
      ControlNet pipeline as needed.
    - The inpainting model is loaded lazily and may coexist with the
      generation pipeline (different checkpoint, different UNet).
    """

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        # Cached components
        self._base_pipe = None          # StableDiffusionXLPipeline (no CN)
        self._cn_scribble = None        # ControlNetModel
        self._cn_depth = None           # ControlNetModel
        self._image_encoder = None      # CLIPVisionModelWithProjection
        self._inpaint_pipe = None       # StableDiffusionXLInpaintPipeline
        self._depth_estimator = None    # Depth Anything V2 Small pipeline

        # Track IP-Adapter state
        self._ip_adapter_loaded_on = None  # reference to pipeline that has it

        self._initialized = False

    # ------------------------------------------------------------------
    # Public: initialization & status
    # ------------------------------------------------------------------

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def initialized(self):
        return self._initialized

    def gpu_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            ),
            "gpu_memory_allocated_gb": round(
                torch.cuda.memory_allocated(0) / (1024**3), 2
            ),
        }

    # ------------------------------------------------------------------
    # Base SDXL pipeline (text-only, no ControlNet)
    # ------------------------------------------------------------------

    def load_base_pipeline(self):
        """Load the plain SDXL pipeline.  Idempotent."""
        if self._base_pipe is not None:
            return
        from backend.config import SDXL_MODEL_ID
        (
            StableDiffusionXLPipeline, _, _, _
        ) = _import_diffusers()

        logger.info("Loading SDXL base pipeline: %s", SDXL_MODEL_ID)
        t0 = time.time()
        self._base_pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_MODEL_ID,
            torch_dtype=self._dtype,
            use_safetensors=True,
        ).to(self._device)
        logger.info("SDXL base loaded in %.1fs", time.time() - t0)
        self._initialized = True

    # ------------------------------------------------------------------
    # ControlNet models
    # ------------------------------------------------------------------

    def _load_scribble_controlnet(self):
        """Load scribble ControlNet on demand. Idempotent."""
        if self._cn_scribble is not None:
            return
        from backend.config import CONTROLNET_SCRIBBLE_ID
        _, _, _, ControlNetModel = _import_diffusers()

        logger.info("Loading scribble ControlNet: %s", CONTROLNET_SCRIBBLE_ID)
        t0 = time.time()
        self._cn_scribble = ControlNetModel.from_pretrained(
            CONTROLNET_SCRIBBLE_ID,
            torch_dtype=self._dtype,
        ).to(self._device)
        logger.info("Scribble ControlNet loaded in %.1fs", time.time() - t0)

    def _load_depth_controlnet(self):
        """Load depth ControlNet on demand. Idempotent."""
        if self._cn_depth is not None:
            return
        from backend.config import CONTROLNET_DEPTH_ID
        _, _, _, ControlNetModel = _import_diffusers()

        logger.info("Loading depth ControlNet: %s", CONTROLNET_DEPTH_ID)
        t0 = time.time()
        self._cn_depth = ControlNetModel.from_pretrained(
            CONTROLNET_DEPTH_ID,
            torch_dtype=self._dtype,
        ).to(self._device)
        logger.info("Depth ControlNet loaded in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Image encoder for IP-Adapter Plus Face (ViT-H)
    # ------------------------------------------------------------------

    def _load_image_encoder(self):
        if self._image_encoder is not None:
            return
        from backend.config import IP_ADAPTER_REPO, IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER
        from transformers import CLIPVisionModelWithProjection

        logger.info("Loading CLIP ViT-H image encoder for IP-Adapter")
        t0 = time.time()
        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
            torch_dtype=self._dtype,
        ).to(self._device)
        logger.info("Image encoder loaded in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Build the correct pipeline for a given modality combination
    # ------------------------------------------------------------------

    def get_generation_pipeline(
        self,
        use_sketch: bool = False,
        use_depth: bool = False,
        use_ip_adapter: bool = False,
        ip_adapter_scale: float = 0.6,
    ):
        """
        Return (pipeline, controlnet_count) ready for inference.

        Rules:
        - No ControlNets → plain StableDiffusionXLPipeline
        - One ControlNet  → StableDiffusionXLControlNetPipeline (single)
        - Two ControlNets → StableDiffusionXLControlNetPipeline + MultiControlNetModel
        - IP-Adapter is loaded/unloaded on whichever pipeline is active.
        """
        self.load_base_pipeline()

        active_cn = []
        if use_sketch:
            self._load_scribble_controlnet()
            active_cn.append(self._cn_scribble)
        if use_depth:
            self._load_depth_controlnet()
            active_cn.append(self._cn_depth)

        cn_count = len(active_cn)

        if cn_count == 0:
            pipe = self._base_pipe
        else:
            (
                _, StableDiffusionXLControlNetPipeline, _, _
            ) = _import_diffusers()

            # Pass a list for multi-ControlNet; single model for single CN
            cn_arg = active_cn[0] if cn_count == 1 else active_cn

            pipe = StableDiffusionXLControlNetPipeline(
                vae=self._base_pipe.vae,
                text_encoder=self._base_pipe.text_encoder,
                text_encoder_2=self._base_pipe.text_encoder_2,
                tokenizer=self._base_pipe.tokenizer,
                tokenizer_2=self._base_pipe.tokenizer_2,
                unet=self._base_pipe.unet,
                scheduler=self._base_pipe.scheduler,
                controlnet=cn_arg,
                image_encoder=self._image_encoder,
                feature_extractor=getattr(self._base_pipe, "feature_extractor", None),
            ).to(self._device)

        # Handle IP-Adapter loading
        if use_ip_adapter:
            self._ensure_ip_adapter(pipe, ip_adapter_scale)
        else:
            self._remove_ip_adapter_if_loaded()  # unload from any pipe

        return pipe, cn_count

    # ------------------------------------------------------------------
    # IP-Adapter management
    # ------------------------------------------------------------------

    def _ensure_ip_adapter(self, pipe, scale: float):
        from backend.config import (
            IP_ADAPTER_REPO,
            IP_ADAPTER_SUBFOLDER,
            IP_ADAPTER_WEIGHT_NAME,
        )
        # Load image encoder first (needed by the pipeline)
        self._load_image_encoder()

        # If already loaded on *this* pipe object, just update scale
        if self._ip_adapter_loaded_on is pipe:
            pipe.set_ip_adapter_scale(scale)
            return

        # Unload from previous pipe if any
        self._remove_ip_adapter_if_loaded(None)

        # Attach image encoder to the pipe before loading adapter
        pipe.image_encoder = self._image_encoder
        if hasattr(pipe, 'feature_extractor') and pipe.feature_extractor is None:
            from transformers import CLIPImageProcessor
            # The h94/IP-Adapter repo doesn't have a preprocessor_config.json
            # in models/image_encoder, so use the standard CLIP processor
            pipe.feature_extractor = CLIPImageProcessor()

        logger.info("Loading IP-Adapter Plus Face weights")
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO,
            subfolder=IP_ADAPTER_SUBFOLDER,
            weight_name=IP_ADAPTER_WEIGHT_NAME,
        )
        pipe.set_ip_adapter_scale(scale)
        self._ip_adapter_loaded_on = pipe

    def _remove_ip_adapter_if_loaded(self, exclude_pipe=None):
        """Unload IP-Adapter from whatever pipeline currently holds it.
        
        If exclude_pipe is provided and matches the loaded pipe, skip unloading
        (used when we're about to reload on the same pipe).
        When called with exclude_pipe=None, always unload.
        """
        if self._ip_adapter_loaded_on is not None:
            if exclude_pipe is not None and self._ip_adapter_loaded_on is exclude_pipe:
                return  # Don't unload from the pipe we're about to reload on
            try:
                self._ip_adapter_loaded_on.unload_ip_adapter()
            except Exception:
                pass
            self._ip_adapter_loaded_on = None

    # ------------------------------------------------------------------
    # Inpainting pipeline (lazy)
    # ------------------------------------------------------------------

    def get_inpainting_pipeline(self):
        if self._inpaint_pipe is not None:
            return self._inpaint_pipe
        from backend.config import INPAINTING_MODEL_ID
        (
            _, _, StableDiffusionXLInpaintPipeline, _
        ) = _import_diffusers()

        logger.info("Loading SDXL inpainting pipeline: %s", INPAINTING_MODEL_ID)
        t0 = time.time()
        self._inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            INPAINTING_MODEL_ID,
            torch_dtype=self._dtype,
            use_safetensors=True,
        ).to(self._device)
        logger.info("Inpainting pipeline loaded in %.1fs", time.time() - t0)
        return self._inpaint_pipe

    # ------------------------------------------------------------------
    # Depth estimator (lazy)
    # ------------------------------------------------------------------

    def get_depth_estimator(self):
        if self._depth_estimator is not None:
            return self._depth_estimator
        from backend.config import DEPTH_ESTIMATOR_ID
        from transformers import pipeline as hf_pipeline

        logger.info("Loading depth estimator: %s", DEPTH_ESTIMATOR_ID)
        t0 = time.time()
        self._depth_estimator = hf_pipeline(
            "depth-estimation",
            model=DEPTH_ESTIMATOR_ID,
            device=self._device,
            torch_dtype=self._dtype,
        )
        logger.info("Depth estimator loaded in %.1fs", time.time() - t0)
        return self._depth_estimator

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def free_generation_pipeline(self):
        """Release ControlNet pipeline memory (keeps base pipe)."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
