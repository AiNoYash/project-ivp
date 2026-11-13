"""
Stage 2 Verification Audit

Comprehensive audit script addressing:
1. Accurate inference timing with torch.cuda.synchronize()
2. IP-Adapter lifecycle and shared UNet state contamination
3. Model compatibility verification
4. Visual evidence (images returned as base64)
5. GPU and scheduler configuration reporting

Run with:
    modal run backend/audit_stage2.py
"""

import modal
import base64
import io
import json

MODAL_GPU = "L40S"
MODAL_VOLUME_NAME = "hf-model-cache"
MODAL_HF_CACHE_PATH = "/root/.cache/huggingface"
MODAL_SECRET_NAME = "hf-auth"
MODAL_APP_NAME = "face-gen-audit"
MODAL_CONTAINER_TIMEOUT = 900  # 15 min for thorough audit

hf_cache_vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
app = modal.App(MODAL_APP_NAME)

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
    )
    .env({"HF_HOME": MODAL_HF_CACHE_PATH})
    .add_local_python_source("backend")
)


def _img_to_b64(img) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _create_test_sketch(width=1024, height=1024):
    import numpy as np
    from PIL import Image
    import cv2
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(img, (width // 2, height // 2), (200, 280), 0, 0, 360, 255, 3)
    cv2.ellipse(img, (width // 2 - 80, height // 2 - 60), (30, 15), 0, 0, 360, 255, 2)
    cv2.ellipse(img, (width // 2 + 80, height // 2 - 60), (30, 15), 0, 0, 360, 255, 2)
    cv2.line(img, (width // 2, height // 2 - 20), (width // 2, height // 2 + 40), 255, 2)
    cv2.ellipse(img, (width // 2, height // 2 + 100), (50, 20), 0, 0, 180, 255, 2)
    return Image.fromarray(img).convert("RGB")


def _create_test_depth(width=1024, height=1024):
    import numpy as np
    from PIL import Image
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2, height / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    depth = (1.0 - dist / max_dist) * 255
    depth = depth.clip(0, 255).astype(np.uint8)
    return Image.fromarray(depth).convert("RGB")


def _create_test_face_photo(width=512, height=512):
    import numpy as np
    from PIL import Image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 180
    img[:, :, 1] = 140
    img[:, :, 2] = 120
    img[:, ::2, :] = [190, 150, 130]
    return Image.fromarray(img)


@app.cls(
    gpu=MODAL_GPU,
    image=container_image,
    volumes={MODAL_HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=MODAL_CONTAINER_TIMEOUT,
)
class Auditor:

    @modal.enter()
    def setup(self):
        import logging
        logging.basicConfig(level=logging.INFO)
        from backend.services.model_registry import ModelRegistry
        self.registry = ModelRegistry()
        self.registry.load_base_pipeline()
        self.registry._load_scribble_controlnet()
        self.registry._load_depth_controlnet()

    @modal.method()
    def run_full_audit(self) -> str:
        import time
        import torch
        import numpy as np
        from PIL import Image
        from backend.services.model_registry import ModelRegistry

        report = {}

        # ==================================================================
        # SECTION 0: GPU & Model Configuration
        # ==================================================================
        gpu_info = self.registry.gpu_info()
        gpu_info["torch_version"] = torch.__version__
        gpu_info["cuda_version"] = torch.version.cuda
        gpu_info["device_name_direct"] = torch.cuda.get_device_name(0)

        pipe_base = self.registry._base_pipe
        scheduler_config = pipe_base.scheduler.config
        report["gpu_info"] = gpu_info
        report["model_config"] = {
            "sdxl_model_id": "stabilityai/stable-diffusion-xl-base-1.0",
            "scheduler_class": pipe_base.scheduler.__class__.__name__,
            "scheduler_config": {
                k: str(v) for k, v in dict(scheduler_config).items()
            },
            "dtype": str(pipe_base.dtype),
            "unet_dtype": str(pipe_base.unet.dtype),
            "vae_dtype": str(pipe_base.vae.dtype),
            "unet_id": id(pipe_base.unet),
        }
        print(f"GPU: {gpu_info['device_name_direct']}")
        print(f"Scheduler: {pipe_base.scheduler.__class__.__name__}")
        print(f"Pipeline dtype: {pipe_base.dtype}")

        # ==================================================================
        # SECTION 1: Accurate Inference Timing
        # ==================================================================
        print("\n" + "=" * 70)
        print("SECTION 1: ACCURATE INFERENCE TIMING")
        print("=" * 70)

        PROMPT = "A detailed portrait of a young man with short brown hair, neutral expression, realistic, high quality photograph"
        NEG = "low quality, blurry, deformed, cartoon, anime"
        SEED = 42
        STEPS = 20
        GUIDANCE = 7.5
        W, H = 1024, 1024

        timing_results = {}

        # --- Text-only ---
        pipe, cn_count = self.registry.get_generation_pipeline()
        gen = torch.Generator(device="cuda").manual_seed(SEED)

        # Warm-up call (first call compiles CUDA kernels)
        torch.cuda.synchronize()
        _ = pipe(prompt="warmup", num_inference_steps=1, width=512, height=512,
                 generator=torch.Generator(device="cuda").manual_seed(0))
        torch.cuda.synchronize()

        # Timed call with CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_text_only_1 = pipe(
            prompt=PROMPT, negative_prompt=NEG,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()

        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000

        timing_results["text_only"] = {
            "steps": STEPS, "guidance_scale": GUIDANCE,
            "resolution": f"{W}x{H}", "seed": SEED,
            "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
        }
        img_text_only_1 = result_text_only_1.images[0]
        print(f"Text-only: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        # --- Text + Sketch ---
        pipe_sk, _ = self.registry.get_generation_pipeline(use_sketch=True)
        sketch = _create_test_sketch()

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_sketch = pipe_sk(
            prompt=PROMPT, negative_prompt=NEG, image=sketch,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000
        timing_results["text_sketch"] = {
            "steps": STEPS, "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
            "controlnet": "scribble", "cn_scale": 0.8,
        }
        img_sketch = result_sketch.images[0]
        print(f"Text+Sketch: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        # --- Text + Depth ---
        pipe_dp, _ = self.registry.get_generation_pipeline(use_depth=True)
        depth = _create_test_depth()

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_depth = pipe_dp(
            prompt=PROMPT, negative_prompt=NEG, image=depth,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000
        timing_results["text_depth"] = {
            "steps": STEPS, "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
            "controlnet": "depth", "cn_scale": 0.5,
        }
        img_depth = result_depth.images[0]
        print(f"Text+Depth: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        # --- Text + Photo (IP-Adapter) ---
        pipe_ip, _ = self.registry.get_generation_pipeline(use_ip_adapter=True, ip_adapter_scale=0.6)
        face = _create_test_face_photo()

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_ip = pipe_ip(
            prompt=PROMPT, negative_prompt=NEG, ip_adapter_image=face,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000
        timing_results["text_photo"] = {
            "steps": STEPS, "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
            "ip_adapter_scale": 0.6,
        }
        img_photo = result_ip.images[0]
        print(f"Text+Photo: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        # --- Text + Sketch + Depth (Multi-CN) ---
        pipe_multi, _ = self.registry.get_generation_pipeline(use_sketch=True, use_depth=True)

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_multi = pipe_multi(
            prompt=PROMPT, negative_prompt=NEG,
            image=[sketch, depth],
            controlnet_conditioning_scale=[0.8, 0.5],
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000
        timing_results["text_sketch_depth"] = {
            "steps": STEPS, "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
        }
        img_multi = result_multi.images[0]
        print(f"Text+Sketch+Depth: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        # --- All modalities ---
        pipe_all, _ = self.registry.get_generation_pipeline(
            use_sketch=True, use_depth=True, use_ip_adapter=True, ip_adapter_scale=0.6)

        gen = torch.Generator(device="cuda").manual_seed(SEED)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_t0 = time.perf_counter()
        start_event.record()

        result_all = pipe_all(
            prompt=PROMPT, negative_prompt=NEG,
            image=[sketch, depth], ip_adapter_image=face,
            controlnet_conditioning_scale=[0.8, 0.5],
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            width=W, height=H, generator=gen,
        )

        end_event.record()
        torch.cuda.synchronize()
        wall_t1 = time.perf_counter()
        cuda_ms = start_event.elapsed_time(end_event)
        wall_ms = (wall_t1 - wall_t0) * 1000
        timing_results["all_modalities"] = {
            "steps": STEPS, "cuda_event_ms": round(cuda_ms, 1),
            "wall_clock_ms": round(wall_ms, 1),
            "ms_per_step_cuda": round(cuda_ms / STEPS, 1),
        }
        img_all = result_all.images[0]
        print(f"All modalities: CUDA={cuda_ms:.0f}ms  Wall={wall_ms:.0f}ms  ({cuda_ms/STEPS:.0f}ms/step)")

        report["timing"] = timing_results

        # ==================================================================
        # SECTION 2: IP-Adapter Lifecycle / State Contamination
        # ==================================================================
        print("\n" + "=" * 70)
        print("SECTION 2: IP-ADAPTER LIFECYCLE & STATE CONTAMINATION")
        print("=" * 70)

        lifecycle = {}

        # Step A: Text-only (baseline) — seed 7777
        LSEED = 7777
        LPROMPT = "A detailed portrait of a woman with red hair, realistic"

        pipe_a, _ = self.registry.get_generation_pipeline()
        unet_id_a = id(pipe_a.unet)
        # Check attention processor types
        attn_types_before = set(type(v).__name__ for v in pipe_a.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_a = pipe_a(prompt=LPROMPT, negative_prompt=NEG,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_a = res_a.images[0]
        px_a = np.array(img_a).astype(np.float64)

        lifecycle["step_A_text_only_baseline"] = {
            "unet_id": unet_id_a,
            "attn_processor_types": sorted(attn_types_before),
            "pixel_mean": round(float(px_a.mean()), 2),
        }
        print(f"Step A: text-only baseline. UNet id={unet_id_a}, attn_procs={sorted(attn_types_before)}")

        # Step B: Text + IP-Adapter
        pipe_b, _ = self.registry.get_generation_pipeline(use_ip_adapter=True, ip_adapter_scale=0.6)
        unet_id_b = id(pipe_b.unet)
        attn_types_ip = set(type(v).__name__ for v in pipe_b.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_b = pipe_b(prompt=LPROMPT, negative_prompt=NEG,
                       ip_adapter_image=face,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_b = res_b.images[0]
        px_b = np.array(img_b).astype(np.float64)

        lifecycle["step_B_text_ip_adapter"] = {
            "unet_id": unet_id_b,
            "same_unet_as_A": unet_id_b == unet_id_a,
            "attn_processor_types": sorted(attn_types_ip),
            "pixel_mean": round(float(px_b.mean()), 2),
        }
        print(f"Step B: text+IP-Adapter. UNet id={unet_id_b} (same={unet_id_b==unet_id_a}), attn_procs={sorted(attn_types_ip)}")

        # Step C: Text-only again (check for contamination)
        pipe_c, _ = self.registry.get_generation_pipeline()
        unet_id_c = id(pipe_c.unet)
        attn_types_after_unload = set(type(v).__name__ for v in pipe_c.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_c = pipe_c(prompt=LPROMPT, negative_prompt=NEG,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_c = res_c.images[0]
        px_c = np.array(img_c).astype(np.float64)

        pixel_diff_a_c = float(np.abs(px_a - px_c).mean())
        images_match = pixel_diff_a_c < 1.0  # should be identical if no contamination

        lifecycle["step_C_text_only_after_ip"] = {
            "unet_id": unet_id_c,
            "attn_processor_types": sorted(attn_types_after_unload),
            "attn_procs_match_baseline": attn_types_after_unload == attn_types_before,
            "pixel_mean": round(float(px_c.mean()), 2),
            "pixel_diff_vs_step_A": round(pixel_diff_a_c, 4),
            "images_match_baseline": images_match,
        }
        print(f"Step C: text-only after IP unload. attn_match={attn_types_after_unload == attn_types_before}, "
              f"pixel_diff_vs_A={pixel_diff_a_c:.4f}, match={images_match}")

        # Step D: Text + Scribble CN
        pipe_d, _ = self.registry.get_generation_pipeline(use_sketch=True)
        unet_id_d = id(pipe_d.unet)
        attn_types_d = set(type(v).__name__ for v in pipe_d.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_d = pipe_d(prompt=LPROMPT, negative_prompt=NEG, image=sketch,
                       controlnet_conditioning_scale=0.8,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_d = res_d.images[0]

        lifecycle["step_D_text_sketch"] = {
            "unet_id": unet_id_d,
            "same_unet_as_A": unet_id_d == unet_id_a,
            "attn_processor_types": sorted(attn_types_d),
        }
        print(f"Step D: text+sketch CN. UNet id={unet_id_d} (same={unet_id_d==unet_id_a})")

        # Step E: Text + Sketch + IP-Adapter
        pipe_e, _ = self.registry.get_generation_pipeline(use_sketch=True, use_ip_adapter=True, ip_adapter_scale=0.6)
        unet_id_e = id(pipe_e.unet)
        attn_types_e = set(type(v).__name__ for v in pipe_e.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_e = pipe_e(prompt=LPROMPT, negative_prompt=NEG,
                       image=sketch, ip_adapter_image=face,
                       controlnet_conditioning_scale=0.8,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_e = res_e.images[0]

        lifecycle["step_E_sketch_ip"] = {
            "unet_id": unet_id_e,
            "same_unet_as_A": unet_id_e == unet_id_a,
            "attn_processor_types": sorted(attn_types_e),
        }
        print(f"Step E: sketch+IP. UNet id={unet_id_e} (same={unet_id_e==unet_id_a}), attn_procs={sorted(attn_types_e)}")

        # Step F: Unload IP-Adapter
        self.registry._remove_ip_adapter_if_loaded()
        attn_types_f = set(type(v).__name__ for v in self.registry._base_pipe.unet.attn_processors.values())
        lifecycle["step_F_after_unload"] = {
            "attn_processor_types": sorted(attn_types_f),
            "attn_procs_match_baseline": attn_types_f == attn_types_before,
        }
        print(f"Step F: after unload. attn_match_baseline={attn_types_f == attn_types_before}, procs={sorted(attn_types_f)}")

        # Step G: Text-only again (final contamination check)
        pipe_g, _ = self.registry.get_generation_pipeline()
        attn_types_g = set(type(v).__name__ for v in pipe_g.unet.attn_processors.values())

        gen = torch.Generator(device="cuda").manual_seed(LSEED)
        torch.cuda.synchronize()
        res_g = pipe_g(prompt=LPROMPT, negative_prompt=NEG,
                       num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                       width=W, height=H, generator=gen)
        torch.cuda.synchronize()
        img_g = res_g.images[0]
        px_g = np.array(img_g).astype(np.float64)
        pixel_diff_a_g = float(np.abs(px_a - px_g).mean())

        lifecycle["step_G_final_text_only"] = {
            "attn_processor_types": sorted(attn_types_g),
            "attn_procs_match_baseline": attn_types_g == attn_types_before,
            "pixel_diff_vs_step_A": round(pixel_diff_a_g, 4),
            "images_match_baseline": pixel_diff_a_g < 1.0,
        }
        print(f"Step G: final text-only. attn_match={attn_types_g == attn_types_before}, "
              f"pixel_diff_vs_A={pixel_diff_a_g:.4f}, match={pixel_diff_a_g < 1.0}")

        # Concurrency note
        lifecycle["concurrency_analysis"] = (
            "The ModelRegistry is NOT thread-safe. The IP-Adapter load/unload mutates "
            "the shared UNet's attention processors in-place. Concurrent requests that "
            "require different modality combinations would race on this mutable state. "
            "For the prototype (single-user, synchronous requests), this is acceptable. "
            "For production, a per-request pipeline copy or locking mechanism would be needed."
        )

        report["ip_adapter_lifecycle"] = lifecycle

        # ==================================================================
        # SECTION 3: Model Compatibility Verification
        # ==================================================================
        print("\n" + "=" * 70)
        print("SECTION 3: MODEL COMPATIBILITY")
        print("=" * 70)

        compat = {}

        # IP-Adapter checkpoint info
        from backend.config import (
            IP_ADAPTER_REPO, IP_ADAPTER_SUBFOLDER, IP_ADAPTER_WEIGHT_NAME,
            IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
            CONTROLNET_SCRIBBLE_ID, CONTROLNET_DEPTH_ID,
        )
        compat["ip_adapter"] = {
            "repo": IP_ADAPTER_REPO,
            "subfolder": IP_ADAPTER_SUBFOLDER,
            "weight_name": IP_ADAPTER_WEIGHT_NAME,
            "image_encoder_subfolder": IP_ADAPTER_IMAGE_ENCODER_SUBFOLDER,
        }

        # Image encoder info
        if self.registry._image_encoder is not None:
            enc = self.registry._image_encoder
            compat["image_encoder"] = {
                "class": enc.__class__.__name__,
                "config_model_type": getattr(enc.config, "model_type", "unknown"),
                "hidden_size": getattr(enc.config, "hidden_size", "unknown"),
                "image_size": getattr(enc.config, "image_size", "unknown"),
                "dtype": str(enc.dtype),
            }
            print(f"Image encoder: {enc.__class__.__name__}, "
                  f"hidden_size={getattr(enc.config, 'hidden_size', '?')}, "
                  f"image_size={getattr(enc.config, 'image_size', '?')}")

        # Feature extractor info — reload IP-Adapter to inspect
        pipe_check, _ = self.registry.get_generation_pipeline(use_ip_adapter=True, ip_adapter_scale=0.6)
        if pipe_check.feature_extractor is not None:
            fe = pipe_check.feature_extractor
            compat["feature_extractor"] = {
                "class": fe.__class__.__name__,
                "size": getattr(fe, "size", "unknown"),
                "crop_size": getattr(fe, "crop_size", "unknown"),
                "do_resize": getattr(fe, "do_resize", "unknown"),
                "do_normalize": getattr(fe, "do_normalize", "unknown"),
            }
            print(f"Feature extractor: {fe.__class__.__name__}, size={getattr(fe, 'size', '?')}")

        # ControlNet info
        compat["controlnets"] = {
            "scribble": {
                "id": CONTROLNET_SCRIBBLE_ID,
                "dtype": str(self.registry._cn_scribble.dtype),
                "class": self.registry._cn_scribble.__class__.__name__,
            },
            "depth": {
                "id": CONTROLNET_DEPTH_ID,
                "dtype": str(self.registry._cn_depth.dtype),
                "class": self.registry._cn_depth.__class__.__name__,
            },
        }
        print(f"Scribble CN: {CONTROLNET_SCRIBBLE_ID} ({self.registry._cn_scribble.dtype})")
        print(f"Depth CN: {CONTROLNET_DEPTH_ID} ({self.registry._cn_depth.dtype})")

        # Unload IP-Adapter for clean state
        self.registry._remove_ip_adapter_if_loaded()

        report["compatibility"] = compat

        # ==================================================================
        # SECTION 4: Collect Images as base64
        # ==================================================================
        print("\n" + "=" * 70)
        print("SECTION 4: COLLECTING OUTPUT IMAGES")
        print("=" * 70)

        images_b64 = {}
        # Conditioning inputs
        images_b64["input_sketch"] = _img_to_b64(sketch)
        images_b64["input_depth"] = _img_to_b64(depth)
        images_b64["input_face_photo"] = _img_to_b64(face)
        # Generated outputs
        images_b64["out_text_only"] = _img_to_b64(img_text_only_1)
        images_b64["out_text_sketch"] = _img_to_b64(img_sketch)
        images_b64["out_text_depth"] = _img_to_b64(img_depth)
        images_b64["out_text_photo"] = _img_to_b64(img_photo)
        images_b64["out_text_sketch_depth"] = _img_to_b64(img_multi)
        images_b64["out_all_modalities"] = _img_to_b64(img_all)
        # Lifecycle images
        images_b64["lifecycle_A_text_baseline"] = _img_to_b64(img_a)
        images_b64["lifecycle_B_text_ip"] = _img_to_b64(img_b)
        images_b64["lifecycle_C_text_after_ip"] = _img_to_b64(img_c)
        images_b64["lifecycle_G_final_text"] = _img_to_b64(img_g)

        report["images_b64"] = images_b64
        print(f"Collected {len(images_b64)} images")

        # ==================================================================
        # SECTION 5: VRAM usage
        # ==================================================================
        report["final_vram"] = {
            "allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
            "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 2),
        }
        print(f"\nVRAM: allocated={report['final_vram']['allocated_gb']}GB, "
              f"max={report['final_vram']['max_allocated_gb']}GB")

        # Save images to volume for local download
        import os
        audit_vol_dir = "/root/.cache/huggingface/audit_outputs"
        os.makedirs(audit_vol_dir, exist_ok=True)

        images_b64 = report.pop("images_b64", {})
        for name, b64_data in images_b64.items():
            img_path = os.path.join(audit_vol_dir, f"{name}.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            print(f"  Saved to volume: {img_path}")

        # Save JSON report to volume
        report_path = os.path.join(audit_vol_dir, "audit_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report saved to volume: {report_path}")

        hf_cache_vol.commit()

        # Return plain JSON string (no torch objects)
        return json.dumps(report, indent=2, default=str)


@app.local_entrypoint()
def run_audit():
    import os
    import subprocess

    print("🔍 Starting Stage 2 Verification Audit on Modal GPU...")
    auditor = Auditor()
    report_json = auditor.run_full_audit.remote()

    # Parse the report
    report = json.loads(report_json)

    # Save JSON report locally
    out_dir = os.path.join(os.path.dirname(__file__), "..", "audit_outputs")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(json.dumps(report, indent=2))

    print("\n📥 Now download images from volume with:")
    print("  modal volume get hf-model-cache audit_outputs/ audit_outputs/")

