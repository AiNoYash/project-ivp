"""
Modal application – Stage 2: Model integration tests.

This file defines the Modal app, container image, and integration test
functions that verify every modality combination works with the selected
models before building the full API.

Run with:
    modal run backend/modal_app.py

For development serving (hot-reload):
    modal serve backend/modal_app.py
"""

import modal

# ---------------------------------------------------------------------------
# Configuration from single source of truth
# ---------------------------------------------------------------------------
from backend.config import (
    MODAL_GPU,
    MODAL_VOLUME_NAME,
    MODAL_HF_CACHE_PATH,
    MODAL_SECRET_NAME,
    MODAL_APP_NAME,
    MODAL_CONTAINER_TIMEOUT,
)

# ---------------------------------------------------------------------------
# Modal Volume & App
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

app = modal.App(MODAL_APP_NAME)

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
container_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "torch==2.4.1",
        "torchvision==0.19.1",
        # Diffusers ecosystem – pinned for compatibility
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        # Image processing
        "Pillow==10.4.0",
        "opencv-python-headless==4.10.0.84",
        "numpy==1.26.4",
        # Web
        "fastapi[standard]==0.112.4",
        "python-multipart==0.0.9",
        "uvicorn==0.30.6",
        # Hugging Face Hub
        "huggingface-hub==0.24.7",
        # IP-Adapter needs the image feature extractor
        "ftfy==6.2.3",
    )
    .env({"HF_HOME": MODAL_HF_CACHE_PATH})
    .add_local_python_source("backend")
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _create_test_sketch(width: int = 1024, height: int = 1024):
    """Create a simple white-on-black scribble sketch for testing."""
    import numpy as np
    from PIL import Image

    img = np.zeros((height, width), dtype=np.uint8)
    # Draw some simple lines to simulate a face outline
    import cv2
    # Oval for face
    cv2.ellipse(img, (width // 2, height // 2), (200, 280), 0, 0, 360, 255, 2)
    # Eyes
    cv2.ellipse(img, (width // 2 - 80, height // 2 - 60), (30, 15), 0, 0, 360, 255, 2)
    cv2.ellipse(img, (width // 2 + 80, height // 2 - 60), (30, 15), 0, 0, 360, 255, 2)
    # Nose
    cv2.line(img, (width // 2, height // 2 - 20), (width // 2, height // 2 + 40), 255, 2)
    # Mouth
    cv2.ellipse(img, (width // 2, height // 2 + 100), (50, 20), 0, 0, 180, 255, 2)

    return Image.fromarray(img).convert("RGB")


def _create_test_depth(width: int = 1024, height: int = 1024):
    """Create a simple synthetic depth map for testing (gradient)."""
    import numpy as np
    from PIL import Image

    # Simulate face-like depth: brighter in center (closer), darker at edges
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2, height / 2
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    # Closer pixels (center) = brighter = higher value
    depth = (1.0 - dist / max_dist) * 255
    depth = depth.clip(0, 255).astype(np.uint8)

    return Image.fromarray(depth).convert("RGB")


def _create_test_face_photo(width: int = 512, height: int = 512):
    """Create a minimal test 'face photo' – a colored rectangle placeholder."""
    from PIL import Image
    import numpy as np
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 180  # Reddish skin-tone-ish
    img[:, :, 1] = 140
    img[:, :, 2] = 120
    # Add some stripe variation with numpy slicing
    img[:, ::2, :] = [190, 150, 130]
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Integration test class – runs on GPU
# ---------------------------------------------------------------------------

@app.cls(
    gpu=MODAL_GPU,
    image=container_image,
    volumes={MODAL_HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
    timeout=MODAL_CONTAINER_TIMEOUT,
)
class IntegrationTester:
    """Loads models once, then runs multiple test methods."""

    @modal.enter()
    def setup(self):
        """Load model registry on container start."""
        import logging
        logging.basicConfig(level=logging.INFO)

        from backend.services.model_registry import ModelRegistry
        self.registry = ModelRegistry()
        self.registry.load_base_pipeline()
        self.registry._load_scribble_controlnet()
        self.registry._load_depth_controlnet()
        print("✅ Base pipeline and ControlNets loaded")

    @modal.method()
    def test_text_only(self) -> dict:
        """Test 1: Text-only generation (no ControlNet, no IP-Adapter)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 1: Text-only generation")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=False, use_depth=False, use_ip_adapter=False
        )
        assert cn_count == 0, f"Expected 0 ControlNets, got {cn_count}"

        prompt = "A detailed portrait of a young man with short brown hair, realistic, high quality"
        seed = 42
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        assert image.size == (1024, 1024), f"Unexpected size: {image.size}"

        output_path = "/tmp/test1_text_only.png"
        image.save(output_path)
        print(f"✅ Text-only generation passed in {elapsed:.1f}s")
        print(f"   Image saved to {output_path}")

        return {
            "test": "text_only",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_sketch_controlnet(self) -> dict:
        """Test 2: Text + sketch (Scribble ControlNet only)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 2: Text + Sketch (Scribble ControlNet)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=True, use_depth=False, use_ip_adapter=False
        )
        assert cn_count == 1, f"Expected 1 ControlNet, got {cn_count}"

        sketch = _create_test_sketch()
        sketch.save("/tmp/test2_input_sketch.png")

        prompt = "A detailed portrait of a woman with long black hair, realistic, high quality"
        seed = 123
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            image=sketch,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test2_sketch_cn.png"
        image.save(output_path)
        print(f"✅ Sketch ControlNet generation passed in {elapsed:.1f}s")

        return {
            "test": "sketch_controlnet",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_depth_controlnet(self) -> dict:
        """Test 3: Text + depth map (Depth ControlNet only)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 3: Text + Depth Map (Depth ControlNet)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=False, use_depth=True, use_ip_adapter=False
        )
        assert cn_count == 1, f"Expected 1 ControlNet, got {cn_count}"

        depth = _create_test_depth()
        depth.save("/tmp/test3_input_depth.png")

        prompt = "A detailed portrait of an elderly man with grey beard, realistic, high quality"
        seed = 456
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            image=depth,
            controlnet_conditioning_scale=0.5,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test3_depth_cn.png"
        image.save(output_path)
        print(f"✅ Depth ControlNet generation passed in {elapsed:.1f}s")

        return {
            "test": "depth_controlnet",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_multi_controlnet(self) -> dict:
        """Test 5: Text + sketch + depth (Multi-ControlNet)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 5: Text + Sketch + Depth (Multi-ControlNet)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=True, use_depth=True, use_ip_adapter=False
        )
        assert cn_count == 2, f"Expected 2 ControlNets, got {cn_count}"

        sketch = _create_test_sketch()
        depth = _create_test_depth()

        prompt = "A detailed portrait of a middle-aged woman, realistic, high quality"
        seed = 789
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            image=[sketch, depth],
            controlnet_conditioning_scale=[0.8, 0.5],
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test5_multi_cn.png"
        image.save(output_path)
        print(f"✅ Multi-ControlNet generation passed in {elapsed:.1f}s")

        return {
            "test": "multi_controlnet",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_ip_adapter_only(self) -> dict:
        """Test 4: Text + photograph (IP-Adapter only, no ControlNet)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 4: Text + Photograph (IP-Adapter Plus Face)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=False, use_depth=False, use_ip_adapter=True,
            ip_adapter_scale=0.6,
        )
        assert cn_count == 0, f"Expected 0 ControlNets, got {cn_count}"

        face_photo = _create_test_face_photo()
        face_photo.save("/tmp/test4_input_face.png")

        prompt = "A detailed portrait of a person, realistic, high quality"
        seed = 1111
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            ip_adapter_image=face_photo,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test4_ip_adapter.png"
        image.save(output_path)
        print(f"✅ IP-Adapter generation passed in {elapsed:.1f}s")

        return {
            "test": "ip_adapter_only",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_ip_adapter_with_sketch(self) -> dict:
        """Test: Text + photograph + sketch (IP-Adapter + Scribble ControlNet)."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST: Text + Photo + Sketch (IP-Adapter + Scribble CN)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=True, use_depth=False, use_ip_adapter=True,
            ip_adapter_scale=0.6,
        )
        assert cn_count == 1

        sketch = _create_test_sketch()
        face_photo = _create_test_face_photo()

        prompt = "A detailed portrait of a person, realistic, high quality"
        seed = 2222
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            image=sketch,
            ip_adapter_image=face_photo,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test_ip_sketch.png"
        image.save(output_path)
        print(f"✅ IP-Adapter + Sketch ControlNet passed in {elapsed:.1f}s")

        return {
            "test": "ip_adapter_with_sketch",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_all_modalities(self) -> dict:
        """Test 6: All inputs – Text + photo + sketch + depth."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 6: All Modalities (IP-Adapter + Multi-ControlNet)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=True, use_depth=True, use_ip_adapter=True,
            ip_adapter_scale=0.6,
        )
        assert cn_count == 2

        sketch = _create_test_sketch()
        depth = _create_test_depth()
        face_photo = _create_test_face_photo()

        prompt = "A detailed portrait of a person with distinct features, realistic, high quality"
        seed = 3333
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            image=[sketch, depth],
            ip_adapter_image=face_photo,
            controlnet_conditioning_scale=[0.8, 0.5],
            num_inference_steps=20,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test6_all_modalities.png"
        image.save(output_path)
        print(f"✅ All modalities generation passed in {elapsed:.1f}s")

        return {
            "test": "all_modalities",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }

    @modal.method()
    def test_text_only_after_controlnet(self) -> dict:
        """Test 9 (partial): Re-run text-only after using ControlNet to verify switching works."""
        import time
        import torch

        print("\n" + "=" * 60)
        print("TEST 9: Text-only after ControlNet (pipeline switching)")
        print("=" * 60)

        pipe, cn_count = self.registry.get_generation_pipeline(
            use_sketch=False, use_depth=False, use_ip_adapter=False
        )
        assert cn_count == 0

        prompt = "A detailed portrait of a teenager, realistic, high quality"
        seed = 9999
        generator = torch.Generator(device=self.registry.device).manual_seed(seed)

        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed",
            num_inference_steps=15,
            guidance_scale=7.5,
            width=1024,
            height=1024,
            generator=generator,
        )
        elapsed = time.time() - t0

        image = result.images[0]
        output_path = "/tmp/test9_text_after_cn.png"
        image.save(output_path)
        print(f"✅ Text-only after CN switching passed in {elapsed:.1f}s")

        return {
            "test": "text_only_after_controlnet",
            "status": "passed",
            "time_seconds": round(elapsed, 1),
            "image_size": list(image.size),
            "seed": seed,
        }


# ---------------------------------------------------------------------------
# Local entrypoint – runs all tests sequentially
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_integration_tests():
    """Run all integration tests sequentially via a single warm container."""
    import json

    print("🚀 Starting model integration tests on Modal GPU...")
    print(f"   GPU: {MODAL_GPU}")
    print(f"   Volume: {MODAL_VOLUME_NAME}")
    print()

    tester = IntegrationTester()
    results = []

    # Run tests in order of complexity
    test_methods = [
        ("Test 1: Text-only", tester.test_text_only),
        ("Test 2: Text + Sketch", tester.test_sketch_controlnet),
        ("Test 3: Text + Depth", tester.test_depth_controlnet),
        ("Test 5: Multi-ControlNet", tester.test_multi_controlnet),
        ("Test 4: IP-Adapter only", tester.test_ip_adapter_only),
        ("Test: IP-Adapter + Sketch", tester.test_ip_adapter_with_sketch),
        ("Test 6: All modalities", tester.test_all_modalities),
        ("Test 9: Text after CN switch", tester.test_text_only_after_controlnet),
    ]

    for name, method in test_methods:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        try:
            result = method.remote()
            result["name"] = name
            results.append(result)
            print(f"  ✅ {name}: PASSED ({result['time_seconds']}s)")
        except Exception as e:
            results.append({"name": name, "status": "failed", "error": str(e)})
            print(f"  ❌ {name}: FAILED – {e}")

    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r.get("status") == "passed")
    failed = sum(1 for r in results if r.get("status") == "failed")
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Failed: {failed}/{len(results)}")
    for r in results:
        status = "✅" if r.get("status") == "passed" else "❌"
        time_info = f" ({r['time_seconds']}s)" if "time_seconds" in r else ""
        error_info = f" – {r['error']}" if "error" in r else ""
        print(f"  {status} {r['name']}{time_info}{error_info}")

    print("\n" + json.dumps(results, indent=2))
