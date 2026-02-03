"""
vLLM Runner Service.

Runs VLM inference using a vLLM server for faster inference.
Requires vLLM server to be running separately.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Image resolution presets (same as vlm_runner.py)
# These control how images are resized before sending to vLLM
IMAGE_RESOLUTION_PRESETS = {
    "micro": {"max_pixels": 100352},   # ~316x316, fastest
    "tiny": {"max_pixels": 200704},    # ~448x448
    "small": {"max_pixels": 401408},   # ~634x634, recommended
    "default": {"max_pixels": None},   # No resize, slowest
}


@dataclass
class VLLMRunResult:
    """Result from running vLLM inference."""
    model: str
    raw_response: str
    parsed_output: dict
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0


class VLLMRunner:
    """Run VLM inference using vLLM server with OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
    ):
        """Initialize vLLM runner.

        Args:
            base_url: vLLM server URL (default: http://localhost:8000)
            timeout: Request timeout in seconds
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._current_model: Optional[str] = None

    def is_available(self) -> bool:
        """Check if vLLM server is running and accessible."""
        try:
            response = self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def get_models(self) -> list[str]:
        """Get list of available models from vLLM server."""
        try:
            response = self._client.get(f"{self._base_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning(f"Failed to get models from vLLM: {e}")
        return []

    def warm_cache(self, system_prompts: list[str]) -> bool:
        """Warm up KV cache by sending dummy requests for each system prompt.

        This pre-fills the prefix cache so subsequent requests with the same
        system prompts are faster (~2x speedup on first real request).

        Args:
            system_prompts: List of system prompts to warm up (e.g., planning, monitoring)

        Returns:
            True if cache warm-up succeeded for all prompts
        """
        if not self.is_available():
            logger.warning("vLLM server not available, skipping cache warm-up")
            return False

        models = self.get_models()
        if not models:
            logger.warning("No models available, skipping cache warm-up")
            return False

        model = models[0]
        success = True

        for i, system_prompt in enumerate(system_prompts):
            try:
                logger.info(f"Warming cache for prompt {i+1}/{len(system_prompts)}...")
                request_data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Ready."}
                    ],
                    "max_tokens": 1,  # Minimal output, just populate cache
                    "temperature": 0.0,
                }
                response = self._client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=request_data,
                )
                if response.status_code != 200:
                    logger.warning(f"Cache warm-up failed for prompt {i+1}: {response.status_code}")
                    success = False
            except Exception as e:
                logger.warning(f"Cache warm-up error for prompt {i+1}: {e}")
                success = False

        if success:
            logger.info(f"KV cache warmed for {len(system_prompts)} prompts")
        return success

    def run(
        self,
        model: str,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images: dict[str, Any],
        context: Optional[dict] = None,
        max_tokens: int = 256,  # Reduced from 512 for faster inference
        temperature: float = 0.0,  # Greedy by default
        image_resolution: str = "small",  # Image resolution preset
    ) -> VLLMRunResult:
        """Run VLM inference via vLLM server.

        Args:
            model: Model name hint (will auto-detect from server if not exact match)
            mode: "planning" or "monitoring"
            system_prompt: System prompt text
            user_prompt: User prompt text
            images: Dict of image paths or base64 encoded images
            context: Optional context dict
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            image_resolution: Image resolution preset ("micro", "tiny", "small", "default")

        Returns:
            VLLMRunResult with response and parsed output
        """
        start_time = time.time()

        # Auto-detect model from server (vLLM typically serves one model at a time)
        # This allows the UI to work with both BF16 and FP8 modes without knowing the exact name
        available_models = self.get_models()
        if available_models:
            # Use the first available model (vLLM serves one model at a time)
            actual_model = available_models[0]
            if actual_model != model:
                logger.info(f"Using vLLM model: {actual_model} (requested: {model})")
            model = actual_model
        else:
            logger.warning(f"No models available from vLLM server, using requested: {model}")

        # Get resolution preset
        resolution_preset = IMAGE_RESOLUTION_PRESETS.get(image_resolution, IMAGE_RESOLUTION_PRESETS["small"])
        max_pixels = resolution_preset.get("max_pixels")

        try:
            # Build message content with images
            user_content = []

            # Add images (single image for both planning and monitoring)
            if mode == "monitoring":
                # Use single image - prefer "after" (current state) or "head"
                for key in ["after", "head", "before"]:
                    if key in images:
                        image_url = self._prepare_image(images[key], max_pixels=max_pixels)
                        if image_url:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                            break
            else:
                # Planning: single image
                for key in ["central", "head", "scene"]:
                    if key in images:
                        image_url = self._prepare_image(images[key], max_pixels=max_pixels)
                        if image_url:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                            break
                else:
                    # Try first available
                    for key, value in images.items():
                        image_url = self._prepare_image(value, max_pixels=max_pixels)
                        if image_url:
                            user_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })
                            break

            # Add text prompt
            user_content.append({"type": "text", "text": user_prompt})

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            # Make request to vLLM server
            request_data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            response = self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=request_data,
            )

            if response.status_code != 200:
                error_msg = f"vLLM server error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return VLLMRunResult(
                    model=model,
                    raw_response="",
                    parsed_output={},
                    success=False,
                    error=error_msg,
                    latency_ms=(time.time() - start_time) * 1000,
                )

            result = response.json()

            # Extract response
            raw_response = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Calculate timing
            latency_ms = (time.time() - start_time) * 1000
            tokens_per_second = (completion_tokens / (latency_ms / 1000)) if latency_ms > 0 else 0

            # Parse response
            parsed = self._parse_response(raw_response)

            logger.info(f"vLLM inference: {latency_ms:.0f}ms, "
                       f"{completion_tokens} tokens, {tokens_per_second:.1f} tok/s")

            return VLLMRunResult(
                model=model,
                raw_response=raw_response,
                parsed_output=parsed,
                success=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_second=tokens_per_second,
            )

        except httpx.ConnectError:
            error_msg = f"Cannot connect to vLLM server at {self._base_url}. Is it running?"
            logger.error(error_msg)
            return VLLMRunResult(
                model=model,
                raw_response="",
                parsed_output={},
                success=False,
                error=error_msg,
                latency_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"vLLM run error: {e}")
            import traceback
            traceback.print_exc()
            return VLLMRunResult(
                model=model,
                raw_response="",
                parsed_output={},
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _prepare_image(self, image: Any, max_pixels: Optional[int] = None) -> Optional[str]:
        """Convert image to base64 data URL for vLLM.

        Args:
            image: Image path, PIL Image, numpy array, or base64 string
            max_pixels: Maximum total pixels (width * height). Image will be resized if larger.
        """
        import io
        from PIL import Image
        import numpy as np

        project_root = Path(__file__).parent.parent.parent.parent

        try:
            pil_image = None

            if isinstance(image, Image.Image):
                pil_image = image
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            elif isinstance(image, str):
                # Handle API URL path
                if image.startswith("/api/images/"):
                    parts = image.replace("/api/images/", "").split("/")
                    if len(parts) == 2:
                        trace_id, filename = parts
                        actual_path = project_root / "logs" / "images" / trace_id / filename
                        if actual_path.exists():
                            pil_image = Image.open(actual_path)
                # Handle direct file path
                elif Path(image).exists():
                    pil_image = Image.open(image)
                # Handle base64 data URL (need to decode and resize)
                elif image.startswith("data:image"):
                    # Decode base64 to resize
                    import re
                    match = re.match(r'data:image/\w+;base64,(.+)', image)
                    if match:
                        img_data = base64.b64decode(match.group(1))
                        pil_image = Image.open(io.BytesIO(img_data))
                # Handle full URL
                elif image.startswith("http"):
                    from urllib.parse import urlparse
                    parsed = urlparse(image)
                    return self._prepare_image(parsed.path, max_pixels=max_pixels)

            if pil_image:
                # Resize image if needed
                if max_pixels is not None:
                    current_pixels = pil_image.width * pil_image.height
                    if current_pixels > max_pixels:
                        # Calculate scale factor to fit within max_pixels
                        scale = (max_pixels / current_pixels) ** 0.5
                        new_width = int(pil_image.width * scale)
                        new_height = int(pil_image.height * scale)
                        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                        logger.debug(f"Resized image from {current_pixels} to {new_width * new_height} pixels")

                # Convert to RGB if needed (removes alpha channel)
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    pil_image = pil_image.convert('RGB')

                # Convert to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG", quality=85)
                b64_data = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{b64_data}"

        except Exception as e:
            logger.warning(f"Failed to prepare image: {e}")

        return None

    def _parse_response(self, response: str) -> dict:
        """Parse VLM response to extract JSON."""
        import re

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Return raw response in a dict
        return {"raw": response}

    def unload(self) -> dict:
        """No-op for vLLM (model managed by server)."""
        return {"message": "vLLM models are managed by the server"}

    def list_available_models(self) -> list[str]:
        """List models available on vLLM server."""
        return self.get_models()
