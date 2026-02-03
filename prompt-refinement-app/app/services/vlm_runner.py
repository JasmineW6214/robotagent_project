"""
VLM Runner Service.

Runs VLM inference on inputs using Qwen3-VL models with custom prompts.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
import base64

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Detailed inference timing metrics."""
    total_ms: float = 0.0
    preprocessing_ms: float = 0.0
    generation_ms: float = 0.0
    postprocessing_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0


# Image resolution presets (pixels = height * width approximation)
# Smaller = fewer image tokens = faster inference
IMAGE_RESOLUTION_PRESETS = {
    "micro": {"min_pixels": 50176, "max_pixels": 100352},   # ~224x224 to ~316x316, fastest
    "tiny": {"min_pixels": 100352, "max_pixels": 200704},   # ~316x316 to ~448x448
    "small": {"min_pixels": 200704, "max_pixels": 401408},  # ~448x448 to ~634x634
    "default": {"min_pixels": None, "max_pixels": None},    # Model default, slowest
}


@dataclass
class VLMRunResult:
    """Result from running VLM inference."""
    model: str
    raw_response: str
    parsed_output: dict
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    metrics: Optional[InferenceMetrics] = None


class VLMRunner:
    """Run VLM inference using Qwen3-VL models with custom prompts."""

    def __init__(
        self,
        use_flash_attention: bool = True,
        use_compile: bool = False,
        image_resolution: str = "small",
    ):
        """Initialize VLM runner (lazy loading of models).

        Args:
            use_flash_attention: Enable Flash Attention 2 for faster inference.
            use_compile: Enable torch.compile() for optimized inference (slower first run).
            image_resolution: Image resolution preset for vision encoder.
                Options: "micro" (fastest), "tiny", "small" (recommended), "default" (slowest).
                Default "small" balances quality and speed (~220 tokens, ~3-5s).
        """
        self._model = None
        self._processor = None
        self._current_model = None
        self._use_flash_attention = use_flash_attention
        self._use_compile = use_compile
        self._is_compiled = False
        self._image_resolution = image_resolution
        self._resolution_preset = IMAGE_RESOLUTION_PRESETS.get(image_resolution, IMAGE_RESOLUTION_PRESETS["micro"])

    def _ensure_processor_loaded(self, model_path) -> None:
        """Ensure processor is loaded with current resolution settings."""
        from transformers import AutoProcessor

        processor_kwargs = {"trust_remote_code": True}
        if self._resolution_preset["min_pixels"] is not None:
            processor_kwargs["min_pixels"] = self._resolution_preset["min_pixels"]
        if self._resolution_preset["max_pixels"] is not None:
            processor_kwargs["max_pixels"] = self._resolution_preset["max_pixels"]

        self._processor = AutoProcessor.from_pretrained(str(model_path), **processor_kwargs)
        logger.info(f"Processor loaded - resolution: {self._image_resolution} "
                   f"(min={self._resolution_preset['min_pixels']}, max={self._resolution_preset['max_pixels']})")

    def _ensure_loaded(self, model: str = "4B-Instruct") -> None:
        """Ensure model is loaded.

        Args:
            model: Model variant (2B-Instruct, 4B-Instruct, 4B-Thinking)
        """
        model_path = project_root / "models" / "qwen3vl" / model

        # If same model is loaded, just check if processor needs reload
        if self._model is not None and self._current_model == model:
            if self._processor is None:
                self._ensure_processor_loaded(model_path)
            return

        # Unload current model if different
        if self._model is not None:
            logger.info(f"Unloading current model {self._current_model} before loading {model}")
            self.unload()

        logger.info(f"Loading VLM model: {model}")

        try:
            import torch
            from transformers import AutoProcessor

            # Try Qwen3VL first (transformers >= 4.57.0), fallback to Qwen2VL
            try:
                from transformers import Qwen3VLForConditionalGeneration as QwenVLModel
                logger.info("Using Qwen3VLForConditionalGeneration")
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
                logger.info("Falling back to Qwen2VLForConditionalGeneration")

            model_path = project_root / "models" / "qwen3vl" / model

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load processor with resolution settings
            self._ensure_processor_loaded(model_path)

            # Build model loading kwargs
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            # Enable Flash Attention 2 if requested and available
            if self._use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Flash Attention 2 enabled")
                except Exception as e:
                    logger.warning(f"Flash Attention 2 not available: {e}")

            self._model = QwenVLModel.from_pretrained(str(model_path), **model_kwargs)

            # Apply torch.compile if requested
            if self._use_compile and not self._is_compiled:
                logger.info("Applying torch.compile() - first inference will be slow...")
                self._model = torch.compile(self._model, mode="reduce-overhead")
                self._is_compiled = True
                logger.info("Model compiled successfully")

            self._current_model = model
            logger.info(f"VLM model loaded: {model}")
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise

    def run(
        self,
        model: str,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images: dict[str, Any],
        context: Optional[dict] = None,
        max_tokens: int = 128,
        disable_thinking: bool = False,
        use_sampling: bool = False,
        image_resolution: Optional[str] = None,
    ) -> VLMRunResult:
        """Run VLM inference with custom prompts.

        Args:
            model: Model variant (2B-Instruct, 4B-Instruct, 4B-Thinking)
            mode: "planning" or "monitoring"
            system_prompt: System prompt text
            user_prompt: User prompt text
            images: Dict of image paths or base64 encoded images
            context: Optional context dict (available_tasks, etc.)
            max_tokens: Maximum output tokens (default: 128 for concise output).
            disable_thinking: If True, prepend /no_think to disable thinking mode.
                            Useful when you want Thinking model to behave like Instruct.
            use_sampling: If True, use sampling (temperature/top_p). Default False (greedy).
                         Greedy decoding is faster and more deterministic.
            image_resolution: Override the default resolution preset.
                             "micro" = fastest (~120 image tokens),
                             "tiny" = fast (~220 tokens),
                             "small" = moderate,
                             "default" = full resolution (~300+ tokens, slow).

        Returns:
            VLMRunResult with response and parsed output
        """
        import torch
        import numpy as np
        from PIL import Image

        start_time = time.time()
        metrics = InferenceMetrics()

        try:
            # Check if resolution changed - only processor needs reload, not model
            if image_resolution and image_resolution != self._image_resolution:
                logger.info(f"Resolution changed: {self._image_resolution} -> {image_resolution}")
                self._image_resolution = image_resolution
                self._resolution_preset = IMAGE_RESOLUTION_PRESETS.get(
                    image_resolution, IMAGE_RESOLUTION_PRESETS["small"]
                )
                # Only reload processor, keep model in memory
                self._processor = None

            self._ensure_loaded(model)

            preprocess_start = time.time()

            # Load images as PIL
            pil_images = []
            image_content = []

            if mode == "monitoring":
                # Before/after images
                for key in ["before", "after"]:
                    if key in images:
                        pil_img = self._load_image_as_pil(images[key])
                        if pil_img:
                            pil_images.append(pil_img)
                            image_content.append({"type": "text", "text": f"{key.upper()} image:"})
                            image_content.append({"type": "image", "image": pil_img})
            else:
                # Planning: single image
                for key in ["central", "head", "scene"]:
                    if key in images:
                        pil_img = self._load_image_as_pil(images[key])
                        if pil_img:
                            pil_images.append(pil_img)
                            image_content.append({"type": "image", "image": pil_img})
                            break
                else:
                    # Try first available
                    for key, value in images.items():
                        pil_img = self._load_image_as_pil(value)
                        if pil_img:
                            pil_images.append(pil_img)
                            image_content.append({"type": "image", "image": pil_img})
                            break

            if not pil_images:
                raise ValueError("No valid images provided")

            # Handle thinking mode control
            is_thinking = "Thinking" in model
            final_user_prompt = user_prompt

            if is_thinking and disable_thinking:
                # Prepend /no_think to disable thinking mode
                final_user_prompt = "/no_think\n" + user_prompt
                logger.info("Thinking mode disabled via /no_think")

            # Add user prompt to content
            image_content.append({"type": "text", "text": final_user_prompt})

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": image_content}
            ]

            # Apply chat template
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self._processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            # Record preprocessing time and input tokens
            metrics.preprocessing_ms = (time.time() - preprocess_start) * 1000
            metrics.input_tokens = inputs["input_ids"].shape[1]

            # Generate response with mode-specific parameters
            # Default: GREEDY decoding for faster, deterministic output
            # Use use_sampling=True for temperature-based sampling

            # Use max_tokens directly from the request
            budget = max_tokens

            if use_sampling:
                # Sampling mode - use temperature/top_p/top_k
                # Based on official Qwen3-VL generation_config.json recommendations
                if mode == "monitoring":
                    temp = 0.5 if is_thinking else 0.3
                    top_p = 0.9 if is_thinking else 0.8
                else:
                    temp = 1.0 if is_thinking else 0.7
                    top_p = 0.95 if is_thinking else 0.8

                gen_kwargs = {
                    "max_new_tokens": budget,
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": 20,
                    "repetition_penalty": 1.0,
                    "do_sample": True,
                }
                logger.info(f"Sampling mode: temp={temp}, top_p={top_p}")
            else:
                # GREEDY decoding (default) - faster and deterministic
                gen_kwargs = {
                    "max_new_tokens": budget,
                    "do_sample": False,  # Greedy decoding
                }
                logger.info("Greedy decoding mode (do_sample=False)")

            if is_thinking:
                logger.info(f"Thinking model budget: {budget} tokens")

            # Generation with timing
            gen_start = time.time()
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )
            metrics.generation_ms = (time.time() - gen_start) * 1000

            # Decode response (skip input tokens)
            postprocess_start = time.time()
            input_len = inputs["input_ids"].shape[1]
            output_len = outputs[0].shape[0] - input_len
            metrics.output_tokens = output_len

            response = self._processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )

            # Parse response
            parsed = self._parse_response(response)
            metrics.postprocessing_ms = (time.time() - postprocess_start) * 1000

            # Calculate total and tokens/second
            metrics.total_ms = (time.time() - start_time) * 1000
            if metrics.generation_ms > 0:
                metrics.tokens_per_second = (output_len / metrics.generation_ms) * 1000

            logger.info(f"Inference metrics: total={metrics.total_ms:.0f}ms, "
                       f"gen={metrics.generation_ms:.0f}ms, "
                       f"tokens={output_len}, "
                       f"tok/s={metrics.tokens_per_second:.1f}")

            return VLMRunResult(
                model=model,
                raw_response=response,
                parsed_output=parsed,
                success=True,
                latency_ms=metrics.total_ms,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"VLM run error: {e}")
            import traceback
            traceback.print_exc()
            return VLMRunResult(
                model=model,
                raw_response="",
                parsed_output={},
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _load_image_as_pil(self, image: Any):
        """Load image from various formats to PIL Image."""
        import numpy as np
        from PIL import Image
        import io

        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, str):
            # Handle API URL path (e.g., /api/images/20260111_212709/file.jpg)
            if image.startswith("/api/images/"):
                parts = image.replace("/api/images/", "").split("/")
                if len(parts) == 2:
                    trace_id, filename = parts
                    actual_path = project_root / "logs" / "images" / trace_id / filename
                    if actual_path.exists():
                        logger.debug(f"Loading image from API path: {actual_path}")
                        return Image.open(actual_path)
            # Handle direct file path
            elif Path(image).exists():
                return Image.open(image)
            # Handle base64 data URL
            elif image.startswith("data:image"):
                b64_data = image.split(",")[1] if "," in image else image
                img_bytes = base64.b64decode(b64_data)
                return Image.open(io.BytesIO(img_bytes))
            # Handle full URL with http (extract path)
            elif image.startswith("http"):
                from urllib.parse import urlparse
                parsed = urlparse(image)
                return self._load_image_as_pil(parsed.path)

        logger.warning(f"Could not load image: {image}")
        return None

    def _extract_thinking_content(self, response: str) -> tuple[str, str]:
        """Extract thinking content and final response from Thinking model output.

        Args:
            response: Raw response that may contain <think>...</think> tags

        Returns:
            Tuple of (thinking_content, final_response)
        """
        import re

        # Match <think>...</think> content
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # Get everything after </think>
            final = response[think_match.end():].strip()
            return thinking, final

        # No thinking tags found
        return "", response

    def _parse_response(self, response: str, include_thinking: bool = False) -> dict:
        """Parse VLM response to extract JSON.

        Args:
            response: Raw VLM response text
            include_thinking: If True and response has <think> tags, include thinking in output

        Returns:
            Parsed JSON dict, or {"raw": response} if parsing fails
        """
        import re

        # Extract thinking content if present
        thinking_content, final_response = self._extract_thinking_content(response)

        # Try to extract JSON from markdown code block in final response
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', final_response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if include_thinking and thinking_content:
                    result["_thinking"] = thinking_content
                return result
            except json.JSONDecodeError:
                pass

        # Try direct JSON parse on final response
        try:
            result = json.loads(final_response)
            if include_thinking and thinking_content:
                result["_thinking"] = thinking_content
            return result
        except json.JSONDecodeError:
            pass

        # Return raw response in a dict
        result = {"raw": final_response}
        if include_thinking and thinking_content:
            result["_thinking"] = thinking_content
        return result

    def list_available_models(self) -> list[str]:
        """List available model variants."""
        models_dir = project_root / "models" / "qwen3vl"
        if not models_dir.exists():
            return []

        return [
            d.name for d in models_dir.iterdir()
            if d.is_dir() and "Instruct" in d.name or "Thinking" in d.name
        ]

    def unload(self) -> dict:
        """Unload model and free GPU memory."""
        import torch
        import gc

        model_name = self._current_model

        if self._model is not None:
            logger.info(f"Unloading model: {self._current_model}")
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        self._current_model = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("GPU memory cleared")

        return {
            "unloaded_model": model_name,
            "gpu_memory_freed": True,
        }


# Mock VLM for testing without GPU
class MockVLMRunner:
    """Mock VLM runner for testing without actual model."""

    def run(
        self,
        model: str,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images: dict[str, Any],
        context: Optional[dict] = None,
    ) -> VLMRunResult:
        """Return mock response."""
        if mode == "planning":
            parsed = {
                "thought": "Mock planning response",
                "next_task": "left_orange_plate",
                "confidence": 0.85,
                "needs_clarification": False,
            }
        else:
            parsed = {
                "success": False,
                "failure_type": "object_not_moved",
                "failure_reason": "Mock: Object still in original position",
                "reasoning": "Mock monitoring response",
                "control_action": "retry",
                "confidence": 0.8,
            }

        return VLMRunResult(
            model=model,
            raw_response=json.dumps(parsed),
            parsed_output=parsed,
            success=True,
            latency_ms=100.0,
        )

    def list_available_models(self) -> list[str]:
        return ["2B-Instruct", "2B-Thinking", "4B-Instruct", "4B-Thinking"]
