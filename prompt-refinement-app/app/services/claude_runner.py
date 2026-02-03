"""
Claude Runner Service.

Runs inference using Claude API as a reference/oracle for comparison.
"""

import json
import logging
import os
import base64
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ClaudeRunResult:
    """Result from running Claude inference."""
    model: str
    raw_response: str
    parsed_output: dict
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0


class ClaudeRunner:
    """Run inference using Claude API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude runner.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        """Ensure Anthropic client is initialized."""
        if self._client is not None:
            return

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("Claude client initialized")
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def run(
        self,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images: dict[str, Any],
        context: Optional[dict] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> ClaudeRunResult:
        """Run Claude inference.

        Args:
            mode: "planning" or "monitoring"
            system_prompt: System prompt text
            user_prompt: User prompt text
            images: Dict of image paths or base64 encoded images
            context: Optional context dict
            model: Claude model to use

        Returns:
            ClaudeRunResult with response and parsed output
        """
        start_time = time.time()

        try:
            self._ensure_client()

            # Build message content with images
            content = []

            # Add images
            for key, value in images.items():
                image_data = self._load_image_as_base64(value)
                if image_data:
                    content.append({
                        "type": "text",
                        "text": f"{key.upper()} image:",
                    })
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    })

            # Add user prompt
            content.append({
                "type": "text",
                "text": user_prompt,
            })

            # Call Claude API
            response = self._client.messages.create(
                model=model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
            )

            raw_response = response.content[0].text
            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            parsed = self._parse_response(raw_response)

            return ClaudeRunResult(
                model=model,
                raw_response=raw_response,
                parsed_output=parsed,
                success=True,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Claude run error: {e}")
            return ClaudeRunResult(
                model=model,
                raw_response="",
                parsed_output={},
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _load_image_as_base64(self, image: Any) -> Optional[str]:
        """Load image and convert to base64.

        Args:
            image: Path string, base64 string, or numpy array

        Returns:
            Base64 encoded image data
        """
        import numpy as np

        if isinstance(image, str):
            # Path or base64
            if Path(image).exists():
                with open(image, "rb") as f:
                    return base64.standard_b64encode(f.read()).decode("utf-8")
            elif image.startswith("data:image"):
                # Already base64 with header
                return image.split(",")[1] if "," in image else image
            else:
                # Assume it's already base64
                return image
        elif isinstance(image, np.ndarray):
            # Convert numpy array to JPEG
            from PIL import Image
            import io

            img = Image.fromarray(image)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

        return None

    def _parse_response(self, response: str) -> dict:
        """Parse Claude response to extract JSON."""
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

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return bool(self.api_key)


class MockClaudeRunner:
    """Mock Claude runner for testing without API."""

    def run(
        self,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        images: dict[str, Any],
        context: Optional[dict] = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> ClaudeRunResult:
        """Return mock Claude response (more accurate than Qwen mock)."""
        if mode == "planning":
            parsed = {
                "thought": "I can see an orange on the left side of the table. Since it's on the left, I should use the left arm to pick it up.",
                "next_task": "left_orange_plate",
                "confidence": 0.92,
                "needs_clarification": False,
            }
        else:
            # Mock Claude as more accurate - correctly identifies failure
            parsed = {
                "success": False,
                "failure_type": "object_not_moved",
                "failure_reason": "The orange is still on the table in the AFTER image, not on the plate. The plate only contains the ketchup bottle.",
                "reasoning": "Comparing BEFORE and AFTER images: In BEFORE, orange is on table next to corn. In AFTER, orange is STILL on table next to corn. The plate only has the ketchup bottle on it. The task 'left_orange_plate' was NOT successful.",
                "control_action": "retry",
                "confidence": 0.95,
            }

        return ClaudeRunResult(
            model=model,
            raw_response=json.dumps(parsed, indent=2),
            parsed_output=parsed,
            success=True,
            latency_ms=500.0,
        )

    def is_available(self) -> bool:
        return True
