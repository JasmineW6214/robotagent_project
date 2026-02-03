#!/usr/bin/env python3
"""
Cross-Attention Capture for SmolVLA Hallucination Investigation.

Captures cross-attention weights from the action expert to VLM prefix
during inference denoising steps. This is critical for understanding
what visual/language features drive action generation.

Key insight: Vision encoder self-attention shows internal image processing,
but cross-attention shows what the ACTION EXPERT attends to when generating
actions - this is what matters for diagnosing hallucination.

Hook location: smolvlm_with_expert.py:575 (after softmax in eager_attention_forward)

**UPDATED TOKEN LAYOUT (3 cameras):**

SmolVLA uses 3 camera images concatenated into a single prefix sequence.
For 512x512 images with SigLIP (14x14 patches → 24x24 grid = 576 patches per camera):

Token layout in VLM prefix (~1777 total):
  [0-575]      Head camera patches (576 = 24x24)
  [576-1151]   Left wrist camera patches (576)
  [1152-1727]  Right wrist camera patches (576)
  [1728-1775]  Language tokens (~48)
  [1776]       State token

CRITICAL: Previous analysis only examined head camera (729 patches assumed).
This update adds per-camera attention breakdown to identify if RIGHT WRIST
camera (where banana is visible at step 200+ in hallucination case) is the trigger.

Usage:
    python cross_attention_capture.py \
        --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --output-dir logs/yogurt_banana_leftarm/cross_attention_analysis/case1
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# TOKEN LAYOUT CONSTANTS (CORRECTED BASED ON ACTUAL MODEL)
# ============================================================================

# IMPORTANT: SmolVLA compresses 512x512 images to only ~64 tokens per camera!
# This is due to the multi_modal_projector pooling after SigLIP.
# Verified by running check_prefix_length.py:
#   - 3 images at 512x512 → 192 image tokens total (64 per camera, 8x8 grid)
#   - Plus ~48 language tokens + ~1 state token = 241 total prefix tokens

PATCHES_PER_CAMERA = 64  # 8x8 grid per camera (VERIFIED)
PATCH_GRID_SIZE = 8  # sqrt(64)
NUM_CAMERAS = 3  # head, left_wrist, right_wrist

# Token indices in VLM prefix (for 3 cameras)
# Layout: [head_patches | left_wrist_patches | right_wrist_patches | language | state]
HEAD_CAMERA_START = 0
HEAD_CAMERA_END = PATCHES_PER_CAMERA  # 64
LEFT_WRIST_START = HEAD_CAMERA_END  # 64
LEFT_WRIST_END = LEFT_WRIST_START + PATCHES_PER_CAMERA  # 128
RIGHT_WRIST_START = LEFT_WRIST_END  # 128
RIGHT_WRIST_END = RIGHT_WRIST_START + PATCHES_PER_CAMERA  # 192

# Language tokens follow images
LANGUAGE_START = RIGHT_WRIST_END  # 192
LANGUAGE_TOKENS_APPROX = 48
LANGUAGE_END = LANGUAGE_START + LANGUAGE_TOKENS_APPROX  # 240

# State token is last
STATE_INDEX = LANGUAGE_END  # 240

# Total prefix tokens
TOTAL_PREFIX_TOKENS = 241  # Verified by check_prefix_length.py

# Action tokens
NUM_ACTION_TOKENS = 50  # Chunk size

# Legacy single-camera constants (for backward compatibility)
LEGACY_PATCH_GRID_SIZE = 27
LEGACY_NUM_IMAGE_PATCHES = LEGACY_PATCH_GRID_SIZE * LEGACY_PATCH_GRID_SIZE  # 729


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PerCameraAttention:
    """Attention breakdown per camera."""
    head_camera: float = 0.0  # % attention to head camera patches
    left_wrist: float = 0.0  # % attention to left wrist patches
    right_wrist: float = 0.0  # % attention to right wrist patches

    # Spatial attention maps per camera (24x24 each)
    head_spatial: Optional[np.ndarray] = None
    left_wrist_spatial: Optional[np.ndarray] = None
    right_wrist_spatial: Optional[np.ndarray] = None


@dataclass
class DenoiseStepAttention:
    """Cross-attention data for a single denoising step."""
    step: int  # 0-9
    time: float  # 1.0 -> 0.1
    layer_idx: int  # Which transformer layer

    # Attention statistics (combined)
    image_attention_ratio: float = 0.0  # % attention to ALL image patches
    language_attention_ratio: float = 0.0  # % attention to language tokens
    state_attention_ratio: float = 0.0  # % attention to state token
    attention_entropy: float = 0.0  # Shannon entropy (lower = more focused)

    # PER-CAMERA BREAKDOWN (NEW)
    per_camera: Optional[PerCameraAttention] = None

    # Spatial attention map (average over heads and action tokens)
    # For backward compatibility, this is the combined/first camera
    spatial_attention: Optional[np.ndarray] = None  # [24, 24] or [27, 27]


@dataclass
class CrossAttentionAnalysis:
    """Complete cross-attention analysis for one inference run."""
    case_dir: str
    task: str
    timestamp: str

    # Per-denoising-step attention (key = step index)
    denoising_steps: dict = field(default_factory=dict)

    # Aggregated metrics
    avg_image_attention: float = 0.0
    avg_language_attention: float = 0.0
    avg_entropy: float = 0.0

    # Temporal trend
    image_attention_trend: str = "stable"  # "increasing", "decreasing", "stable"


# ============================================================================
# CROSS-ATTENTION CAPTURE HOOK
# ============================================================================

class CrossAttentionCaptureHook:
    """
    Captures cross-attention weights during SmolVLA inference.

    Hooks into eager_attention_forward() to capture attention probabilities
    after softmax. Only captures when query length = 50 (action tokens),
    indicating cross-attention rather than self-attention.

    Also hooks into denoise_step() to track which denoising iteration we're in.
    """

    def __init__(self):
        self.attention_data = {}  # {denoising_step: {layer_idx: attention_tensor}}
        self.current_denoising_step = 0
        self.is_capturing = False
        self._original_forward = None
        self._original_denoise_step = None
        self._model = None
        self._hooks = []

    def register_hooks(self, vlm_with_expert, model=None):
        """
        Register hook by wrapping the eager_attention_forward method.

        Note: We wrap the method rather than using register_forward_hook because
        we need to capture intermediate values (attention probs) not just output.
        """
        # Store original forward method and model reference
        self._original_forward = vlm_with_expert.eager_attention_forward
        self._vlm_with_expert = vlm_with_expert

        # Create capturing wrapper that properly replicates the original function
        @functools.wraps(self._original_forward)
        def capturing_forward(
            attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        ):
            # Get attention head configuration from model
            num_att_heads = vlm_with_expert.num_attention_heads
            num_key_value_heads = vlm_with_expert.num_key_value_heads
            num_key_value_groups = num_att_heads // num_key_value_heads

            sequence_length = key_states.shape[1]

            # Expand key states for grouped query attention (lines 548-553)
            key_states_expanded = key_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            key_states_expanded = key_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            # Expand value states (lines 555-560)
            value_states_expanded = value_states[:, :, :, None, :].expand(
                batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
            )
            value_states_expanded = value_states_expanded.reshape(
                batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
            )

            # Upcast to float32 (lines 562-564)
            query_states_f32 = query_states.to(dtype=torch.float32)
            key_states_f32 = key_states_expanded.to(dtype=torch.float32)

            # Transpose for attention computation (lines 566-567)
            query_states_t = query_states_f32.transpose(1, 2)
            key_states_t = key_states_f32.transpose(1, 2)

            # Compute attention weights (lines 569-570)
            att_weights = torch.matmul(query_states_t, key_states_t.transpose(2, 3))
            att_weights = att_weights * (head_dim ** -0.5)

            # Apply mask (lines 572-574)
            att_weights = att_weights.to(dtype=torch.float32)
            big_neg = torch.finfo(att_weights.dtype).min
            masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

            # Softmax (line 575)
            probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)

            # CAPTURE: Store probs if in cross-attention mode
            # Query shape after transpose: [batch, heads, query_len, head_dim]
            query_len = query_states_t.shape[2]
            key_len = key_states_t.shape[2]

            if self.is_capturing and query_len == NUM_ACTION_TOKENS and key_len > NUM_ACTION_TOKENS:
                step_key = self.current_denoising_step
                if step_key not in self.attention_data:
                    self.attention_data[step_key] = {}

                layer_idx = len(self.attention_data[step_key])
                # Store detached, float32 CPU tensor
                # probs shape: [batch, heads, query_len, key_len]
                self.attention_data[step_key][layer_idx] = probs.detach().float().cpu()

            # Continue with original computation (lines 576-582)
            probs = probs.to(dtype=value_states_expanded.dtype)
            att_output = torch.matmul(probs, value_states_expanded.permute(0, 2, 1, 3))
            att_output = att_output.permute(0, 2, 1, 3)
            att_output = att_output.reshape(
                batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim
            )

            return att_output

        # Replace method
        vlm_with_expert.eager_attention_forward = capturing_forward

        # Also hook into the model's denoise_step to track denoising iterations
        if model is not None:
            self._model = model
            self._original_denoise_step = model.denoise_step

            @functools.wraps(self._original_denoise_step)
            def tracking_denoise_step(*args, **kwargs):
                # Increment step counter BEFORE denoise_step runs
                # (step counter starts at 0, increments after each iteration)
                result = self._original_denoise_step(*args, **kwargs)
                # After denoise_step completes, increment for next iteration
                if self.is_capturing:
                    self.current_denoising_step += 1
                return result

            model.denoise_step = tracking_denoise_step

    def remove_hooks(self):
        """Restore original methods."""
        if self._vlm_with_expert is not None and self._original_forward is not None:
            self._vlm_with_expert.eager_attention_forward = self._original_forward
        if self._model is not None and self._original_denoise_step is not None:
            self._model.denoise_step = self._original_denoise_step
        self.attention_data = {}
        self._original_forward = None
        self._original_denoise_step = None

    def start_capture(self, denoising_step: int):
        """Start capturing for a specific denoising step."""
        self.is_capturing = True
        self.current_denoising_step = denoising_step

    def stop_capture(self):
        """Stop capturing."""
        self.is_capturing = False

    def clear(self):
        """Clear all captured data."""
        self.attention_data = {}
        self.current_denoising_step = 0

    def get_step_attention(self, step: int) -> Optional[dict]:
        """Get captured attention for a denoising step."""
        return self.attention_data.get(step)


# ============================================================================
# SPATIAL ATTENTION MAPPING
# ============================================================================

def map_attention_to_spatial(
    cross_attention: np.ndarray,
    patch_grid_size: int = PATCH_GRID_SIZE,
    debug: bool = False,
) -> np.ndarray:
    """
    Map cross-attention weights to spatial image regions.

    Args:
        cross_attention: Attention weights [num_heads, action_tokens, prefix_tokens]
                        or [batch, num_heads, action_tokens, prefix_tokens]
        debug: Print debug info about token layout

    Returns:
        spatial_attention: [patch_grid_size, patch_grid_size] normalized attention map
    """
    # Handle batch dimension
    if cross_attention.ndim == 4:
        cross_attention = cross_attention[0]  # [heads, actions, prefix]

    num_heads, num_actions, prefix_len = cross_attention.shape

    if debug:
        print(f"    Cross-attention shape: heads={num_heads}, actions={num_actions}, prefix={prefix_len}")

    # SmolVLA uses dynamic image token count based on the multi_modal_projector
    # The actual number of image tokens can vary. Let's infer it from the prefix length.
    #
    # Typical layout: [image_special] + [image_patches] + [language_tokens] + [state]
    # Language tokens are typically ~20-50, state is 1
    #
    # For prefix_len=291 with ~25 language tokens + 1 state = ~265 image tokens
    # This suggests a different patch configuration than 27x27

    # Try to infer the number of image tokens
    # Assume language tokens + state tokens ≈ 50 tokens at the end
    estimated_lang_state = min(50, prefix_len // 5)  # Conservative estimate
    estimated_image_tokens = prefix_len - estimated_lang_state

    # Find closest perfect square for spatial mapping
    sqrt_tokens = int(np.sqrt(estimated_image_tokens))
    actual_grid_size = sqrt_tokens

    if debug:
        print(f"    Estimated image tokens: {estimated_image_tokens}, grid: {actual_grid_size}x{actual_grid_size}")

    # Extract attention to estimated image region (all but last ~50 tokens)
    image_end = prefix_len - estimated_lang_state
    attn_to_image = cross_attention[:, :, :image_end]

    # Average over heads and action tokens
    attn_avg = attn_to_image.mean(axis=(0, 1))

    # Reshape to spatial grid (use the closest perfect square)
    target_size = actual_grid_size * actual_grid_size
    if len(attn_avg) >= target_size:
        attn_to_reshape = attn_avg[:target_size]
    else:
        # Pad if needed
        attn_to_reshape = np.pad(attn_avg, (0, target_size - len(attn_avg)))

    spatial_attn = attn_to_reshape.reshape(actual_grid_size, actual_grid_size)

    # Normalize to [0, 1]
    spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-10)

    return spatial_attn


def infer_token_layout(prefix_len: int, debug: bool = False) -> dict:
    """
    Infer token layout from actual prefix length.

    IMPORTANT: The attention matrix key dimension may include action tokens for self-attention.
    If key_len > 250, it likely includes 50 action tokens that should be excluded.

    SmolVLA prefix layouts (verified by check_prefix_length.py):
    - 3 cameras: ~241 tokens (64 patches × 3 cameras + 48 lang + 1 state)
    - Attention key_len: ~291 (241 prefix + 50 action self-attention)

    Returns:
        dict with camera boundaries and layout info
    """
    # If key_len includes action tokens (50), remove them first
    # Check if prefix_len > 250, indicating action tokens are included
    actual_prefix_len = prefix_len
    if prefix_len > 250:
        actual_prefix_len = prefix_len - NUM_ACTION_TOKENS  # Remove 50 action tokens
        if debug:
            print(f"    Detected action tokens in key: {prefix_len} - 50 = {actual_prefix_len} prefix tokens")

    # Estimate language+state tokens at end (~49)
    estimated_lang_state = 49  # 48 language + 1 state

    total_image_tokens = actual_prefix_len - estimated_lang_state

    # Determine number of cameras based on actual prefix length
    # 3 cameras with 64 patches each = 192 image tokens → ~241 total prefix
    if 180 <= total_image_tokens <= 210:  # Around 192 = 3 cameras × 64
        num_cameras = 3
        patches_per_camera = total_image_tokens // 3  # ~64
    elif total_image_tokens > 210:  # More tokens = likely 1 camera with more patches
        num_cameras = 1
        patches_per_camera = total_image_tokens
    else:
        # Fallback: assume single camera
        num_cameras = 1
        patches_per_camera = max(total_image_tokens, 64)

    grid_size = int(np.sqrt(patches_per_camera))

    layout = {
        "num_cameras": num_cameras,
        "patches_per_camera": patches_per_camera,
        "grid_size": grid_size,
        "total_image_tokens": total_image_tokens,
        "lang_state_tokens": estimated_lang_state,
        "actual_prefix_len": actual_prefix_len,  # Without action tokens
        "raw_key_len": prefix_len,  # Original key length
    }

    if num_cameras == 3:
        # Exact boundaries for 3-camera layout
        layout["head_start"] = 0
        layout["head_end"] = patches_per_camera  # ~64
        layout["left_start"] = patches_per_camera  # ~64
        layout["left_end"] = 2 * patches_per_camera  # ~128
        layout["right_start"] = 2 * patches_per_camera  # ~128
        layout["right_end"] = 3 * patches_per_camera  # ~192
        layout["lang_start"] = 3 * patches_per_camera  # ~192
        layout["lang_end"] = actual_prefix_len - 1
        layout["state_idx"] = actual_prefix_len - 1
    else:
        layout["head_start"] = 0
        layout["head_end"] = total_image_tokens
        layout["left_start"] = None
        layout["left_end"] = None
        layout["right_start"] = None
        layout["right_end"] = None
        layout["lang_start"] = total_image_tokens
        layout["lang_end"] = actual_prefix_len - 1
        layout["state_idx"] = actual_prefix_len - 1

    if debug:
        print(f"    Key length: {prefix_len}, actual prefix: {actual_prefix_len}, inferred {num_cameras} cameras")
        print(f"    Image tokens: {total_image_tokens}, patches per camera: {patches_per_camera} ({grid_size}x{grid_size} grid)")
        if num_cameras == 3:
            print(f"    Head=[0:{layout['head_end']}], Left=[{layout['left_start']}:{layout['left_end']}], Right=[{layout['right_start']}:{layout['right_end']}]")
            print(f"    Lang=[{layout['lang_start']}:{layout['lang_end']}], State=[{layout['state_idx']}]")

    return layout


def compute_attention_metrics(
    cross_attention: np.ndarray,
    prefix_len: Optional[int] = None,
    debug: bool = False,
) -> dict:
    """
    Compute attention distribution metrics WITH PER-CAMERA BREAKDOWN.

    Args:
        cross_attention: [heads, actions, key_len] or [batch, heads, actions, key_len]
                        Note: key_len may include action tokens for self-attention
        debug: Print debug info

    Returns:
        dict with:
        - image_ratio, language_ratio, state_ratio, entropy (combined)
        - per_camera: {head_camera, left_wrist, right_wrist} (if 3 cameras detected)
    """
    if cross_attention.ndim == 4:
        cross_attention = cross_attention[0]

    num_heads, num_actions, key_len = cross_attention.shape

    # Infer token layout (handles action tokens in key dimension)
    layout = infer_token_layout(key_len, debug=debug)

    # Get actual prefix length (without action tokens)
    actual_prefix_len = layout["actual_prefix_len"]

    # Only analyze attention to prefix tokens, not action self-attention
    # Truncate to actual prefix length
    cross_attention = cross_attention[:, :, :actual_prefix_len]

    # Compute total attention per region (sum over heads and actions)
    total_attn = cross_attention.sum()

    # Combined image attention
    total_image_end = layout["total_image_tokens"]
    image_attn = cross_attention[:, :, :total_image_end].sum() / total_attn if total_image_end > 0 else 0

    # Language attention
    lang_start = layout["lang_start"]
    lang_end = layout["lang_end"]
    if lang_end > lang_start:
        lang_attn = cross_attention[:, :, lang_start:lang_end].sum() / total_attn
    else:
        lang_attn = 0

    # State attention
    state_idx = layout["state_idx"]
    state_attn = cross_attention[:, :, state_idx:].sum() / total_attn

    # Entropy (measure of focus vs diffusion)
    attn_avg = cross_attention.mean(axis=0)  # [actions, prefix]
    attn_flat = attn_avg.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-10)
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))

    result = {
        "image_attention_ratio": float(image_attn),
        "language_attention_ratio": float(lang_attn),
        "state_attention_ratio": float(state_attn),
        "attention_entropy": float(entropy),
        "layout": layout,
    }

    # PER-CAMERA BREAKDOWN (if 3 cameras detected)
    if layout["num_cameras"] == 3:
        head_attn = cross_attention[:, :, layout["head_start"]:layout["head_end"]].sum() / total_attn
        left_attn = cross_attention[:, :, layout["left_start"]:layout["left_end"]].sum() / total_attn
        right_attn = cross_attention[:, :, layout["right_start"]:layout["right_end"]].sum() / total_attn

        result["per_camera"] = {
            "head_camera": float(head_attn),
            "left_wrist": float(left_attn),
            "right_wrist": float(right_attn),  # CRITICAL: Check if this is elevated in hallucination case
        }

        if debug:
            print(f"    Per-camera attention: Head={head_attn:.3f}, Left={left_attn:.3f}, Right={right_attn:.3f}")

    return result


def map_attention_to_spatial_per_camera(
    cross_attention: np.ndarray,
    layout: dict,
    debug: bool = False,
    use_global_normalization: bool = True,
) -> dict:
    """
    Map cross-attention to spatial heatmaps for each camera.

    Args:
        cross_attention: [heads, actions, prefix] or [batch, heads, actions, prefix]
        layout: Token layout from infer_token_layout()
        use_global_normalization: If True, normalize all cameras using global min/max
                                  to enable cross-camera comparison. Default: True.

    Returns:
        dict with spatial attention maps per camera:
        - 'head': np.ndarray [grid, grid]
        - 'left_wrist': np.ndarray [grid, grid] (if 3 cameras)
        - 'right_wrist': np.ndarray [grid, grid] (if 3 cameras)
        - 'combined': np.ndarray (all image tokens combined)
    """
    if cross_attention.ndim == 4:
        cross_attention = cross_attention[0]

    grid_size = layout["grid_size"]
    patches_per_camera = layout["patches_per_camera"]
    target_size = grid_size * grid_size

    result = {}
    raw_spatials = {}  # Store raw values before normalization

    def extract_and_reshape(start, end, name):
        """Extract attention slice and reshape to spatial grid."""
        attn_slice = cross_attention[:, :, start:end]
        attn_avg = attn_slice.mean(axis=(0, 1))  # Average over heads and actions

        # Handle size mismatch
        if len(attn_avg) >= target_size:
            attn_to_reshape = attn_avg[:target_size]
        else:
            attn_to_reshape = np.pad(attn_avg, (0, target_size - len(attn_avg)))

        spatial = attn_to_reshape.reshape(grid_size, grid_size)
        raw_spatials[name] = spatial.copy()  # Store raw before normalization
        return spatial

    # Head camera (always present)
    result["head"] = extract_and_reshape(layout["head_start"], layout["head_end"], "head")

    # Left and right wrist (if 3 cameras)
    if layout["num_cameras"] == 3:
        result["left_wrist"] = extract_and_reshape(layout["left_start"], layout["left_end"], "left_wrist")
        result["right_wrist"] = extract_and_reshape(layout["right_start"], layout["right_end"], "right_wrist")

    # Apply normalization
    if use_global_normalization and len(raw_spatials) > 1:
        # Global normalization: use min/max across ALL cameras for fair comparison
        all_values = np.concatenate([s.flatten() for s in raw_spatials.values()])
        global_min = all_values.min()
        global_max = all_values.max()

        for name in raw_spatials:
            spatial = raw_spatials[name]
            result[name] = (spatial - global_min) / (global_max - global_min + 1e-10)
    else:
        # Per-camera normalization (original behavior)
        for name in raw_spatials:
            spatial = raw_spatials[name]
            result[name] = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-10)

    # Combined (all image tokens) - always uses its own normalization
    total_img_end = layout["total_image_tokens"]
    attn_all_img = cross_attention[:, :, :total_img_end].mean(axis=(0, 1))
    # For combined, try to make a reasonable grid
    combined_grid = int(np.sqrt(len(attn_all_img)))
    combined_target = combined_grid * combined_grid
    if len(attn_all_img) >= combined_target:
        combined_spatial = attn_all_img[:combined_target].reshape(combined_grid, combined_grid)
    else:
        combined_spatial = attn_all_img.reshape(-1, 1)  # Fallback to 1D
    combined_spatial = (combined_spatial - combined_spatial.min()) / (combined_spatial.max() - combined_spatial.min() + 1e-10)
    result["combined"] = combined_spatial

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_attention_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay attention heatmap on image.

    Args:
        image: RGB image [H, W, 3]
        attention_map: Normalized attention [patch_h, patch_w]
        alpha: Blend factor

    Returns:
        Image with heatmap overlay [H, W, 3]
    """
    h, w = image.shape[:2]

    # Resize attention map to image size
    attn_resized = cv2.resize(attention_map.astype(np.float32), (w, h))

    # Create heatmap using jet colormap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def plot_temporal_evolution(
    step_data: dict[int, DenoiseStepAttention],
    images: Optional[list[np.ndarray]] = None,
    output_path: Path = None,
    title: str = "Cross-Attention Evolution Through Denoising",
):
    """
    Plot 10-panel figure showing attention evolution across denoising steps.

    Args:
        step_data: {step_idx: DenoiseStepAttention}
        images: Optional list of images to overlay attention on
        output_path: Where to save the figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for step in range(10):
        row = step // 5
        col = step % 5
        ax = axes[row, col]

        if step in step_data and step_data[step].spatial_attention is not None:
            spatial_attn = step_data[step].spatial_attention
            time = step_data[step].time
            entropy = step_data[step].attention_entropy

            if images and len(images) > 0:
                # Overlay on first image
                overlay = create_attention_heatmap_overlay(images[0], spatial_attn, alpha=0.6)
                ax.imshow(overlay)
            else:
                # Just show attention map
                im = ax.imshow(spatial_attn, cmap='hot', vmin=0, vmax=1)

            ax.set_title(f"Step {step} (t={time:.1f})\nH={entropy:.2f}", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Step {step}\nNo data", ha='center', va='center')

        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal evolution: {output_path}")

    plt.close()


def plot_attention_metrics_over_time(
    step_data: dict[int, DenoiseStepAttention],
    output_path: Path = None,
    title: str = "Attention Metrics Across Denoising Steps",
):
    """
    Plot line graphs of attention metrics over denoising steps.
    """
    steps = sorted(step_data.keys())
    if not steps:
        return

    image_ratios = [step_data[s].image_attention_ratio for s in steps]
    lang_ratios = [step_data[s].language_attention_ratio for s in steps]
    entropies = [step_data[s].attention_entropy for s in steps]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Image attention ratio
    axes[0].plot(steps, image_ratios, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Denoising Step')
    axes[0].set_ylabel('Image Attention Ratio')
    axes[0].set_title('Attention to Image Patches')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Language attention ratio
    axes[1].plot(steps, lang_ratios, 'g-o', linewidth=2, markersize=6)
    axes[1].set_xlabel('Denoising Step')
    axes[1].set_ylabel('Language Attention Ratio')
    axes[1].set_title('Attention to Language Tokens')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Entropy
    axes[2].plot(steps, entropies, 'r-o', linewidth=2, markersize=6)
    axes[2].set_xlabel('Denoising Step')
    axes[2].set_ylabel('Attention Entropy')
    axes[2].set_title('Attention Focus (lower = more focused)')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention metrics: {output_path}")

    plt.close()


def plot_per_camera_attention(
    step_data: dict[int, DenoiseStepAttention],
    output_path: Path = None,
    title: str = "Per-Camera Attention Breakdown",
):
    """
    Plot per-camera attention breakdown over denoising steps.

    CRITICAL: This visualization shows which camera receives attention.
    If right_wrist attention is elevated in hallucination case, it may
    indicate the banana in right wrist camera is triggering the behavior.
    """
    steps = sorted(step_data.keys())
    if not steps:
        return

    # Check if per-camera data is available
    has_per_camera = any(
        step_data[s].per_camera is not None
        for s in steps
    )

    if not has_per_camera:
        print("  No per-camera attention data available (single camera mode)")
        return

    head_ratios = []
    left_ratios = []
    right_ratios = []

    for s in steps:
        if step_data[s].per_camera:
            head_ratios.append(step_data[s].per_camera.head_camera)
            left_ratios.append(step_data[s].per_camera.left_wrist)
            right_ratios.append(step_data[s].per_camera.right_wrist)
        else:
            head_ratios.append(0)
            left_ratios.append(0)
            right_ratios.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked area chart or line chart
    ax.plot(steps, head_ratios, 'b-o', linewidth=2, markersize=6, label='Head Camera')
    ax.plot(steps, left_ratios, 'g-o', linewidth=2, markersize=6, label='Left Wrist')
    ax.plot(steps, right_ratios, 'r-o', linewidth=2, markersize=6, label='Right Wrist ⚠️')

    ax.set_xlabel('Denoising Step', fontsize=12)
    ax.set_ylabel('Attention Ratio', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation if right wrist is elevated
    if right_ratios and max(right_ratios) > 0.25:
        max_idx = right_ratios.index(max(right_ratios))
        ax.annotate(
            f'Right wrist elevated: {max(right_ratios):.1%}',
            xy=(steps[max_idx], right_ratios[max_idx]),
            xytext=(steps[max_idx] + 1, right_ratios[max_idx] + 0.05),
            fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-camera attention: {output_path}")

    plt.close()


def plot_per_camera_spatial_heatmaps(
    spatial_maps: dict,
    images: dict = None,
    output_path: Path = None,
    title: str = "Spatial Attention Per Camera",
):
    """
    Plot spatial attention heatmaps for each camera side-by-side.

    Args:
        spatial_maps: {'head': np.array, 'left_wrist': np.array, 'right_wrist': np.array}
        images: Optional {'head': image, 'left_wrist': image, 'right_wrist': image}
        output_path: Where to save
        title: Plot title
    """
    num_cameras = len([k for k in spatial_maps.keys() if k != 'combined'])

    if num_cameras == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        axes = [ax]
        camera_keys = ['head']
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        camera_keys = ['head', 'left_wrist', 'right_wrist']

    camera_labels = {
        'head': 'Head Camera',
        'left_wrist': 'Left Wrist Camera',
        'right_wrist': 'Right Wrist Camera ⚠️',  # Highlight right wrist
    }

    for ax, cam_key in zip(axes, camera_keys):
        if cam_key not in spatial_maps:
            ax.text(0.5, 0.5, f"No data for {cam_key}", ha='center', va='center')
            ax.axis('off')
            continue

        spatial_attn = spatial_maps[cam_key]

        if images and cam_key in images and images[cam_key] is not None:
            # Overlay on image
            overlay = create_attention_heatmap_overlay(images[cam_key], spatial_attn, alpha=0.6)
            ax.imshow(overlay)
        else:
            # Just show heatmap
            im = ax.imshow(spatial_attn, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(camera_labels.get(cam_key, cam_key), fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-camera spatial heatmaps: {output_path}")

    plt.close()


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_case_cross_attention(
    case_dir: Path,
    output_dir: Path,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> CrossAttentionAnalysis:
    """
    Analyze cross-attention for a captured inference case.

    This loads the saved images from the case, runs inference with
    cross-attention capture hooks, and generates analysis.
    """
    print(f"Analyzing cross-attention for case: {case_dir}")

    # Load metadata
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {case_dir}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    task = metadata.get("task_original", "Unknown task")
    checkpoint = checkpoint_path or metadata.get("checkpoint")
    print(f"Task: {task}")
    print(f"Checkpoint: {checkpoint}")

    # Load images
    images_dir = case_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory in {case_dir}")

    image_files = sorted(images_dir.glob("step_*_head.jpg"))
    if not image_files:
        raise FileNotFoundError("No head camera images found")

    # We'll analyze at specific inference steps (50-step intervals)
    # Pick representative steps: 200, 250, 300 (critical for hallucination)
    priority_steps = [0, 100, 200, 250, 300, 350]
    available_steps = sorted(set(int(p.stem.split('_')[1]) for p in image_files))
    steps_to_analyze = [s for s in priority_steps if s in available_steps]

    if not steps_to_analyze:
        steps_to_analyze = available_steps[:3]

    print(f"Will analyze steps: {steps_to_analyze}")

    # Load model
    print("Loading SmolVLA policy...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.eval()
    policy.to(device)

    # Load preprocessor
    dataset_path = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
    dataset_metadata = LeRobotDatasetMetadata(repo_id="multitasks", root=str(dataset_path))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Setup cross-attention capture hook
    capture = CrossAttentionCaptureHook()
    capture.register_hooks(policy.model.vlm_with_expert, model=policy.model)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis results
    analysis = CrossAttentionAnalysis(
        case_dir=str(case_dir),
        task=task,
        timestamp=datetime.now().isoformat(),
    )

    all_step_data = {}  # {inference_step: {denoising_step: DenoiseStepAttention}}
    images_for_overlay = {}  # {inference_step: image}

    # Process each inference step
    for step_num in steps_to_analyze:
        print(f"\n  Processing inference step {step_num}...")

        # Load images for this step
        head_path = images_dir / f"step_{step_num:04d}_head.jpg"
        left_wrist_path = images_dir / f"step_{step_num:04d}_left_wrist.jpg"
        right_wrist_path = images_dir / f"step_{step_num:04d}_right_wrist.jpg"

        if not all(p.exists() for p in [head_path, left_wrist_path, right_wrist_path]):
            print(f"    Missing camera images for step {step_num}")
            continue

        # Load images
        head_img = cv2.cvtColor(cv2.imread(str(head_path)), cv2.COLOR_BGR2RGB)
        left_wrist_img = cv2.cvtColor(cv2.imread(str(left_wrist_path)), cv2.COLOR_BGR2RGB)
        right_wrist_img = cv2.cvtColor(cv2.imread(str(right_wrist_path)), cv2.COLOR_BGR2RGB)

        # Store ALL camera images for this inference step (for per-camera heatmaps)
        images_for_overlay[step_num] = {
            'head': head_img,
            'left_wrist': left_wrist_img,
            'right_wrist': right_wrist_img,
        }

        # Create observation
        state = np.zeros(12, dtype=np.float32)

        def img_to_tensor(img):
            return torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        observation = {
            "observation.state": torch.from_numpy(state).float().unsqueeze(0).to(device),
            "observation.images.camera1": img_to_tensor(head_img).to(device),
            "observation.images.camera2": img_to_tensor(left_wrist_img).to(device),
            "observation.images.camera3": img_to_tensor(right_wrist_img).to(device),
            "task": task,
        }

        preprocessed_obs = preprocessor(observation)

        # Reset policy to force new chunk generation
        policy.reset()

        # Clear previous capture
        capture.clear()

        # Run inference with capture enabled for all denoising steps
        # Note: We need to modify the policy's sample_actions to capture per-step
        # For now, we capture whatever cross-attention happens during inference

        capture.is_capturing = True

        try:
            with torch.no_grad():
                action = policy.select_action(preprocessed_obs)
            print(f"    Inference OK")
        except Exception as e:
            print(f"    Inference error: {e}")
            continue

        capture.is_capturing = False

        # Process captured attention
        # The capture contains attention from multiple denoising steps
        # (each step calls eager_attention_forward multiple times)

        # Initialize nested dict for this inference step
        if step_num not in all_step_data:
            all_step_data[step_num] = {}

        for denoise_step, layer_data in capture.attention_data.items():
            # Use attention from middle layer (layer 8 typically has semantic info)
            target_layer = min(8, len(layer_data) - 1)

            if target_layer in layer_data:
                attn = layer_data[target_layer].numpy()

                # Compute metrics WITH per-camera breakdown
                is_first = (denoise_step == 0 and step_num == steps_to_analyze[0])
                metrics = compute_attention_metrics(attn, debug=is_first)

                # Compute per-camera spatial maps
                layout = metrics.get("layout", {})
                spatial_maps = map_attention_to_spatial_per_camera(attn, layout, debug=is_first)

                # Use head camera spatial attention as the default
                spatial_attn = spatial_maps.get("head", spatial_maps.get("combined"))

                # Build per-camera attention dataclass if available
                per_camera = None
                if "per_camera" in metrics:
                    per_camera = PerCameraAttention(
                        head_camera=metrics["per_camera"]["head_camera"],
                        left_wrist=metrics["per_camera"]["left_wrist"],
                        right_wrist=metrics["per_camera"]["right_wrist"],
                        head_spatial=spatial_maps.get("head"),
                        left_wrist_spatial=spatial_maps.get("left_wrist"),
                        right_wrist_spatial=spatial_maps.get("right_wrist"),
                    )

                # Estimate time from step (t = 1.0 - step * 0.1)
                time = 1.0 - denoise_step * 0.1

                step_attention = DenoiseStepAttention(
                    step=denoise_step,
                    time=time,
                    layer_idx=target_layer,
                    image_attention_ratio=metrics["image_attention_ratio"],
                    language_attention_ratio=metrics["language_attention_ratio"],
                    state_attention_ratio=metrics["state_attention_ratio"],
                    attention_entropy=metrics["attention_entropy"],
                    per_camera=per_camera,
                    spatial_attention=spatial_attn,
                )

                all_step_data[step_num][denoise_step] = step_attention

                # Print per-camera breakdown for first step
                if is_first and per_camera:
                    print(f"    Per-camera attention: Head={per_camera.head_camera:.1%}, Left={per_camera.left_wrist:.1%}, Right={per_camera.right_wrist:.1%}")

        print(f"    Captured {len(capture.attention_data)} denoising steps")

    # Flatten all data for aggregation: collect all DenoiseStepAttention objects
    all_attention_data = []
    for inf_step, denoise_dict in all_step_data.items():
        for denoise_step, data in denoise_dict.items():
            all_attention_data.append(data)

    # Compute aggregated metrics
    if all_attention_data:
        analysis.avg_image_attention = np.mean([d.image_attention_ratio for d in all_attention_data])
        analysis.avg_language_attention = np.mean([d.language_attention_ratio for d in all_attention_data])
        analysis.avg_entropy = np.mean([d.attention_entropy for d in all_attention_data])

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Temporal evolution - use last inference step's denoising progression
    last_inf_step = max(all_step_data.keys())
    last_images = images_for_overlay.get(last_inf_step, {})
    head_image_for_overlay = last_images.get('head') if isinstance(last_images, dict) else last_images

    plot_temporal_evolution(
        all_step_data[last_inf_step],
        images=[head_image_for_overlay],
        output_path=output_dir / "temporal_evolution.png",
        title=f"Cross-Attention Evolution (inf step {last_inf_step})\nTask: {task[:50]}...",
    )

    # Metrics over time for last inference step
    plot_attention_metrics_over_time(
        all_step_data[last_inf_step],
        output_path=output_dir / "attention_metrics.png",
    )

    # NEW: Per-camera attention breakdown over denoising steps
    plot_per_camera_attention(
        all_step_data[last_inf_step],
        output_path=output_dir / "per_camera_attention.png",
        title=f"Per-Camera Attention (inf step {last_inf_step})\n⚠️ Check if Right Wrist is elevated in hallucination case",
    )

    # Save individual spatial attention maps and generate heatmaps for ALL inference steps
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)

    # Key denoise step for visualizations (middle of denoising process)
    mid_denoise_step = 5

    # Generate per-camera spatial heatmaps for ALL key inference steps
    print("\nGenerating per-camera spatial heatmaps for all inference steps...")
    for inf_step, denoise_dict in all_step_data.items():
        # Generate per-camera 3-panel heatmap for this inference step (denoise step 5)
        if mid_denoise_step in denoise_dict:
            data = denoise_dict[mid_denoise_step]
            if data.per_camera and inf_step in images_for_overlay:
                spatial_maps = {
                    'head': data.per_camera.head_spatial,
                    'left_wrist': data.per_camera.left_wrist_spatial,
                    'right_wrist': data.per_camera.right_wrist_spatial,
                }
                img_dict = images_for_overlay[inf_step]
                plot_per_camera_spatial_heatmaps(
                    spatial_maps,
                    images=img_dict if isinstance(img_dict, dict) else None,
                    output_path=heatmaps_dir / f"inf_{inf_step:04d}_per_camera.png",
                    title=f"Per-Camera Spatial Attention (inf {inf_step}, denoise {mid_denoise_step})",
                )

        # Save .npy files and generate overlays for all denoise steps
        for denoise_step, data in denoise_dict.items():
            if data.spatial_attention is not None:
                # Save raw spatial attention as .npy for quantitative analysis
                np.save(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_spatial.npy", data.spatial_attention)

                # Save per-camera spatial maps if available
                if data.per_camera:
                    if data.per_camera.head_spatial is not None:
                        np.save(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_head.npy", data.per_camera.head_spatial)
                    if data.per_camera.left_wrist_spatial is not None:
                        np.save(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_left_wrist.npy", data.per_camera.left_wrist_spatial)
                    if data.per_camera.right_wrist_spatial is not None:
                        np.save(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_right_wrist.npy", data.per_camera.right_wrist_spatial)

                # Generate 3-panel per-camera overlay for key denoise steps (0, 5, 9)
                if denoise_step in [0, 5, 9] and data.per_camera and inf_step in images_for_overlay:
                    img_dict = images_for_overlay[inf_step]
                    if isinstance(img_dict, dict):
                        spatial_maps = {
                            'head': data.per_camera.head_spatial,
                            'left_wrist': data.per_camera.left_wrist_spatial,
                            'right_wrist': data.per_camera.right_wrist_spatial,
                        }
                        # Create 3-panel figure
                        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                        camera_info = [
                            ('head', 'Head Camera', img_dict.get('head')),
                            ('left_wrist', 'Left Wrist', img_dict.get('left_wrist')),
                            ('right_wrist', 'Right Wrist ⚠️', img_dict.get('right_wrist')),
                        ]
                        for ax, (cam_key, cam_label, cam_img) in zip(axes, camera_info):
                            if cam_img is not None and spatial_maps.get(cam_key) is not None:
                                overlay = create_attention_heatmap_overlay(cam_img, spatial_maps[cam_key], alpha=0.6)
                                ax.imshow(overlay)
                            else:
                                ax.text(0.5, 0.5, "No image", ha='center', va='center')
                            ax.set_title(cam_label, fontsize=11, fontweight='bold')
                            ax.axis('off')

                        # Add overall title with metrics
                        title = f"Inf {inf_step}, Denoise {denoise_step} (t={data.time:.1f})\n"
                        title += f"Head: {data.per_camera.head_camera:.1%}, Left: {data.per_camera.left_wrist:.1%}, Right: {data.per_camera.right_wrist:.1%}"
                        plt.suptitle(title, fontsize=10)
                        plt.tight_layout()
                        plt.savefig(heatmaps_dir / f"inf_{inf_step:04d}_denoise_{denoise_step:02d}_3panel.png", dpi=150, bbox_inches='tight')
                        plt.close()

    # Also save a summary per-camera spatial heatmap for the last inference step (backward compatible)
    if mid_denoise_step in all_step_data[last_inf_step]:
        data = all_step_data[last_inf_step][mid_denoise_step]
        if data.per_camera:
            spatial_maps = {
                'head': data.per_camera.head_spatial,
                'left_wrist': data.per_camera.left_wrist_spatial,
                'right_wrist': data.per_camera.right_wrist_spatial,
            }
            plot_per_camera_spatial_heatmaps(
                spatial_maps,
                images=last_images if isinstance(last_images, dict) else None,
                output_path=output_dir / "per_camera_spatial_heatmaps.png",
                title=f"Per-Camera Spatial Attention (inf {last_inf_step}, denoise {mid_denoise_step})",
            )

    # Store in analysis with nested structure (includes per-camera data)
    analysis_dict = {}
    for inf_step, denoise_dict in all_step_data.items():
        analysis_dict[f"inf_{inf_step}"] = {}
        for denoise_step, data in denoise_dict.items():
            step_dict = {
                "step": data.step,
                "time": data.time,
                "layer_idx": data.layer_idx,
                "image_attention_ratio": data.image_attention_ratio,
                "language_attention_ratio": data.language_attention_ratio,
                "state_attention_ratio": data.state_attention_ratio,
                "attention_entropy": data.attention_entropy,
            }
            # Add per-camera breakdown if available
            if data.per_camera:
                step_dict["per_camera"] = {
                    "head_camera": data.per_camera.head_camera,
                    "left_wrist": data.per_camera.left_wrist,
                    "right_wrist": data.per_camera.right_wrist,
                }
            analysis_dict[f"inf_{inf_step}"][f"denoise_{denoise_step}"] = step_dict

    # Save analysis data
    analysis_data = {
        "case_dir": str(case_dir),
        "task": task,
        "timestamp": analysis.timestamp,
        "avg_image_attention": analysis.avg_image_attention,
        "avg_language_attention": analysis.avg_language_attention,
        "avg_entropy": analysis.avg_entropy,
        "inference_steps": list(all_step_data.keys()),
        "attention_data": analysis_dict,
    }

    with open(output_dir / "cross_attention_analysis.json", 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"\nAnalysis saved to: {output_dir}")
    print(f"  - temporal_evolution.png")
    print(f"  - attention_metrics.png")
    print(f"  - per_camera_attention.png  ← Check right wrist attention!")
    print(f"  - per_camera_spatial_heatmaps.png  ← Side-by-side camera heatmaps")
    print(f"  - heatmaps/")
    print(f"      - inf_XXXX_per_camera.png  ← 3-panel heatmaps for ALL inference steps")
    print(f"      - inf_XXXX_denoise_XX_3panel.png  ← 3-panel for key denoise steps (0,5,9)")
    print(f"      - *.npy files for quantitative analysis")
    print(f"  - cross_attention_analysis.json (with per_camera breakdown)")

    return analysis


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Attention Capture for SmolVLA Hallucination Investigation"
    )

    parser.add_argument("--case-dir", type=Path, required=True,
                       help="Case directory with images and metadata")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                       help="Output directory for analysis results. "
                            "Default: logs/analysis/{case_name}/cross_attention/")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                       help="Model checkpoint (uses case metadata if not specified)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                       help="Compare two cases side-by-side")
    parser.add_argument("--case1", type=Path, help="First case for comparison")
    parser.add_argument("--case2", type=Path, help="Second case for comparison")

    args = parser.parse_args()

    # Set default output directory based on case name if not specified
    if args.output_dir is None:
        case_name = args.case_dir.name
        args.output_dir = Path("logs/analysis") / case_name / "cross_attention"
        print(f"Using default output directory: {args.output_dir}")

    print("=" * 60)
    print("SmolVLA Cross-Attention Analysis")
    print("=" * 60)

    if args.compare:
        if not args.case1 or not args.case2:
            print("ERROR: --case1 and --case2 required for comparison mode")
            sys.exit(1)

        print(f"Comparing:")
        print(f"  Case 1: {args.case1}")
        print(f"  Case 2: {args.case2}")

        out1 = args.output_dir / "case1"
        out2 = args.output_dir / "case2"

        analyze_case_cross_attention(args.case1, out1, args.checkpoint, args.device)
        analyze_case_cross_attention(args.case2, out2, args.checkpoint, args.device)

        # TODO: Generate comparison visualization

    else:
        analyze_case_cross_attention(
            args.case_dir,
            args.output_dir,
            args.checkpoint,
            args.device,
        )


if __name__ == "__main__":
    main()
