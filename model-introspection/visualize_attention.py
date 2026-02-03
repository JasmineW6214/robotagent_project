#!/usr/bin/env python3
"""
Enhanced Attention Visualization for SmolVLA Hallucination Investigation.

Extends the base attention visualization with:
1. Runtime attention capture during live inference
2. Spatial attention heatmaps overlaid on images
3. Cross-attention analysis between action expert and VLM embeddings
4. Temporal evolution of attention through inference steps

Key Questions:
- Where does the model attend when hallucinating?
- Is attention on distractor objects correlated with hallucination?
- Does attention entropy differ between normal and hallucination cases?

Usage:
    # Analyze with saved images from a case
    python visualize_attention.py \
        --checkpoint outputs/smolvla_bimanual \
        --case-dir ../cases/hallucination/case_001 \
        --output-dir ../reports/attention_analysis

    # Compare attention between two cases
    python visualize_attention.py \
        --compare \
        --case1 ../cases/hallucination/case_001 \
        --case2 ../cases/normal/case_002 \
        --output-dir ../reports/attention_comparison
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AttentionData:
    """Container for attention weights from a single inference step."""
    step: int
    timestamp: float
    task: str

    # Cross-attention weights (action expert attending to VLM)
    # Shape: [num_heads, action_tokens, vlm_tokens]
    cross_attention: Optional[np.ndarray] = None

    # Self-attention weights within VLM
    # Shape: [num_heads, seq_len, seq_len]
    vlm_attention: Optional[np.ndarray] = None

    # Token information
    tokens: list[str] = field(default_factory=list)
    num_image_patches: int = 0
    num_language_tokens: int = 0

    # Computed metrics
    attention_entropy: float = 0.0
    image_attention_ratio: float = 0.0
    language_attention_ratio: float = 0.0


# ============================================================================
# ATTENTION CAPTURE
# ============================================================================

class AttentionCaptureHook:
    """Hook to capture attention weights during forward pass."""

    def __init__(self):
        self.attention_weights = {}
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on attention layers."""
        for name, module in model.named_modules():
            # Look for attention layers in the expert model
            if 'cross_attn' in name.lower() or 'crossattention' in name.lower():
                hook = module.register_forward_hook(self._make_hook(f"cross_{name}"))
                self.hooks.append(hook)
                print(f"  Registered hook: {name}")

            # VLM attention layers
            elif 'self_attn' in name.lower() and 'vlm' in name.lower():
                hook = module.register_forward_hook(self._make_hook(f"vlm_{name}"))
                self.hooks.append(hook)
                print(f"  Registered hook: {name}")

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            # Different models return attention differently
            if hasattr(output, 'attentions') and output.attentions is not None:
                self.attention_weights[layer_name] = output.attentions
            elif isinstance(output, tuple):
                # Try to find attention in tuple output
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and o.dim() >= 3:
                        # Likely attention weights if shape is [batch, heads, seq, seq]
                        if o.shape[-1] == o.shape[-2] or 'attn' in layer_name:
                            self.attention_weights[f"{layer_name}_{i}"] = o
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.attention_weights = {}

    def get_weights(self) -> dict:
        """Get captured weights as numpy arrays."""
        result = {}
        for name, weights in self.attention_weights.items():
            if isinstance(weights, torch.Tensor):
                # Convert bfloat16 to float32 before numpy conversion
                result[name] = weights.detach().float().cpu().numpy()
            elif isinstance(weights, tuple):
                for i, w in enumerate(weights):
                    if isinstance(w, torch.Tensor):
                        result[f"{name}_{i}"] = w.detach().float().cpu().numpy()
        return result


# ============================================================================
# ATTENTION ANALYSIS
# ============================================================================

def compute_attention_entropy(attention: np.ndarray) -> float:
    """
    Compute entropy of attention distribution.

    Higher entropy = more diffuse attention (less focused)
    Lower entropy = more concentrated attention (more focused)
    """
    # Flatten and normalize
    attn_flat = attention.flatten()
    attn_flat = attn_flat / (attn_flat.sum() + 1e-10)

    # Compute entropy
    entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
    return float(entropy)


def compute_spatial_attention_map(
    attention: np.ndarray,
    query_idx: int,
    num_patches_h: int = 16,
    num_patches_w: int = 16,
) -> np.ndarray:
    """
    Extract spatial attention map for a specific query token.

    Args:
        attention: Attention weights [num_heads, queries, keys]
        query_idx: Which query to visualize
        num_patches_h: Image patches vertically (SigLIP default: 16)
        num_patches_w: Image patches horizontally

    Returns:
        Attention map [num_patches_h, num_patches_w]
    """
    # Average over heads
    if attention.ndim == 3:
        attn = attention.mean(axis=0)  # [queries, keys]
    else:
        attn = attention

    # Get attention from query to image patches
    num_patches = num_patches_h * num_patches_w

    # Assume image patches are at beginning of sequence
    attn_to_image = attn[query_idx, :num_patches]

    # Reshape to spatial grid
    attn_map = attn_to_image.reshape(num_patches_h, num_patches_w)

    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-10)

    return attn_map


def analyze_token_attention(
    attention: np.ndarray,
    tokens: list[str],
    target_words: list[str] = None,
) -> dict:
    """
    Analyze which tokens receive most attention.

    Returns dict with per-token attention statistics.
    """
    if target_words is None:
        target_words = ['plate', 'bin', 'pick', 'place', 'banana', 'orange', 'yogurt']

    # Average over heads if needed
    if attention.ndim == 3:
        attn = attention.mean(axis=0)
    else:
        attn = attention

    # Sum attention TO each token (column-wise)
    attention_received = attn.sum(axis=0)
    attention_received = attention_received / (attention_received.sum() + 1e-10)

    # Find target word indices and their attention
    target_attention = {}
    for i, token in enumerate(tokens):
        token_lower = token.lower().strip()
        for target in target_words:
            if target in token_lower:
                target_attention[f"{token}_{i}"] = {
                    "index": i,
                    "token": token,
                    "attention_received": float(attention_received[i]) if i < len(attention_received) else 0,
                }

    # Overall statistics
    return {
        "total_tokens": len(tokens),
        "attention_entropy": compute_attention_entropy(attn),
        "target_words_found": len(target_attention),
        "target_attention": target_attention,
        "max_attention_idx": int(np.argmax(attention_received)),
        "max_attention_value": float(np.max(attention_received)),
    }


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
    attn_resized = cv2.resize(attention_map, (w, h))

    # Create heatmap using jet colormap
    heatmap = plt.cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def plot_attention_summary(
    attention: np.ndarray,
    tokens: list[str],
    output_path: Path,
    title: str = "Attention Distribution",
    highlight_words: list[str] = None,
):
    """
    Plot bar chart of attention distribution across tokens.
    """
    if highlight_words is None:
        highlight_words = ['plate', 'bin', 'pick', 'place']

    # Average over heads
    if attention.ndim == 3:
        attn = attention.mean(axis=0)
    else:
        attn = attention

    # Sum attention TO each token
    attention_received = attn.sum(axis=0)
    attention_received = attention_received / (attention_received.sum() + 1e-10)

    # Limit to token count
    n = min(len(tokens), len(attention_received))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n)
    bars = ax.bar(x, attention_received[:n], color='steelblue', alpha=0.7)

    # Highlight target words
    for i, token in enumerate(tokens[:n]):
        token_lower = token.lower().strip()
        for target in highlight_words:
            if target in token_lower:
                bars[i].set_color('red')
                bars[i].set_alpha(0.9)
                break

    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Attention Received', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens[:n], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention summary: {output_path}")


def plot_spatial_attention_grid(
    image: np.ndarray,
    attention_maps: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    title: str = "Spatial Attention Maps",
):
    """
    Plot grid of spatial attention maps.

    Args:
        image: Original image
        attention_maps: List of attention maps to visualize
        labels: Labels for each map
        output_path: Where to save
        title: Plot title
    """
    n = len(attention_maps)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols + 1, figsize=(4 * (cols + 1), 4 * rows))
    if rows == 1:
        axes = [axes]

    # Show original image in first column
    for row in range(rows):
        axes[row][0].imshow(image)
        axes[row][0].set_title('Original')
        axes[row][0].axis('off')

    # Show attention maps
    for i, (attn_map, label) in enumerate(zip(attention_maps, labels)):
        row = i // cols
        col = (i % cols) + 1

        overlay = create_attention_heatmap_overlay(image, attn_map)
        axes[row][col].imshow(overlay)
        axes[row][col].set_title(label, fontsize=10)
        axes[row][col].axis('off')

    # Hide unused subplots
    for row in range(rows):
        for col in range(n % cols + 1 if row == rows - 1 and n % cols != 0 else cols + 1, cols + 1):
            if row < len(axes) and col < len(axes[row]):
                axes[row][col].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spatial attention grid: {output_path}")


def plot_attention_comparison(
    attn1: np.ndarray,
    attn2: np.ndarray,
    tokens: list[str],
    output_path: Path,
    label1: str = "Hallucination",
    label2: str = "Normal",
):
    """
    Compare attention distributions between two cases.
    """
    # Average over heads
    if attn1.ndim == 3:
        attn1 = attn1.mean(axis=0)
    if attn2.ndim == 3:
        attn2 = attn2.mean(axis=0)

    # Sum attention received
    received1 = attn1.sum(axis=0)
    received1 = received1 / (received1.sum() + 1e-10)
    received2 = attn2.sum(axis=0)
    received2 = received2 / (received2.sum() + 1e-10)

    n = min(len(tokens), len(received1), len(received2))

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n)
    width = 0.35

    ax.bar(x - width/2, received1[:n], width, label=label1, color='red', alpha=0.7)
    ax.bar(x + width/2, received2[:n], width, label=label2, color='blue', alpha=0.7)

    ax.set_xlabel('Tokens', fontsize=12)
    ax.set_ylabel('Attention Received', fontsize=12)
    ax.set_title(f'Attention Comparison: {label1} vs {label2}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tokens[:n], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention comparison: {output_path}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_case_attention(
    checkpoint_path: str,
    case_dir: Path,
    output_dir: Path,
    device: str = "cuda",
):
    """
    Analyze attention for a single case using saved images.
    """
    print(f"Analyzing case: {case_dir}")

    # Load metadata
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {case_dir}")
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    task = metadata.get("task_original", "Unknown task")
    print(f"Task: {task}")

    # Find images
    images_dir = case_dir / "images"
    if not images_dir.exists():
        print(f"WARNING: No images directory found. Using placeholder analysis.")
        return _analyze_without_images(checkpoint_path, task, output_dir, device)

    # Load first image for analysis
    image_files = sorted(images_dir.glob("step_*_head.jpg"))
    if not image_files:
        print("WARNING: No head camera images found")
        return None

    # Load model
    print("Loading SmolVLA policy...")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Setup attention capture
    capture = AttentionCaptureHook()
    capture.register_hooks(policy.model)

    # Load preprocessor for proper observation formatting
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # Load dataset stats for normalization
    dataset_path = PROJECT_ROOT / "datasets_bimanuel" / "multitasks"
    dataset_metadata = LeRobotDatasetMetadata(repo_id="multitasks", root=str(dataset_path))

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # Process representative images (need all 3 cameras per step)
    results = []
    all_step_numbers = sorted(set(int(p.stem.split('_')[1]) for p in image_files))

    # Focus on critical steps for hallucination analysis (200, 250, 300)
    # Also include some early steps for comparison
    priority_steps = [0, 100, 200, 250, 300, 350]
    step_numbers = [s for s in priority_steps if s in all_step_numbers]
    # Add remaining steps if we don't have enough
    if len(step_numbers) < 6:
        remaining = [s for s in all_step_numbers if s not in step_numbers]
        step_numbers.extend(remaining[:6 - len(step_numbers)])
    step_numbers = sorted(step_numbers)

    for step_num in step_numbers:
        print(f"  Processing step {step_num}...")

        # Load all 3 camera images for this step
        head_path = images_dir / f"step_{step_num:04d}_head.jpg"
        left_wrist_path = images_dir / f"step_{step_num:04d}_left_wrist.jpg"
        right_wrist_path = images_dir / f"step_{step_num:04d}_right_wrist.jpg"

        if not all(p.exists() for p in [head_path, left_wrist_path, right_wrist_path]):
            print(f"    Missing camera images for step {step_num}")
            continue

        # Load images (BGR -> RGB)
        head_img = cv2.cvtColor(cv2.imread(str(head_path)), cv2.COLOR_BGR2RGB)
        left_wrist_img = cv2.cvtColor(cv2.imread(str(left_wrist_path)), cv2.COLOR_BGR2RGB)
        right_wrist_img = cv2.cvtColor(cv2.imread(str(right_wrist_path)), cv2.COLOR_BGR2RGB)

        # Create observation matching SmolVLA expected format
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

        # Preprocess observation
        preprocessed_obs = preprocessor(observation)

        # Reset policy to force new chunk generation (otherwise uses cached actions)
        policy.reset()

        # Run inference with attention capture
        capture.clear()

        try:
            with torch.no_grad():
                action = policy.select_action(preprocessed_obs)
            print(f"    Inference OK, captured {len(capture.attention_weights)} attention layers")
        except Exception as e:
            print(f"    Inference error: {e}")
            continue

        # Collect attention data
        attn_weights = capture.get_weights()
        if attn_weights:
            results.append({
                "step": step_num,
                "image": f"step_{step_num:04d}_head.jpg",
                "head_image": head_img,
                "attention_layers": list(attn_weights.keys()),
                "weights": attn_weights,
            })

    capture.remove_hooks()

    # Generate visualizations
    if results:
        _generate_attention_visualizations(results, task, output_dir)

    return results


def _analyze_without_images(
    checkpoint_path: str,
    task: str,
    output_dir: Path,
    device: str,
) -> dict:
    """Analyze tokenization and model structure without images."""
    print("Performing tokenization-only analysis...")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            trust_remote_code=True
        )

        tokens = tokenizer.tokenize(task)

        analysis = {
            "task": task,
            "tokens": tokens,
            "num_tokens": len(tokens),
            "target_words": [],
        }

        # Find target words
        target_words = ['plate', 'bin', 'pick', 'place', 'banana', 'orange', 'yogurt']
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip()
            for target in target_words:
                if target in token_lower:
                    analysis["target_words"].append({
                        "word": target,
                        "token": token,
                        "index": i,
                    })

        # Save analysis
        with open(output_dir / "tokenization_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"Tokenization analysis saved to: {output_dir}")
        return analysis

    except Exception as e:
        print(f"Tokenization analysis failed: {e}")
        return None


def _generate_attention_visualizations(results: list, task: str, output_dir: Path):
    """Generate visualization plots from attention results."""
    print("Generating attention visualizations...")

    # Save raw metadata
    with open(output_dir / "attention_data.json", 'w') as f:
        serializable = []
        for r in results:
            sr = {"image": r["image"], "attention_layers": r["attention_layers"]}
            serializable.append(sr)
        json.dump(serializable, f, indent=2)

    # Create subdirectories for visualizations
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(exist_ok=True)

    # SigLIP default patch configuration
    # SmolVLA uses SigLIP-SO400M which has 14x14 patch size, 384x384 input
    # This gives 27x27 = 729 patches (plus CLS token = 730 total)
    # But attention is [seq_len, seq_len] where seq_len includes CLS token
    PATCH_SIZE = 27  # 384/14 = 27.4 -> 27 patches per dimension
    CLS_OFFSET = 1   # First token is CLS token

    for result in results:
        step = result.get("step", 0)
        image = result.get("head_image")
        weights = result.get("weights", {})

        if image is None:
            print(f"  Skipping step {step}: no image")
            continue

        print(f"  Generating heatmaps for step {step}...")

        # Collect attention maps from different layers
        layer_attention_maps = []
        layer_labels = []

        for layer_name, attn_weight in weights.items():
            # Skip if not from vision encoder
            if "vision_model" not in layer_name:
                continue

            # attn_weight shape: [batch, num_heads, seq_len, seq_len]
            # or [num_heads, seq_len, seq_len]
            if attn_weight.ndim == 4:
                attn_weight = attn_weight[0]  # Remove batch dimension

            num_heads, seq_len, _ = attn_weight.shape

            # Extract layer number from name
            layer_num = None
            for part in layer_name.split('.'):
                if part.isdigit():
                    layer_num = int(part)
                    break

            # Compute spatial attention map
            # Average attention FROM the CLS token TO all patches (shows what CLS attends to)
            # CLS token is at index 0, patches start at index 1
            if seq_len > PATCH_SIZE * PATCH_SIZE:
                # CLS attends to patches: use attention from CLS (row 0) to patches (cols 1:)
                cls_to_patches = attn_weight[:, 0, CLS_OFFSET:CLS_OFFSET + PATCH_SIZE * PATCH_SIZE]
                # Average over heads
                avg_attn = cls_to_patches.mean(axis=0)
            else:
                # No CLS token, just patches
                # Use average attention received by each patch (column-wise sum)
                avg_attn = attn_weight.mean(axis=(0, 1))[:PATCH_SIZE * PATCH_SIZE]

            # Reshape to spatial grid
            try:
                spatial_attn = avg_attn.reshape(PATCH_SIZE, PATCH_SIZE)
            except ValueError:
                # Try to infer patch grid from attention size
                num_patches = len(avg_attn)
                patch_side = int(np.sqrt(num_patches))
                if patch_side * patch_side == num_patches:
                    spatial_attn = avg_attn.reshape(patch_side, patch_side)
                else:
                    print(f"    Cannot reshape layer {layer_name}: {num_patches} patches")
                    continue

            # Normalize
            spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-10)

            layer_attention_maps.append(spatial_attn)
            layer_labels.append(f"L{layer_num}" if layer_num is not None else layer_name.split('.')[-2])

        if not layer_attention_maps:
            print(f"    No valid attention maps for step {step}")
            continue

        # Generate individual heatmap overlays for key layers
        # Focus on early (L0-L2), middle (L5-L6), and late (L10-L11) layers
        key_layer_indices = []
        for target in [0, 1, 2, 5, 6, 10, 11]:
            for i, label in enumerate(layer_labels):
                if f"L{target}" == label:
                    key_layer_indices.append(i)
                    break

        # If no key layers found, use first, middle, last
        if not key_layer_indices:
            key_layer_indices = [0, len(layer_attention_maps)//2, len(layer_attention_maps)-1]
            key_layer_indices = [i for i in key_layer_indices if i < len(layer_attention_maps)]

        # Create combined visualization
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Row 1: Original + early layers
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original", fontsize=10)
        axes[0, 0].axis('off')

        for col, idx in enumerate(key_layer_indices[:3]):
            if idx < len(layer_attention_maps):
                overlay = create_attention_heatmap_overlay(image, layer_attention_maps[idx], alpha=0.5)
                axes[0, col + 1].imshow(overlay)
                axes[0, col + 1].set_title(f"{layer_labels[idx]} Attention", fontsize=10)
                axes[0, col + 1].axis('off')

        # Row 2: Late layers + average
        for col, idx in enumerate(key_layer_indices[3:6]):
            if idx < len(layer_attention_maps):
                overlay = create_attention_heatmap_overlay(image, layer_attention_maps[idx], alpha=0.5)
                axes[1, col].imshow(overlay)
                axes[1, col].set_title(f"{layer_labels[idx]} Attention", fontsize=10)
                axes[1, col].axis('off')

        # Average across all layers
        avg_all = np.mean(layer_attention_maps, axis=0)
        avg_all = (avg_all - avg_all.min()) / (avg_all.max() - avg_all.min() + 1e-10)
        overlay_avg = create_attention_heatmap_overlay(image, avg_all, alpha=0.5)
        axes[1, 3].imshow(overlay_avg)
        axes[1, 3].set_title("Average All Layers", fontsize=10)
        axes[1, 3].axis('off')

        # Hide unused axes
        for ax in axes.flat:
            if not ax.images:
                ax.axis('off')

        plt.suptitle(f"Step {step}: Vision Attention Heatmaps\nTask: {task[:60]}...", fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save combined visualization
        combined_path = heatmaps_dir / f"step_{step:04d}_attention_combined.png"
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {combined_path.name}")

        # Save individual average heatmap for easy comparison
        single_fig, single_ax = plt.subplots(figsize=(8, 8))
        single_ax.imshow(overlay_avg)
        single_ax.set_title(f"Step {step}: Average Attention", fontsize=12)
        single_ax.axis('off')

        single_path = heatmaps_dir / f"step_{step:04d}_attention_avg.png"
        plt.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Generate summary across all steps
    if len(results) > 1:
        _generate_temporal_summary(results, task, output_dir, PATCH_SIZE)

    print(f"Attention analysis saved to: {output_dir}")


def _generate_temporal_summary(results: list, task: str, output_dir: Path, patch_size: int = 27):
    """Generate summary showing attention evolution across time steps."""
    print("  Generating temporal summary...")

    # Collect average attention maps per step
    step_maps = []
    step_nums = []

    for result in results:
        step = result.get("step", 0)
        weights = result.get("weights", {})

        all_layer_maps = []
        for layer_name, attn_weight in weights.items():
            if "vision_model" not in layer_name:
                continue

            if attn_weight.ndim == 4:
                attn_weight = attn_weight[0]

            num_heads, seq_len, _ = attn_weight.shape

            # Use last layer for summary (most semantic)
            if "layers.11" not in layer_name:
                continue

            if seq_len > patch_size * patch_size:
                cls_to_patches = attn_weight[:, 0, 1:1 + patch_size * patch_size]
                avg_attn = cls_to_patches.mean(axis=0)
            else:
                avg_attn = attn_weight.mean(axis=(0, 1))[:patch_size * patch_size]

            try:
                spatial_attn = avg_attn.reshape(patch_size, patch_size)
                spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-10)
                all_layer_maps.append(spatial_attn)
            except ValueError:
                continue

        if all_layer_maps:
            step_maps.append(np.mean(all_layer_maps, axis=0))
            step_nums.append(step)

    if len(step_maps) < 2:
        return

    # Plot temporal evolution
    n_steps = len(step_maps)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    if n_steps == 1:
        axes = [axes]

    for i, (step_map, step_num) in enumerate(zip(step_maps, step_nums)):
        im = axes[i].imshow(step_map, cmap='jet', vmin=0, vmax=1)
        axes[i].set_title(f"Step {step_num}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f"Attention Evolution (Last Layer)\nTask: {task[:50]}...", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes, shrink=0.6, label='Attention')
    plt.tight_layout()

    summary_path = output_dir / "attention_temporal_evolution.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved temporal summary: {summary_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Attention Visualization for SmolVLA Hallucination Investigation"
    )

    # Mode
    parser.add_argument("--compare", action="store_true", help="Compare two cases")

    # Single case analysis
    parser.add_argument("--checkpoint", "-c", help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dir", type=Path, help="Case directory with images")

    # Comparison mode
    parser.add_argument("--case1", type=Path, help="First case for comparison")
    parser.add_argument("--case2", type=Path, help="Second case for comparison")

    # Output
    parser.add_argument("--output-dir", "-o", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SmolVLA Enhanced Attention Visualization")
    print("=" * 60)

    if args.compare:
        if not args.case1 or not args.case2:
            print("ERROR: --case1 and --case2 required for comparison")
            sys.exit(1)

        if not args.checkpoint:
            print("ERROR: --checkpoint required")
            sys.exit(1)

        print(f"Comparing:")
        print(f"  Case 1: {args.case1}")
        print(f"  Case 2: {args.case2}")

        # Analyze both cases
        out1 = args.output_dir / "case1"
        out2 = args.output_dir / "case2"
        out1.mkdir(exist_ok=True)
        out2.mkdir(exist_ok=True)

        analyze_case_attention(args.checkpoint, args.case1, out1, args.device)
        analyze_case_attention(args.checkpoint, args.case2, out2, args.device)

    else:
        if not args.checkpoint or not args.case_dir:
            print("ERROR: --checkpoint and --case-dir required for single case analysis")
            sys.exit(1)

        analyze_case_attention(
            args.checkpoint,
            args.case_dir,
            args.output_dir,
            args.device,
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
