#!/usr/bin/env python3
"""
KV Cache Content Analysis Tool for SmolVLA Hallucination Investigation.

Captures and analyzes the KV (Key-Value) cache generated from the prefix embedding.
The KV cache is computed once per action chunk and reused for all 10 denoising steps.

This tool helps answer:
1. Is the KV difference concentrated in specific transformer layers?
2. Which token region (head cam, left wrist, right wrist, language, state) has largest KV difference?
3. Does KV difference correlate with action difference?

KV Cache Structure:
  - past_key_values: List[Tuple[key_states, value_states]]
  - Each: [batch, num_heads, seq_len, head_dim]
  - 16 layers (for SmolVLA base), 16 heads per layer, 72 head_dim

Usage:
    python kv_cache_content_analysis.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir outputs/kv_cache_analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

PATCHES_PER_CAMERA = 64
NUM_CAMERAS = 3
IMAGE_SIZE = 512
# Camera names as they appear in case directories
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
# Camera names as expected by the model config
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]

# Token indices
HEAD_CAMERA_START = 0
HEAD_CAMERA_END = 64
LEFT_WRIST_START = 64
LEFT_WRIST_END = 128
RIGHT_WRIST_START = 128
RIGHT_WRIST_END = 192
LANGUAGE_START = 192
LANGUAGE_END = 240
STATE_INDEX = 240
TOTAL_PREFIX_TOKENS = 241

TOKEN_REGIONS = {
    "head_camera": (HEAD_CAMERA_START, HEAD_CAMERA_END),
    "left_wrist": (LEFT_WRIST_START, LEFT_WRIST_END),
    "right_wrist": (RIGHT_WRIST_START, RIGHT_WRIST_END),
    "language": (LANGUAGE_START, LANGUAGE_END),
    "state": (STATE_INDEX, STATE_INDEX + 1)
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LayerKVStats:
    """Statistics for a single transformer layer's KV cache."""
    layer_idx: int

    # Key statistics
    key_mean: float
    key_std: float
    key_norm: float

    # Value statistics
    value_mean: float
    value_std: float
    value_norm: float

    # Per-region norms (key)
    key_region_norms: Dict[str, float] = field(default_factory=dict)

    # Per-region norms (value)
    value_region_norms: Dict[str, float] = field(default_factory=dict)


@dataclass
class CaseKVCache:
    """KV cache data for a single case."""
    case_name: str
    step: int
    num_layers: int
    num_heads: int
    head_dim: int
    seq_len: int

    layer_stats: List[LayerKVStats]

    # Raw tensors stored as numpy for comparison
    key_states: Optional[np.ndarray] = None  # [num_layers, num_heads, seq_len, head_dim]
    value_states: Optional[np.ndarray] = None


@dataclass
class LayerComparison:
    """Comparison of a single layer's KV cache between two cases."""
    layer_idx: int

    # Key comparison
    key_l2_distance: float
    key_cosine_similarity: float
    key_region_l2: Dict[str, float]

    # Value comparison
    value_l2_distance: float
    value_cosine_similarity: float
    value_region_l2: Dict[str, float]

    # Combined
    total_kv_l2: float


@dataclass
class KVCacheComparison:
    """Full comparison of KV cache between two cases."""
    case_a: str
    case_b: str
    step: int

    layer_comparisons: List[LayerComparison]

    # Aggregated metrics
    total_key_l2: float
    total_value_l2: float
    total_kv_l2: float

    # Which layers differ most
    most_different_layer: int
    most_different_layer_l2: float

    # Which region differs most across all layers
    most_different_region: str
    region_total_l2: Dict[str, float]


# ============================================================================
# KV CACHE EXTRACTION
# ============================================================================

class KVCacheExtractor:
    """Extracts KV cache from SmolVLA model."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_model(self):
        """Load the SmolVLA model."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print(f"Loading model from {self.checkpoint_path}...")
        self.model = SmolVLAPolicy.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def load_case_data(self, case_dir: str, step: int) -> dict:
        """Load images and trace data from case directory."""
        case_path = Path(case_dir)

        # Load images
        images = []
        for camera_file_name in CAMERA_FILE_NAMES:
            image_path = case_path / "images" / f"step_{step:04d}_{camera_file_name}.jpg"
            if not image_path.exists():
                image_path = case_path / "images" / f"step_{step:04d}_{camera_name}.png"

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            images.append(img_tensor)

        # Load state from trace
        trace_path = case_path / "trace.jsonl"
        state = None
        with open(trace_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["step"] == step:
                    state = data["state_normalized"]
                    break

        if state is None:
            raise ValueError(f"Could not find step {step} in trace")

        # Load task from metadata
        metadata_path = case_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        task = metadata.get("task_original", metadata.get("task", ""))

        return {
            "images": images,
            "state": state,
            "task": task,
            "case_name": case_path.name
        }

    def prepare_batch(self, case_data: dict) -> dict:
        """Prepare batch for model inference."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

        # Stack images
        images = []
        for img in case_data["images"]:
            images.append(img.unsqueeze(0))

        # Process state
        state = torch.tensor(case_data["state"], dtype=torch.float32).unsqueeze(0)

        # Tokenize task
        processor = self.model.model.vlm_with_expert.processor
        task_with_newline = case_data["task"] + "\n" if not case_data["task"].endswith("\n") else case_data["task"]

        tokenized = processor.tokenizer(
            task_with_newline,
            padding="max_length",
            max_length=48,
            truncation=True,
            return_tensors="pt"
        )

        batch = {
            OBS_STATE: state.to(self.device),
            OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(self.device),
            # Convert attention_mask to boolean (tokenizer returns int64, but model expects bool)
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].bool().to(self.device),
        }

        # Use model's expected camera names (camera1, camera2, camera3)
        for idx, camera_model_name in enumerate(CAMERA_MODEL_NAMES):
            key = f"observation.images.{camera_model_name}"
            batch[key] = images[idx].to(self.device)

        return batch

    def extract_kv_cache(self, case_dir: str, step: int) -> CaseKVCache:
        """Extract KV cache for a case at a specific step."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        case_data = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(case_data)

        # Prepare inputs
        # prepare_images/prepare_state are on SmolVLAPolicy (self.model)
        images, img_masks = self.model.prepare_images(batch)
        state = self.model.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        with torch.no_grad():
            # Get prefix embeddings
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Compute KV cache
            _, past_key_values = self.model.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.config.use_cache,
                fill_kv_cache=True,
            )

        # Extract and analyze KV cache
        # past_key_values is a dict: {layer_idx: {"key_states": tensor, "value_states": tensor}}
        # key_states shape: [batch, seq_len, num_kv_heads, head_dim] (grouped query attention)
        num_layers = len(past_key_values)
        first_layer = past_key_values[0]
        first_key = first_layer["key_states"]
        seq_len = first_key.shape[1]
        num_heads = first_key.shape[2]
        head_dim = first_key.shape[3]

        # Stack all layers
        all_keys = []
        all_values = []
        layer_stats = []

        for layer_idx in range(num_layers):
            layer_data = past_key_values[layer_idx]
            key_states = layer_data["key_states"]
            value_states = layer_data["value_states"]
            # Convert to numpy
            # Convert to float32 before numpy (BFloat16 not supported by numpy)
            # Shape after [0]: [seq_len, num_kv_heads, head_dim]
            key_np = key_states[0].float().cpu().numpy()
            value_np = value_states[0].float().cpu().numpy()

            all_keys.append(key_np)
            all_values.append(value_np)

            # Compute per-region norms for keys (slicing on seq_len dimension)
            key_region_norms = {}
            for region_name, (start, end) in TOKEN_REGIONS.items():
                region_key = key_np[start:end, :, :]  # [region_len, heads, dim]
                key_region_norms[region_name] = float(np.linalg.norm(region_key))

            # Compute per-region norms for values
            value_region_norms = {}
            for region_name, (start, end) in TOKEN_REGIONS.items():
                region_value = value_np[start:end, :, :]
                value_region_norms[region_name] = float(np.linalg.norm(region_value))

            stats = LayerKVStats(
                layer_idx=layer_idx,
                key_mean=float(np.mean(key_np)),
                key_std=float(np.std(key_np)),
                key_norm=float(np.linalg.norm(key_np)),
                value_mean=float(np.mean(value_np)),
                value_std=float(np.std(value_np)),
                value_norm=float(np.linalg.norm(value_np)),
                key_region_norms=key_region_norms,
                value_region_norms=value_region_norms
            )
            layer_stats.append(stats)

        return CaseKVCache(
            case_name=case_data["case_name"],
            step=step,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            layer_stats=layer_stats,
            key_states=np.stack(all_keys),
            value_states=np.stack(all_values)
        )


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compare_kv_caches(kv_a: CaseKVCache, kv_b: CaseKVCache) -> KVCacheComparison:
    """Compare KV caches between two cases."""
    layer_comparisons = []
    region_total_l2 = {r: 0.0 for r in TOKEN_REGIONS.keys()}

    total_key_l2 = 0.0
    total_value_l2 = 0.0
    max_layer_l2 = 0.0
    max_layer_idx = 0

    for layer_idx in range(kv_a.num_layers):
        key_a = kv_a.key_states[layer_idx]
        key_b = kv_b.key_states[layer_idx]
        value_a = kv_a.value_states[layer_idx]
        value_b = kv_b.value_states[layer_idx]

        # Key comparison
        key_l2 = float(np.linalg.norm(key_a - key_b))
        key_cosine = cosine_similarity(key_a, key_b)

        # Value comparison
        value_l2 = float(np.linalg.norm(value_a - value_b))
        value_cosine = cosine_similarity(value_a, value_b)

        # Per-region L2 for keys (shape: [seq_len, num_heads, head_dim])
        key_region_l2 = {}
        for region_name, (start, end) in TOKEN_REGIONS.items():
            region_key_a = key_a[start:end, :, :]
            region_key_b = key_b[start:end, :, :]
            l2 = float(np.linalg.norm(region_key_a - region_key_b))
            key_region_l2[region_name] = l2
            region_total_l2[region_name] += l2

        # Per-region L2 for values (shape: [seq_len, num_heads, head_dim])
        value_region_l2 = {}
        for region_name, (start, end) in TOKEN_REGIONS.items():
            region_value_a = value_a[start:end, :, :]
            region_value_b = value_b[start:end, :, :]
            l2 = float(np.linalg.norm(region_value_a - region_value_b))
            value_region_l2[region_name] = l2
            region_total_l2[region_name] += l2

        layer_kv_l2 = key_l2 + value_l2

        layer_comparisons.append(LayerComparison(
            layer_idx=layer_idx,
            key_l2_distance=key_l2,
            key_cosine_similarity=key_cosine,
            key_region_l2=key_region_l2,
            value_l2_distance=value_l2,
            value_cosine_similarity=value_cosine,
            value_region_l2=value_region_l2,
            total_kv_l2=layer_kv_l2
        ))

        total_key_l2 += key_l2
        total_value_l2 += value_l2

        if layer_kv_l2 > max_layer_l2:
            max_layer_l2 = layer_kv_l2
            max_layer_idx = layer_idx

    # Find most different region
    most_diff_region = max(region_total_l2.keys(), key=lambda k: region_total_l2[k])

    return KVCacheComparison(
        case_a=kv_a.case_name,
        case_b=kv_b.case_name,
        step=kv_a.step,
        layer_comparisons=[asdict(lc) for lc in layer_comparisons],
        total_key_l2=total_key_l2,
        total_value_l2=total_value_l2,
        total_kv_l2=total_key_l2 + total_value_l2,
        most_different_layer=max_layer_idx,
        most_different_layer_l2=max_layer_l2,
        most_different_region=most_diff_region,
        region_total_l2=region_total_l2
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_layer_comparison(comparison: KVCacheComparison, output_dir: Path):
    """Visualize per-layer KV cache differences."""
    num_layers = len(comparison.layer_comparisons)
    layers = list(range(num_layers))

    key_l2 = [lc["key_l2_distance"] for lc in comparison.layer_comparisons]
    value_l2 = [lc["value_l2_distance"] for lc in comparison.layer_comparisons]
    total_l2 = [lc["total_kv_l2"] for lc in comparison.layer_comparisons]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: Key and Value L2 distances by layer
    ax1 = axes[0]
    width = 0.35
    x = np.array(layers)
    ax1.bar(x - width/2, key_l2, width, label='Key L2', color='blue', alpha=0.7)
    ax1.bar(x + width/2, value_l2, width, label='Value L2', color='red', alpha=0.7)
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("L2 Distance")
    ax1.set_title(f"KV Cache L2 Distance by Layer\n{comparison.case_a[:25]} vs {comparison.case_b[:25]}")
    ax1.legend()
    ax1.set_xticks(layers)

    # Highlight most different layer
    ax1.axvline(x=comparison.most_different_layer, color='green', linestyle='--',
                label=f'Most diff layer: {comparison.most_different_layer}')

    # Bottom: Per-region breakdown for most different layer
    ax2 = axes[1]
    most_diff = comparison.layer_comparisons[comparison.most_different_layer]
    regions = list(TOKEN_REGIONS.keys())
    x = np.arange(len(regions))

    key_region_l2 = [most_diff["key_region_l2"][r] for r in regions]
    value_region_l2 = [most_diff["value_region_l2"][r] for r in regions]

    ax2.bar(x - width/2, key_region_l2, width, label='Key L2', color='blue', alpha=0.7)
    ax2.bar(x + width/2, value_region_l2, width, label='Value L2', color='red', alpha=0.7)
    ax2.set_xlabel("Token Region")
    ax2.set_ylabel("L2 Distance")
    ax2.set_title(f"Layer {comparison.most_different_layer} (Most Different): Per-Region Breakdown")
    ax2.set_xticks(x)
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    safe_name = f"{comparison.case_a[:15]}_{comparison.case_b[:15]}"
    plt.savefig(output_dir / f"layer_comparison_{safe_name}.png", dpi=150)
    plt.close()


def visualize_region_by_layer_heatmap(comparison: KVCacheComparison, output_dir: Path):
    """Create heatmap of [layers x regions] for KV differences."""
    regions = list(TOKEN_REGIONS.keys())
    num_layers = len(comparison.layer_comparisons)

    # Build matrices
    key_matrix = np.zeros((num_layers, len(regions)))
    value_matrix = np.zeros((num_layers, len(regions)))

    for layer_idx, lc in enumerate(comparison.layer_comparisons):
        for region_idx, region_name in enumerate(regions):
            key_matrix[layer_idx, region_idx] = lc["key_region_l2"][region_name]
            value_matrix[layer_idx, region_idx] = lc["value_region_l2"][region_name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # Key heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(key_matrix, cmap='hot', aspect='auto')
    ax1.set_xlabel("Token Region")
    ax1.set_ylabel("Layer")
    ax1.set_title("Key States L2 Distance")
    ax1.set_xticks(range(len(regions)))
    ax1.set_xticklabels(regions, rotation=45, ha='right')
    ax1.set_yticks(range(num_layers))
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Value heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(value_matrix, cmap='hot', aspect='auto')
    ax2.set_xlabel("Token Region")
    ax2.set_ylabel("Layer")
    ax2.set_title("Value States L2 Distance")
    ax2.set_xticks(range(len(regions)))
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.set_yticks(range(num_layers))
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(f"KV Cache Difference: {comparison.case_a[:20]} vs {comparison.case_b[:20]}")
    plt.tight_layout()
    safe_name = f"{comparison.case_a[:15]}_{comparison.case_b[:15]}"
    plt.savefig(output_dir / f"region_by_layer_heatmap_{safe_name}.png", dpi=150)
    plt.close()


def visualize_region_totals(comparisons: List[KVCacheComparison], output_dir: Path):
    """Visualize total KV difference by region across all comparisons."""
    fig, ax = plt.subplots(figsize=(10, 6))

    regions = list(TOKEN_REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.25

    for idx, comp in enumerate(comparisons):
        values = [comp.region_total_l2[r] for r in regions]
        label = f"{comp.case_a[:10]}..vs..{comp.case_b[:10]}"
        ax.bar(x + idx * width, values, width, label=label)

    ax.set_xlabel("Token Region")
    ax.set_ylabel("Total L2 Distance (Sum across layers)")
    ax.set_title("Total KV Cache Difference by Region")
    ax.set_xticks(x + width * (len(comparisons) - 1) / 2)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "region_totals.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(caches: List[CaseKVCache], comparisons: List[KVCacheComparison],
                   output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# KV Cache Content Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"\n**Step analyzed**: {caches[0].step}")

    first_cache = caches[0]
    report.append(f"\n## KV Cache Structure")
    report.append(f"- Number of layers: {first_cache.num_layers}")
    report.append(f"- Number of heads: {first_cache.num_heads}")
    report.append(f"- Head dimension: {first_cache.head_dim}")
    report.append(f"- Sequence length: {first_cache.seq_len}")

    report.append("\n## Cases Analyzed")
    for cache in caches:
        report.append(f"\n### {cache.case_name}")
        report.append(f"- Total key norm: {sum(s.key_norm for s in cache.layer_stats):.2f}")
        report.append(f"- Total value norm: {sum(s.value_norm for s in cache.layer_stats):.2f}")

    report.append("\n## Pairwise Comparisons")
    report.append("\n| Case A | Case B | Total Key L2 | Total Value L2 | Most Diff Layer | Most Diff Region |")
    report.append("|--------|--------|--------------|----------------|-----------------|------------------|")
    for comp in comparisons:
        short_a = comp.case_a[:15] + ".." if len(comp.case_a) > 15 else comp.case_a
        short_b = comp.case_b[:15] + ".." if len(comp.case_b) > 15 else comp.case_b
        report.append(f"| {short_a} | {short_b} | "
                     f"{comp.total_key_l2:.2f} | "
                     f"{comp.total_value_l2:.2f} | "
                     f"{comp.most_different_layer} | "
                     f"{comp.most_different_region} |")

    report.append("\n## Region-Level Analysis")
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:25]} vs {comp.case_b[:25]}")
        report.append("\n| Region | Total L2 Distance | % of Total |")
        report.append("|--------|-------------------|------------|")
        total = sum(comp.region_total_l2.values())
        for region, l2 in comp.region_total_l2.items():
            pct = l2 / total * 100 if total > 0 else 0
            report.append(f"| {region} | {l2:.2f} | {pct:.1f}% |")

    report.append("\n## Key Findings")

    # Analyze patterns
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:20]} vs {comp.case_b[:20]}")

        # Layer analysis
        report.append(f"\n**Layer Distribution:**")
        report.append(f"- Most different layer: **{comp.most_different_layer}** "
                     f"(L2: {comp.most_different_layer_l2:.2f})")

        # Check if early/late layers differ more
        early_l2 = sum(lc["total_kv_l2"] for lc in comp.layer_comparisons[:8])
        late_l2 = sum(lc["total_kv_l2"] for lc in comp.layer_comparisons[8:])
        if early_l2 > late_l2:
            report.append(f"- Early layers (0-7) differ more: {early_l2:.2f} vs {late_l2:.2f}")
        else:
            report.append(f"- Late layers (8-15) differ more: {late_l2:.2f} vs {early_l2:.2f}")

        # Region analysis
        report.append(f"\n**Region Distribution:**")
        report.append(f"- Most different region: **{comp.most_different_region}**")

        # Check if vision dominates
        camera_regions = ["head_camera", "left_wrist", "right_wrist"]
        vision_l2 = sum(comp.region_total_l2[r] for r in camera_regions)
        lang_l2 = comp.region_total_l2["language"]
        state_l2 = comp.region_total_l2["state"]
        total = vision_l2 + lang_l2 + state_l2

        report.append(f"- Vision tokens: {vision_l2 / total * 100:.1f}%")
        report.append(f"- Language tokens: {lang_l2 / total * 100:.1f}%")
        report.append(f"- State token: {state_l2 / total * 100:.1f}%")

        # Right wrist specifically
        right_wrist_pct = comp.region_total_l2["right_wrist"] / total * 100
        report.append(f"- **Right wrist camera**: {right_wrist_pct:.1f}% of total difference")

    report.append("\n## Visualizations")
    report.append("\n- `layer_comparison_*.png`: Per-layer KV distance breakdown")
    report.append("- `region_by_layer_heatmap_*.png`: [Layer x Region] heatmaps")
    report.append("- `region_totals.png`: Total difference by region")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Import config for defaults
    from investigation_config import (
        get_default_checkpoint, get_halluc_vs_clean, get_output_dir, DEFAULT_STEP, DEVICE
    )

    parser = argparse.ArgumentParser(description="Analyze KV cache content between cases")
    parser.add_argument("--checkpoint", default=get_default_checkpoint(),
                       help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dirs", nargs="+", default=get_halluc_vs_clean(),
                       help="Paths to case directories")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                       help="Inference step to analyze")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: auto-generated with timestamp)")
    parser.add_argument("--device", default=DEVICE, help="Device to use")

    args = parser.parse_args()

    # Auto-generate output dir if not specified
    if args.output_dir is None:
        output_dir = get_output_dir("kv_cache")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize extractor
    extractor = KVCacheExtractor(args.checkpoint, args.device)
    extractor.load_model()

    # Extract KV caches
    print(f"\nExtracting KV caches at step {args.step}...")
    all_caches = []
    for case_dir in args.case_dirs:
        print(f"  Processing: {case_dir}")
        cache = extractor.extract_kv_cache(case_dir, args.step)
        all_caches.append(cache)
        print(f"    Layers: {cache.num_layers}, Heads: {cache.num_heads}, SeqLen: {cache.seq_len}")

    # Compare all pairs
    print("\nComparing KV caches...")
    comparisons = []
    for i in range(len(all_caches)):
        for j in range(i + 1, len(all_caches)):
            comp = compare_kv_caches(all_caches[i], all_caches[j])
            comparisons.append(comp)
            print(f"  {comp.case_a[:25]} vs {comp.case_b[:25]}: "
                  f"total_kv_l2={comp.total_kv_l2:.2f}, "
                  f"most_diff_layer={comp.most_different_layer}, "
                  f"most_diff_region={comp.most_different_region}")

    # Save raw data
    comparison_data = [asdict(c) for c in comparisons]
    with open(output_dir / "comparisons.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    for comp in comparisons:
        visualize_layer_comparison(comp, output_dir)
        visualize_region_by_layer_heatmap(comp, output_dir)
    visualize_region_totals(comparisons, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_caches, comparisons, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
