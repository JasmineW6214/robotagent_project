#!/usr/bin/env python3
"""
Prefix Embedding Analysis Tool for SmolVLA Hallucination Investigation.

Analyzes the combined prefix embedding (vision + language + state) differences
between hallucination and normal cases.

This tool helps answer:
1. Do vision tokens [0-191] differ more than language [192-239] or state [240]?
2. Within vision, which camera region contributes most to the difference?
3. Is the difference localized to specific patch positions?

TOKEN LAYOUT (VERIFIED):
  [0-63]     Head camera patches (8x8 grid)
  [64-127]   Left wrist camera patches (8x8 grid)
  [128-191]  Right wrist camera patches (8x8 grid)
  [192-239]  Language tokens (~48)
  [240]      State token
  ─────────────────────────────────────────
  TOTAL:     241 prefix tokens

Usage:
    python prefix_embedding_analysis.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                    logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir outputs/prefix_embedding_analysis
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import torch
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch scikit-learn")
    sys.exit(1)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS - TOKEN LAYOUT
# ============================================================================

PATCHES_PER_CAMERA = 64  # 8x8 grid after connector
PATCH_GRID_SIZE = 8
NUM_CAMERAS = 3
IMAGE_SIZE = 512

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

TOTAL_IMAGE_TOKENS = 192
TOTAL_LANGUAGE_TOKENS = 48
TOTAL_PREFIX_TOKENS = 241

# Camera names as they appear in case directories
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
# Camera names as expected by the model config
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]
# Mapping from file name to model name
CAMERA_NAME_MAPPING = dict(zip(CAMERA_FILE_NAMES, CAMERA_MODEL_NAMES))

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
class RegionAnalysis:
    """Analysis for a single token region."""
    name: str
    start_idx: int
    end_idx: int
    num_tokens: int
    mean_l2_distance: float
    max_l2_distance: float
    min_l2_distance: float
    total_l2_distance: float
    cosine_similarity: float


@dataclass
class PrefixEmbeddingComparison:
    """Comparison between two cases' prefix embeddings."""
    case_a: str
    case_b: str
    step: int

    # Per-token L2 distances
    per_token_l2: List[float]  # [241] values

    # Region summaries
    region_analysis: Dict[str, RegionAnalysis]

    # Overall metrics
    total_l2_distance: float
    overall_cosine_similarity: float

    # Identification of largest difference
    most_different_region: str
    most_different_token_idx: int
    most_different_token_region: str


@dataclass
class CasePrefixEmbedding:
    """Prefix embedding for a single case."""
    case_name: str
    step: int
    prefix_embedding: np.ndarray  # [241, hidden_dim]
    hidden_dim: int


# ============================================================================
# PREFIX EMBEDDING EXTRACTION
# ============================================================================

class PrefixEmbeddingExtractor:
    """Extracts prefix embeddings from SmolVLA model."""

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

        # Stack images: [3, C, H, W] -> [1, 3, C, H, W] per camera
        images = []
        for img in case_data["images"]:
            images.append(img.unsqueeze(0))  # Add batch dim

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

        # Add images to batch using model's expected camera names
        for idx, camera_model_name in enumerate(CAMERA_MODEL_NAMES):
            key = f"observation.images.{camera_model_name}"
            batch[key] = images[idx].to(self.device)

        return batch

    def extract_prefix_embedding(self, case_dir: str, step: int) -> CasePrefixEmbedding:
        """Extract prefix embedding for a case at a specific step."""
        case_data = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(case_data)

        # Prepare images and state for embed_prefix
        # prepare_images and prepare_state are on SmolVLAPolicy (self.model)
        # embed_prefix is on VLAFlowMatching (self.model.model)
        images, img_masks = self.model.prepare_images(batch)
        state = self.model.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        with torch.no_grad():
            # Call embed_prefix to get the combined prefix embedding
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )

        # prefix_embs shape: [batch, seq_len, hidden_dim]
        # Convert to float32 before numpy (BFloat16 not supported by numpy)
        prefix_np = prefix_embs[0].float().cpu().numpy()

        return CasePrefixEmbedding(
            case_name=case_data["case_name"],
            step=step,
            prefix_embedding=prefix_np,
            hidden_dim=prefix_np.shape[-1]
        )


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def get_token_region(token_idx: int) -> str:
    """Get the region name for a token index."""
    for region_name, (start, end) in TOKEN_REGIONS.items():
        if start <= token_idx < end:
            return region_name
    return "unknown"


def compute_region_analysis(emb_a: np.ndarray, emb_b: np.ndarray,
                            region_name: str, start_idx: int, end_idx: int) -> RegionAnalysis:
    """Compute analysis for a specific token region."""
    region_a = emb_a[start_idx:end_idx]
    region_b = emb_b[start_idx:end_idx]

    # Per-token L2 distances
    token_l2 = np.linalg.norm(region_a - region_b, axis=1)

    # Cosine similarity for the whole region
    flat_a = region_a.flatten()
    flat_b = region_b.flatten()
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    cosine_sim = float(np.dot(flat_a, flat_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    return RegionAnalysis(
        name=region_name,
        start_idx=start_idx,
        end_idx=end_idx,
        num_tokens=end_idx - start_idx,
        mean_l2_distance=float(np.mean(token_l2)),
        max_l2_distance=float(np.max(token_l2)),
        min_l2_distance=float(np.min(token_l2)),
        total_l2_distance=float(np.sum(token_l2)),
        cosine_similarity=cosine_sim
    )


def compare_prefix_embeddings(case_a: CasePrefixEmbedding,
                              case_b: CasePrefixEmbedding) -> PrefixEmbeddingComparison:
    """Compare prefix embeddings between two cases."""
    emb_a = case_a.prefix_embedding
    emb_b = case_b.prefix_embedding

    # Per-token L2 distances
    per_token_l2 = np.linalg.norm(emb_a - emb_b, axis=1).tolist()

    # Region analysis
    region_analysis = {}
    for region_name, (start, end) in TOKEN_REGIONS.items():
        region_analysis[region_name] = compute_region_analysis(
            emb_a, emb_b, region_name, start, end
        )

    # Overall metrics
    total_l2 = float(np.sum(per_token_l2))
    flat_a = emb_a.flatten()
    flat_b = emb_b.flatten()
    overall_cosine = float(np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b)))

    # Find most different region
    most_diff_region = max(region_analysis.values(), key=lambda r: r.mean_l2_distance)

    # Find most different token
    most_diff_token_idx = int(np.argmax(per_token_l2))
    most_diff_token_region = get_token_region(most_diff_token_idx)

    return PrefixEmbeddingComparison(
        case_a=case_a.case_name,
        case_b=case_b.case_name,
        step=case_a.step,
        per_token_l2=per_token_l2,
        region_analysis={k: asdict(v) for k, v in region_analysis.items()},
        total_l2_distance=total_l2,
        overall_cosine_similarity=overall_cosine,
        most_different_region=most_diff_region.name,
        most_different_token_idx=most_diff_token_idx,
        most_different_token_region=most_diff_token_region
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_token_distances(comparison: PrefixEmbeddingComparison, output_dir: Path):
    """Create heatmap of per-token L2 distances."""
    per_token_l2 = np.array(comparison.per_token_l2)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # Top: Bar chart of all tokens
    ax1 = axes[0]
    colors = []
    region_colors = {
        "head_camera": "#1f77b4",  # blue
        "left_wrist": "#2ca02c",  # green
        "right_wrist": "#d62728",  # red
        "language": "#9467bd",  # purple
        "state": "#ff7f0e"  # orange
    }
    for i in range(len(per_token_l2)):
        region = get_token_region(i)
        colors.append(region_colors.get(region, "gray"))

    ax1.bar(range(len(per_token_l2)), per_token_l2, color=colors, width=1.0)
    ax1.set_xlabel("Token Index")
    ax1.set_ylabel("L2 Distance")
    ax1.set_title(f"Per-Token L2 Distance\n{comparison.case_a[:30]} vs {comparison.case_b[:30]}")

    # Add region labels
    region_labels = []
    for region_name, color in region_colors.items():
        region_labels.append(mpatches.Patch(color=color, label=region_name))
    ax1.legend(handles=region_labels, loc='upper right')

    # Add vertical lines for region boundaries
    for region_name, (start, end) in TOKEN_REGIONS.items():
        ax1.axvline(x=start, color='gray', linestyle='--', alpha=0.5)

    # Bottom: Heatmap view of camera patches
    ax2 = axes[1]

    # Create grid for camera patches
    camera_l2 = per_token_l2[:TOTAL_IMAGE_TOKENS].reshape(3, PATCH_GRID_SIZE, PATCH_GRID_SIZE)
    combined_grid = np.hstack([camera_l2[0], camera_l2[1], camera_l2[2]])

    im = ax2.imshow(combined_grid, cmap='hot', aspect='equal')
    ax2.set_title("Camera Patch L2 Distances (Head | Left Wrist | Right Wrist)")
    ax2.set_xticks([4, 12, 20])
    ax2.set_xticklabels(["Head", "Left Wrist", "Right Wrist"])
    ax2.set_yticks([])
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    safe_name = f"{comparison.case_a[:15]}_{comparison.case_b[:15]}"
    plt.savefig(output_dir / f"token_distances_{safe_name}.png", dpi=150)
    plt.close()


def visualize_region_comparison(comparisons: List[PrefixEmbeddingComparison], output_dir: Path):
    """Create comparison of region-level differences."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Mean L2 distance by region
    ax1 = axes[0]
    regions = list(TOKEN_REGIONS.keys())
    x = np.arange(len(regions))
    width = 0.25

    for idx, comp in enumerate(comparisons):
        values = [comp.region_analysis[r]["mean_l2_distance"] for r in regions]
        label = f"{comp.case_a[:12]}..vs..{comp.case_b[:12]}"
        ax1.bar(x + idx * width, values, width, label=label)

    ax1.set_xlabel("Token Region")
    ax1.set_ylabel("Mean L2 Distance")
    ax1.set_title("Mean L2 Distance by Region")
    ax1.set_xticks(x + width * (len(comparisons) - 1) / 2)
    ax1.set_xticklabels(regions, rotation=45, ha='right')
    ax1.legend(fontsize=7)

    # Right: Cosine similarity by region
    ax2 = axes[1]
    for idx, comp in enumerate(comparisons):
        values = [comp.region_analysis[r]["cosine_similarity"] for r in regions]
        label = f"{comp.case_a[:12]}..vs..{comp.case_b[:12]}"
        ax2.bar(x + idx * width, values, width, label=label)

    ax2.set_xlabel("Token Region")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Cosine Similarity by Region")
    ax2.set_xticks(x + width * (len(comparisons) - 1) / 2)
    ax2.set_xticklabels(regions, rotation=45, ha='right')
    ax2.legend(fontsize=7)
    ax2.set_ylim(0.9, 1.0)  # Focus on high similarity range

    plt.tight_layout()
    plt.savefig(output_dir / "region_comparison.png", dpi=150)
    plt.close()


def visualize_pca(all_embeddings: List[CasePrefixEmbedding], output_dir: Path):
    """PCA visualization of full prefix embeddings."""
    # Flatten embeddings
    data = []
    labels = []
    for emb in all_embeddings:
        data.append(emb.prefix_embedding.flatten())
        labels.append(emb.case_name)

    data = np.array(data)

    if len(data) < 2:
        print("Warning: Need at least 2 cases for PCA visualization")
        return

    # Apply PCA
    n_components = min(2, len(data))
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(data)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(projected[idx, 0], projected[idx, 1] if n_components > 1 else 0,
                  c=[color], label=label[:30], s=200, marker='o')
        ax.annotate(label[:20], (projected[idx, 0], projected[idx, 1] if n_components > 1 else 0),
                   fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    if n_components > 1:
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("PCA of Full Prefix Embeddings")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_prefix_embeddings.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(embeddings: List[CasePrefixEmbedding],
                   comparisons: List[PrefixEmbeddingComparison],
                   output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Prefix Embedding Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"\n**Step analyzed**: {embeddings[0].step}")
    report.append(f"\n**Prefix length**: {embeddings[0].prefix_embedding.shape[0]} tokens")
    report.append(f"\n**Hidden dimension**: {embeddings[0].hidden_dim}")

    report.append("\n## Token Layout")
    report.append("```")
    report.append("[0-63]     Head camera patches (8x8 grid)")
    report.append("[64-127]   Left wrist camera patches (8x8 grid)")
    report.append("[128-191]  Right wrist camera patches (8x8 grid)")
    report.append("[192-239]  Language tokens (~48)")
    report.append("[240]      State token")
    report.append("─────────────────────────────────────────")
    report.append("TOTAL:     241 prefix tokens")
    report.append("```")

    report.append("\n## Cases Analyzed")
    for emb in embeddings:
        report.append(f"\n### {emb.case_name}")
        report.append(f"- Embedding shape: {emb.prefix_embedding.shape}")
        report.append(f"- Mean magnitude: {np.mean(np.linalg.norm(emb.prefix_embedding, axis=1)):.4f}")

    report.append("\n## Pairwise Comparisons")
    report.append("\n| Case A | Case B | Total L2 | Cosine Sim | Most Diff Region |")
    report.append("|--------|--------|----------|------------|------------------|")
    for comp in comparisons:
        short_a = comp.case_a[:18] + ".." if len(comp.case_a) > 18 else comp.case_a
        short_b = comp.case_b[:18] + ".." if len(comp.case_b) > 18 else comp.case_b
        report.append(f"| {short_a} | {short_b} | "
                     f"{comp.total_l2_distance:.2f} | "
                     f"{comp.overall_cosine_similarity:.4f} | "
                     f"{comp.most_different_region} |")

    report.append("\n## Region-Level Analysis")
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:25]} vs {comp.case_b[:25]}")
        report.append("\n| Region | Num Tokens | Mean L2 | Max L2 | Cosine Sim |")
        report.append("|--------|------------|---------|--------|------------|")
        for region_name in TOKEN_REGIONS.keys():
            r = comp.region_analysis[region_name]
            report.append(f"| {region_name} | {r['num_tokens']} | "
                         f"{r['mean_l2_distance']:.4f} | "
                         f"{r['max_l2_distance']:.4f} | "
                         f"{r['cosine_similarity']:.4f} |")

    report.append("\n## Key Findings")

    # Analyze halluc vs normal
    halluc_comps = [c for c in comparisons if "ha_bana" in c.case_a.lower() or "ha_bana" in c.case_b.lower()]
    if halluc_comps:
        report.append("\n### Hallucination vs Normal Case Analysis")
        for comp in halluc_comps:
            report.append(f"\n**{comp.case_a[:25]}** vs **{comp.case_b[:25]}**:")

            # Find which camera has largest difference
            camera_regions = ["head_camera", "left_wrist", "right_wrist"]
            camera_l2 = {r: comp.region_analysis[r]["mean_l2_distance"] for r in camera_regions}
            max_camera = max(camera_l2.keys(), key=lambda k: camera_l2[k])

            report.append(f"- **Most different camera**: {max_camera} (mean L2: {camera_l2[max_camera]:.4f})")
            report.append(f"- Right wrist mean L2: {camera_l2['right_wrist']:.4f}")
            report.append(f"- Language tokens mean L2: {comp.region_analysis['language']['mean_l2_distance']:.4f}")
            report.append(f"- State token L2: {comp.region_analysis['state']['mean_l2_distance']:.4f}")

            # Check if vision dominates
            vision_l2 = sum(comp.region_analysis[r]["total_l2_distance"] for r in camera_regions)
            lang_l2 = comp.region_analysis["language"]["total_l2_distance"]
            state_l2 = comp.region_analysis["state"]["total_l2_distance"]

            vision_pct = vision_l2 / comp.total_l2_distance * 100
            lang_pct = lang_l2 / comp.total_l2_distance * 100
            state_pct = state_l2 / comp.total_l2_distance * 100

            report.append(f"\n**Contribution to total difference:**")
            report.append(f"- Vision tokens: {vision_pct:.1f}%")
            report.append(f"- Language tokens: {lang_pct:.1f}%")
            report.append(f"- State token: {state_pct:.1f}%")

    report.append("\n## Visualizations")
    report.append("\n- `token_distances_*.png`: Per-token L2 distance heatmaps")
    report.append("- `region_comparison.png`: Region-level comparison bar charts")
    report.append("- `pca_prefix_embeddings.png`: PCA of full prefix embeddings")

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

    parser = argparse.ArgumentParser(description="Analyze prefix embeddings between cases")
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
        output_dir = get_output_dir("prefix_embedding")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize extractor
    extractor = PrefixEmbeddingExtractor(args.checkpoint, args.device)
    extractor.load_model()

    # Extract embeddings for all cases
    print(f"\nExtracting prefix embeddings at step {args.step}...")
    all_embeddings = []
    for case_dir in args.case_dirs:
        print(f"  Processing: {case_dir}")
        embedding = extractor.extract_prefix_embedding(case_dir, args.step)
        all_embeddings.append(embedding)
        print(f"    Shape: {embedding.prefix_embedding.shape}")

    # Compare all pairs
    print("\nComparing prefix embeddings...")
    comparisons = []
    for i in range(len(all_embeddings)):
        for j in range(i + 1, len(all_embeddings)):
            comp = compare_prefix_embeddings(all_embeddings[i], all_embeddings[j])
            comparisons.append(comp)
            print(f"  {comp.case_a[:25]} vs {comp.case_b[:25]}: "
                  f"total_l2={comp.total_l2_distance:.2f}, "
                  f"most_diff={comp.most_different_region}")

    # Save raw data
    comparison_data = [asdict(c) for c in comparisons]
    with open(output_dir / "comparisons.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    for comp in comparisons:
        visualize_token_distances(comp, output_dir)
    visualize_region_comparison(comparisons, output_dir)
    visualize_pca(all_embeddings, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_embeddings, comparisons, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - comparisons.json: Raw comparison data")
    print(f"  - token_distances_*.png: Per-token visualizations")
    print(f"  - region_comparison.png: Region-level comparison")
    print(f"  - pca_prefix_embeddings.png: PCA visualization")
    print(f"  - report.md: Summary report")


if __name__ == "__main__":
    main()
