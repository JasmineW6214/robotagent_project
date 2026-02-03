#!/usr/bin/env python3
"""
Vision Feature Comparison Tool for SmolVLA Hallucination Investigation.

Compares SigLIP encoder visual features between hallucination and normal cases
at a specific inference step (e.g., step 200 post-task completion).

This tool helps answer:
1. Does the head camera clearly show "no target object" in both cases?
2. Does banana presence change right wrist camera features significantly?
3. Which camera has the largest feature difference between cases?

Usage:
    python vision_feature_comparison.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
                    logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir outputs/vision_feature_comparison
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch scikit-learn")
    sys.exit(1)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# CONSTANTS
# ============================================================================

CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]
PATCHES_PER_CAMERA = 64  # 8x8 grid after connector projection
PATCH_GRID_SIZE = 8
IMAGE_SIZE = 512  # SmolVLA input image size


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CameraFeatures:
    """Features extracted from a single camera."""
    name: str
    raw_features: Optional[np.ndarray] = None  # [64, hidden_dim]
    post_connector_features: Optional[np.ndarray] = None  # [64, hidden_dim]
    feature_mean: Optional[float] = None
    feature_std: Optional[float] = None
    feature_norm: Optional[float] = None


@dataclass
class CaseFeatures:
    """Features extracted from a single case."""
    case_name: str
    step: int
    cameras: dict  # {camera_name: CameraFeatures}
    combined_features: Optional[np.ndarray] = None  # [192, hidden_dim] all cameras


@dataclass
class FeatureComparison:
    """Comparison results between cases."""
    case_a: str
    case_b: str
    per_camera_cosine_similarity: dict  # {camera_name: float}
    per_camera_l2_distance: dict  # {camera_name: float}
    overall_cosine_similarity: float
    overall_l2_distance: float
    most_different_camera: str
    largest_difference: float


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class VisionFeatureExtractor:
    """Extracts visual features from SmolVLA's SigLIP encoder."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    def load_model(self):
        """Load the SmolVLA model."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print(f"Loading model from {self.checkpoint_path}...")
        self.model = SmolVLAPolicy.from_pretrained(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for SmolVLA."""
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 512x512
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize to [0, 1] then to model expected range
        img = img.astype(np.float32) / 255.0

        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        # Add batch dimension [1, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor.to(self.device)

    def extract_raw_vision_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract raw SigLIP encoder features before connector."""
        with torch.no_grad():
            # Get vision encoder hidden states
            vlm_with_expert = self.model.model.vlm_with_expert
            vision_model = vlm_with_expert.get_vlm_model().vision_model

            # Forward through vision encoder
            hidden_states = vision_model(
                pixel_values=image_tensor.to(dtype=vision_model.dtype)
            ).last_hidden_state

            # Convert to float32 before numpy (BFloat16 not supported by numpy)
            return hidden_states.float().cpu().numpy()

    def extract_post_connector_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract features after multi_modal_projector (connector)."""
        with torch.no_grad():
            vlm_with_expert = self.model.model.vlm_with_expert

            # Use the embed_image method which includes connector
            features = vlm_with_expert.embed_image(image_tensor)

            # Convert to float32 before numpy (BFloat16 not supported by numpy)
            return features.float().cpu().numpy()

    def extract_case_features(self, case_dir: str, step: int) -> CaseFeatures:
        """Extract features for all cameras at a given step."""
        case_path = Path(case_dir)
        case_name = case_path.name

        cameras = {}
        all_features = []

        for camera_name in CAMERA_NAMES:
            # Find image file
            image_path = case_path / "images" / f"step_{step:04d}_{camera_name}.jpg"
            if not image_path.exists():
                # Try alternative naming
                alt_path = case_path / "images" / f"step_{step:04d}_{camera_name}.png"
                if alt_path.exists():
                    image_path = alt_path
                else:
                    print(f"Warning: Image not found: {image_path}")
                    continue

            # Preprocess and extract features
            img_tensor = self.preprocess_image(str(image_path))
            post_connector = self.extract_post_connector_features(img_tensor)

            # post_connector shape: [1, 64, hidden_dim]
            features = post_connector[0]  # [64, hidden_dim]

            camera_features = CameraFeatures(
                name=camera_name,
                post_connector_features=features,
                feature_mean=float(np.mean(features)),
                feature_std=float(np.std(features)),
                feature_norm=float(np.linalg.norm(features))
            )
            cameras[camera_name] = camera_features
            all_features.append(features)

        # Combine all camera features
        combined = np.concatenate(all_features, axis=0) if all_features else None

        return CaseFeatures(
            case_name=case_name,
            step=step,
            cameras=cameras,
            combined_features=combined
        )


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance between two feature arrays."""
    return float(np.linalg.norm(a - b))


def compare_cases(case_a: CaseFeatures, case_b: CaseFeatures) -> FeatureComparison:
    """Compare features between two cases."""
    per_camera_cosine = {}
    per_camera_l2 = {}

    for camera_name in CAMERA_NAMES:
        if camera_name not in case_a.cameras or camera_name not in case_b.cameras:
            continue

        feat_a = case_a.cameras[camera_name].post_connector_features
        feat_b = case_b.cameras[camera_name].post_connector_features

        per_camera_cosine[camera_name] = cosine_similarity(feat_a, feat_b)
        per_camera_l2[camera_name] = l2_distance(feat_a, feat_b)

    # Find most different camera (lowest cosine similarity)
    most_different = min(per_camera_cosine.keys(), key=lambda k: per_camera_cosine[k])
    largest_diff = per_camera_l2[most_different]

    # Overall comparison
    overall_cosine = cosine_similarity(case_a.combined_features, case_b.combined_features)
    overall_l2 = l2_distance(case_a.combined_features, case_b.combined_features)

    return FeatureComparison(
        case_a=case_a.case_name,
        case_b=case_b.case_name,
        per_camera_cosine_similarity=per_camera_cosine,
        per_camera_l2_distance=per_camera_l2,
        overall_cosine_similarity=overall_cosine,
        overall_l2_distance=overall_l2,
        most_different_camera=most_different,
        largest_difference=largest_diff
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_pca(all_case_features: list, output_dir: Path):
    """Create PCA visualization of patch embeddings."""
    # Collect all patch features with labels
    all_patches = []
    labels = []
    camera_labels = []

    for case_feat in all_case_features:
        for camera_name, cam_feat in case_feat.cameras.items():
            patches = cam_feat.post_connector_features
            for patch_idx in range(patches.shape[0]):
                all_patches.append(patches[patch_idx])
                labels.append(case_feat.case_name)
                camera_labels.append(camera_name)

    all_patches = np.array(all_patches)

    # Apply PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_patches)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by case
    case_names = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(case_names)))
    color_map = {name: colors[i] for i, name in enumerate(case_names)}

    ax1 = axes[0]
    for case_name in case_names:
        mask = [l == case_name for l in labels]
        ax1.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[color_map[case_name]],
            label=case_name[:30] + "..." if len(case_name) > 30 else case_name,
            alpha=0.6,
            s=20
        )
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax1.set_title("PCA of Patch Embeddings (by Case)")
    ax1.legend(fontsize=8)

    # Plot 2: Color by camera
    camera_colors = {"head": "blue", "left_wrist": "green", "right_wrist": "red"}
    ax2 = axes[1]
    for camera_name in CAMERA_NAMES:
        mask = [l == camera_name for l in camera_labels]
        ax2.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=camera_colors[camera_name],
            label=camera_name,
            alpha=0.6,
            s=20
        )
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax2.set_title("PCA of Patch Embeddings (by Camera)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "pca_visualization.png", dpi=150)
    plt.close()

    # Save PCA info
    pca_info = {
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "total_variance_explained": float(sum(pca.explained_variance_ratio_))
    }
    with open(output_dir / "pca_info.json", "w") as f:
        json.dump(pca_info, f, indent=2)


def visualize_spatial_difference(case_a: CaseFeatures, case_b: CaseFeatures,
                                  output_dir: Path):
    """Create spatial heatmaps showing where features differ per camera."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, camera_name in enumerate(CAMERA_NAMES):
        ax = axes[idx]

        if camera_name not in case_a.cameras or camera_name not in case_b.cameras:
            ax.set_title(f"{camera_name}\n(missing)")
            continue

        feat_a = case_a.cameras[camera_name].post_connector_features
        feat_b = case_b.cameras[camera_name].post_connector_features

        # Compute per-patch L2 distance
        patch_diffs = np.linalg.norm(feat_a - feat_b, axis=1)  # [64]

        # Reshape to 8x8 grid
        diff_grid = patch_diffs.reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)

        # Plot heatmap
        im = ax.imshow(diff_grid, cmap='hot', aspect='equal')
        ax.set_title(f"{camera_name}\nL2 diff: {np.mean(patch_diffs):.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Spatial Feature Difference\n{case_a.case_name[:25]}... vs {case_b.case_name[:25]}...")
    plt.tight_layout()

    # Create safe filename
    safe_name_a = case_a.case_name.replace("/", "_")[:20]
    safe_name_b = case_b.case_name.replace("/", "_")[:20]
    plt.savefig(output_dir / f"spatial_diff_{safe_name_a}_vs_{safe_name_b}.png", dpi=150)
    plt.close()


def visualize_comparison_matrix(comparisons: list, output_dir: Path):
    """Create comparison matrix visualization."""
    # Get unique case names
    case_names = list(set([c.case_a for c in comparisons] + [c.case_b for c in comparisons]))
    n_cases = len(case_names)

    # Create matrices
    cosine_matrix = np.eye(n_cases)
    l2_matrix = np.zeros((n_cases, n_cases))

    for comp in comparisons:
        i = case_names.index(comp.case_a)
        j = case_names.index(comp.case_b)
        cosine_matrix[i, j] = comp.overall_cosine_similarity
        cosine_matrix[j, i] = comp.overall_cosine_similarity
        l2_matrix[i, j] = comp.overall_l2_distance
        l2_matrix[j, i] = comp.overall_l2_distance

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Cosine similarity matrix
    ax1 = axes[0]
    im1 = ax1.imshow(cosine_matrix, cmap='RdYlGn', vmin=0.8, vmax=1.0)
    ax1.set_xticks(range(n_cases))
    ax1.set_yticks(range(n_cases))
    short_names = [n[:15] + "..." if len(n) > 15 else n for n in case_names]
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(short_names, fontsize=8)
    ax1.set_title("Cosine Similarity")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Add values
    for i in range(n_cases):
        for j in range(n_cases):
            ax1.text(j, i, f"{cosine_matrix[i, j]:.3f}",
                    ha='center', va='center', fontsize=7)

    # L2 distance matrix
    ax2 = axes[1]
    im2 = ax2.imshow(l2_matrix, cmap='Blues')
    ax2.set_xticks(range(n_cases))
    ax2.set_yticks(range(n_cases))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(short_names, fontsize=8)
    ax2.set_title("L2 Distance")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Add values
    for i in range(n_cases):
        for j in range(n_cases):
            ax2.text(j, i, f"{l2_matrix[i, j]:.1f}",
                    ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_matrix.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(all_case_features: list, comparisons: list, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Vision Feature Comparison Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"\n**Step analyzed**: {all_case_features[0].step}")
    report.append(f"\n**Number of cases**: {len(all_case_features)}")

    report.append("\n## Cases Analyzed")
    for case in all_case_features:
        report.append(f"\n### {case.case_name}")
        report.append(f"- Combined feature shape: {case.combined_features.shape}")
        for cam_name, cam_feat in case.cameras.items():
            report.append(f"- {cam_name}: mean={cam_feat.feature_mean:.4f}, "
                         f"std={cam_feat.feature_std:.4f}, norm={cam_feat.feature_norm:.2f}")

    report.append("\n## Pairwise Comparisons")
    report.append("\n| Case A | Case B | Cosine Sim | L2 Distance | Most Different Camera |")
    report.append("|--------|--------|------------|-------------|----------------------|")
    for comp in comparisons:
        short_a = comp.case_a[:20] + "..." if len(comp.case_a) > 20 else comp.case_a
        short_b = comp.case_b[:20] + "..." if len(comp.case_b) > 20 else comp.case_b
        report.append(f"| {short_a} | {short_b} | "
                     f"{comp.overall_cosine_similarity:.4f} | "
                     f"{comp.overall_l2_distance:.2f} | "
                     f"{comp.most_different_camera} |")

    report.append("\n## Per-Camera Comparison Details")
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:30]} vs {comp.case_b[:30]}")
        report.append("\n| Camera | Cosine Similarity | L2 Distance |")
        report.append("|--------|------------------|-------------|")
        for cam in CAMERA_NAMES:
            if cam in comp.per_camera_cosine_similarity:
                report.append(f"| {cam} | "
                             f"{comp.per_camera_cosine_similarity[cam]:.4f} | "
                             f"{comp.per_camera_l2_distance[cam]:.2f} |")

    report.append("\n## Key Findings")
    # Find hallucination case (contains "ha_" in name)
    halluc_cases = [c for c in all_case_features if "ha_bana" in c.case_name.lower() or "halluc" in c.case_name.lower()]
    normal_cases = [c for c in all_case_features if c not in halluc_cases]

    if halluc_cases and normal_cases:
        report.append("\n### Hallucination vs Normal Case Analysis")
        for halluc in halluc_cases:
            for normal in normal_cases:
                # Find comparison
                for comp in comparisons:
                    if (comp.case_a == halluc.case_name and comp.case_b == normal.case_name) or \
                       (comp.case_b == halluc.case_name and comp.case_a == normal.case_name):
                        report.append(f"\n**{halluc.case_name[:30]}** vs **{normal.case_name[:30]}**:")
                        report.append(f"- Overall cosine similarity: {comp.overall_cosine_similarity:.4f}")
                        report.append(f"- Most different camera: **{comp.most_different_camera}**")
                        report.append(f"- Right wrist cosine sim: "
                                     f"{comp.per_camera_cosine_similarity.get('right_wrist', 'N/A'):.4f}")

    report.append("\n## Visualizations")
    report.append("\n- `pca_visualization.png`: PCA projection of patch embeddings")
    report.append("- `comparison_matrix.png`: Similarity/distance matrix")
    report.append("- `spatial_diff_*.png`: Per-camera spatial difference heatmaps")

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

    parser = argparse.ArgumentParser(description="Compare vision features between cases")
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
        output_dir = get_output_dir("vision_features")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize extractor
    extractor = VisionFeatureExtractor(args.checkpoint, args.device)
    extractor.load_model()

    # Extract features for all cases
    print(f"\nExtracting features at step {args.step}...")
    all_case_features = []
    for case_dir in args.case_dirs:
        print(f"  Processing: {case_dir}")
        features = extractor.extract_case_features(case_dir, args.step)
        all_case_features.append(features)
        print(f"    Combined shape: {features.combined_features.shape}")

    # Compare all pairs
    print("\nComparing cases...")
    comparisons = []
    for i in range(len(all_case_features)):
        for j in range(i + 1, len(all_case_features)):
            comp = compare_cases(all_case_features[i], all_case_features[j])
            comparisons.append(comp)
            print(f"  {comp.case_a[:25]} vs {comp.case_b[:25]}: "
                  f"cosine={comp.overall_cosine_similarity:.4f}, "
                  f"most_diff={comp.most_different_camera}")

    # Save raw comparison data
    comparison_data = [asdict(c) for c in comparisons]
    with open(output_dir / "comparisons.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_pca(all_case_features, output_dir)
    visualize_comparison_matrix(comparisons, output_dir)

    # Spatial difference for first comparison (halluc vs normal)
    if len(all_case_features) >= 2:
        visualize_spatial_difference(all_case_features[0], all_case_features[-1], output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_case_features, comparisons, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - comparisons.json: Raw comparison data")
    print(f"  - pca_visualization.png: PCA projection")
    print(f"  - comparison_matrix.png: Similarity matrix")
    print(f"  - report.md: Summary report")


if __name__ == "__main__":
    main()
