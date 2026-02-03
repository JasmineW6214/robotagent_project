#!/usr/bin/env python3
"""
Counterfactual Masking for SmolVLA Hallucination Investigation.

Tests the causal effect of visual distractor objects by:
1. Loading images from hallucination cases
2. Masking out the distractor region in the specified camera
3. Running inference on both original and masked images
4. Comparing action outputs

If masking the distractor eliminates the hallucination-causing actions,
this proves a causal visual link.

IMPORTANT: Based on per-camera cross-attention analysis, the banana (distractor)
is visible in the RIGHT WRIST camera and triggers hallucination there.
Default camera to mask is now "right_wrist".

Usage:
    python counterfactual_masking.py \
        --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --distractor-bbox 380,260,450,320 \
        --camera right_wrist \
        --output-dir logs/analysis/counterfactual
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MaskingResult:
    """Result of counterfactual masking experiment for one step."""
    inference_step: int
    original_action: List[float]  # First action from chunk
    masked_action: List[float]
    action_delta: List[float]  # original - masked
    action_delta_norm: float  # L2 norm of delta
    original_action_norm: float
    masked_action_norm: float


@dataclass
class CounterfactualAnalysis:
    """Complete analysis of counterfactual masking experiment."""
    case_dir: str
    distractor_bbox: Tuple[int, int, int, int]
    mask_type: str  # "mean", "blur", "inpaint"
    camera: str  # "head", "left_wrist", "right_wrist"
    timestamp: str
    results: List[MaskingResult]
    avg_action_delta_norm: float
    max_action_delta_norm: float


# ============================================================================
# MASKING FUNCTIONS
# ============================================================================

def apply_mask_mean(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Mask region by filling with image mean color.
    """
    x1, y1, x2, y2 = bbox
    masked = image.copy()
    mean_color = image.mean(axis=(0, 1)).astype(image.dtype)
    masked[y1:y2, x1:x2] = mean_color
    return masked


def apply_mask_blur(image: np.ndarray, bbox: Tuple[int, int, int, int], ksize: int = 51) -> np.ndarray:
    """
    Mask region by applying heavy Gaussian blur.
    """
    x1, y1, x2, y2 = bbox
    masked = image.copy()
    region = masked[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
    masked[y1:y2, x1:x2] = blurred
    return masked


def apply_mask_inpaint(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Mask region using OpenCV inpainting.
    """
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    # Use Navier-Stokes based inpainting
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    return inpainted


def apply_mask_noise(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Mask region by filling with random noise (same distribution as image).
    """
    x1, y1, x2, y2 = bbox
    masked = image.copy()
    h, w = y2 - y1, x2 - x1
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    masked[y1:y2, x1:x2] = noise
    return masked


MASK_FUNCTIONS = {
    "mean": apply_mask_mean,
    "blur": apply_mask_blur,
    "inpaint": apply_mask_inpaint,
    "noise": apply_mask_noise,
}


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def run_inference(
    policy,
    preprocessor,
    head_img: np.ndarray,
    left_wrist_img: np.ndarray,
    right_wrist_img: np.ndarray,
    task: str,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run single inference and return the action chunk.
    """
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

    policy.reset()

    with torch.no_grad():
        action = policy.select_action(preprocessed_obs)

    return action.cpu().numpy()


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_counterfactual_analysis(
    case_dir: Path,
    output_dir: Path,
    distractor_bbox: Tuple[int, int, int, int],
    camera: str = "right_wrist",
    mask_type: str = "mean",
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> CounterfactualAnalysis:
    """
    Run counterfactual masking experiment.

    Args:
        case_dir: Directory with case data (images, metadata)
        output_dir: Output directory for analysis results
        distractor_bbox: Bounding box of distractor as (x1, y1, x2, y2)
        camera: Which camera to mask - "head", "left_wrist", or "right_wrist"
        mask_type: Type of masking - "mean", "blur", "inpaint", "noise"
        checkpoint_path: Override checkpoint path
        device: Device for inference
    """
    print(f"Running counterfactual analysis for: {case_dir}")
    print(f"Distractor bbox: {distractor_bbox}")
    print(f"Camera to mask: {camera}")
    print(f"Mask type: {mask_type}")

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
    image_files = sorted(images_dir.glob("step_*_head.jpg"))
    available_steps = sorted(set(int(p.stem.split('_')[1]) for p in image_files))

    # Focus on hallucination-critical steps
    priority_steps = [0, 100, 200, 250, 300, 350]
    steps_to_analyze = [s for s in priority_steps if s in available_steps]

    if not steps_to_analyze:
        steps_to_analyze = available_steps[:6]

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

    # Get mask function
    mask_fn = MASK_FUNCTIONS.get(mask_type, apply_mask_mean)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masked_images"
    masks_dir.mkdir(exist_ok=True)

    results = []

    for step_num in steps_to_analyze:
        print(f"\n  Processing step {step_num}...")

        # Load images
        head_path = images_dir / f"step_{step_num:04d}_head.jpg"
        left_wrist_path = images_dir / f"step_{step_num:04d}_left_wrist.jpg"
        right_wrist_path = images_dir / f"step_{step_num:04d}_right_wrist.jpg"

        if not all(p.exists() for p in [head_path, left_wrist_path, right_wrist_path]):
            print(f"    Missing camera images for step {step_num}")
            continue

        head_img = cv2.cvtColor(cv2.imread(str(head_path)), cv2.COLOR_BGR2RGB)
        left_wrist_img = cv2.cvtColor(cv2.imread(str(left_wrist_path)), cv2.COLOR_BGR2RGB)
        right_wrist_img = cv2.cvtColor(cv2.imread(str(right_wrist_path)), cv2.COLOR_BGR2RGB)

        # Create masked version of the specified camera
        if camera == "head":
            masked_head_img = mask_fn(head_img, distractor_bbox)
            masked_left_wrist_img = left_wrist_img
            masked_right_wrist_img = right_wrist_img
            masked_img_for_save = masked_head_img
        elif camera == "left_wrist":
            masked_head_img = head_img
            masked_left_wrist_img = mask_fn(left_wrist_img, distractor_bbox)
            masked_right_wrist_img = right_wrist_img
            masked_img_for_save = masked_left_wrist_img
        elif camera == "right_wrist":
            masked_head_img = head_img
            masked_left_wrist_img = left_wrist_img
            masked_right_wrist_img = mask_fn(right_wrist_img, distractor_bbox)
            masked_img_for_save = masked_right_wrist_img
        else:
            raise ValueError(f"Unknown camera: {camera}. Must be 'head', 'left_wrist', or 'right_wrist'")

        # Save masked image for visualization
        cv2.imwrite(
            str(masks_dir / f"step_{step_num:04d}_{camera}_masked.jpg"),
            cv2.cvtColor(masked_img_for_save, cv2.COLOR_RGB2BGR)
        )

        # Run inference with original images
        print(f"    Running original inference...")
        original_action = run_inference(
            policy, preprocessor, head_img, left_wrist_img, right_wrist_img, task, device
        )

        # Run inference with masked image
        print(f"    Running masked inference (masking {camera})...")
        masked_action = run_inference(
            policy, preprocessor, masked_head_img, masked_left_wrist_img, masked_right_wrist_img, task, device
        )

        # Compute delta
        action_delta = original_action - masked_action
        action_delta_norm = float(np.linalg.norm(action_delta))
        original_norm = float(np.linalg.norm(original_action))
        masked_norm = float(np.linalg.norm(masked_action))

        print(f"    Original action norm: {original_norm:.4f}")
        print(f"    Masked action norm: {masked_norm:.4f}")
        print(f"    Action delta norm: {action_delta_norm:.4f}")

        results.append(MaskingResult(
            inference_step=step_num,
            original_action=original_action.tolist(),
            masked_action=masked_action.tolist(),
            action_delta=action_delta.tolist(),
            action_delta_norm=action_delta_norm,
            original_action_norm=original_norm,
            masked_action_norm=masked_norm,
        ))

    # Compute aggregates
    avg_delta = np.mean([r.action_delta_norm for r in results]) if results else 0
    max_delta = max([r.action_delta_norm for r in results]) if results else 0

    analysis = CounterfactualAnalysis(
        case_dir=str(case_dir),
        distractor_bbox=distractor_bbox,
        mask_type=mask_type,
        camera=camera,
        timestamp=datetime.now().isoformat(),
        results=results,
        avg_action_delta_norm=float(avg_delta),
        max_action_delta_norm=float(max_delta),
    )

    return analysis


def plot_counterfactual_results(
    analysis: CounterfactualAnalysis,
    output_path: Optional[Path] = None,
):
    """Plot counterfactual analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = [r.inference_step for r in analysis.results]

    # Plot 1: Action norms
    ax1 = axes[0, 0]
    ax1.plot(steps, [r.original_action_norm for r in analysis.results],
             'r-o', label='Original', linewidth=2, markersize=6)
    ax1.plot(steps, [r.masked_action_norm for r in analysis.results],
             'b-o', label='Masked', linewidth=2, markersize=6)
    ax1.set_xlabel('Inference Step')
    ax1.set_ylabel('Action Norm')
    ax1.set_title('Action Magnitude: Original vs Masked')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Action delta norm
    ax2 = axes[0, 1]
    ax2.bar(steps, [r.action_delta_norm for r in analysis.results], color='purple', alpha=0.7)
    ax2.axhline(y=analysis.avg_action_delta_norm, color='r', linestyle='--', label=f'Avg: {analysis.avg_action_delta_norm:.3f}')
    ax2.set_xlabel('Inference Step')
    ax2.set_ylabel('Action Delta Norm')
    ax2.set_title('Effect of Masking on Actions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-joint comparison for first step (first action from chunk)
    ax3 = axes[1, 0]
    if analysis.results:
        r = analysis.results[0]  # First step
        # original_action is a flat array of action chunk (50 actions * 12 joints = 600)
        # or just the first action (12 joints)
        original_arr = np.array(r.original_action).flatten()
        masked_arr = np.array(r.masked_action).flatten()
        # Take first 12 values (first action) if action chunk is large
        if len(original_arr) > 12:
            original_first = original_arr[:12]
            masked_first = masked_arr[:12]
        else:
            original_first = original_arr
            masked_first = masked_arr
        joints = list(range(len(original_first)))
        x = np.arange(len(joints))
        width = 0.35
        ax3.bar(x - width/2, original_first, width, label='Original', color='red', alpha=0.7)
        ax3.bar(x + width/2, masked_first, width, label='Masked', color='blue', alpha=0.7)
        ax3.set_xlabel('Joint Index')
        ax3.set_ylabel('Action Value')
        ax3.set_title(f'First Action at Step {r.inference_step}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Action delta by joint for all steps (first action only)
    ax4 = axes[1, 1]
    if analysis.results:
        # Take first 12 values (first action) from each delta
        deltas = []
        for r in analysis.results:
            delta_arr = np.array(r.action_delta).flatten()
            deltas.append(delta_arr[:12] if len(delta_arr) > 12 else delta_arr)
        deltas = np.array(deltas)  # [steps, 12]
        if deltas.size > 0:
            vmax = np.abs(deltas).max()
            im = ax4.imshow(deltas.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax4.set_xlabel('Inference Step')
            ax4.set_ylabel('Joint Index')
            ax4.set_title('Action Delta Heatmap (Original - Masked)')
            ax4.set_xticks(range(len(steps)))
            ax4.set_xticklabels(steps)
            plt.colorbar(im, ax=ax4)

    plt.suptitle(f'Counterfactual Masking Analysis (camera: {analysis.camera}, mask: {analysis.mask_type})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")

    plt.close()


def visualize_masking(
    original_path: Path,
    masked_path: Path,
    bbox: Tuple[int, int, int, int],
    camera: str,
    output_path: Path,
):
    """Visualize original vs masked image."""
    original = cv2.imread(str(original_path))
    masked = cv2.imread(str(masked_path))

    if original is None or masked is None:
        print(f"Warning: Could not load images for visualization")
        return

    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original with bbox
    axes[0].imshow(original_rgb)
    x1, y1, x2, y2 = bbox
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f'{camera} camera: Original (with distractor)')
    axes[0].axis('off')

    # Masked
    axes[1].imshow(masked_rgb)
    axes[1].set_title(f'{camera} camera: Masked (distractor removed)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Counterfactual masking analysis for SmolVLA")
    parser.add_argument("--case-dir", type=str, required=True,
                       help="Directory with case data (images, metadata)")
    parser.add_argument("--distractor-bbox", type=str, required=True,
                       help="Distractor bounding box as x1,y1,x2,y2")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for analysis results")
    parser.add_argument("--camera", type=str, default="right_wrist",
                       choices=["head", "left_wrist", "right_wrist"],
                       help="Which camera to mask (default: right_wrist)")
    parser.add_argument("--mask-type", type=str, default="mean",
                       choices=["mean", "blur", "inpaint", "noise"],
                       help="Type of masking to apply")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Override checkpoint path")

    args = parser.parse_args()

    distractor_bbox = tuple(map(int, args.distractor_bbox.split(',')))

    print("=" * 60)
    print("SmolVLA Counterfactual Masking Analysis")
    print("=" * 60)

    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir)

    # Run analysis
    analysis = run_counterfactual_analysis(
        case_dir=case_dir,
        output_dir=output_dir,
        distractor_bbox=distractor_bbox,
        camera=args.camera,
        mask_type=args.mask_type,
        checkpoint_path=args.checkpoint,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nCamera masked: {analysis.camera}")
    print(f"Mask type: {analysis.mask_type}")
    print(f"Average action delta norm: {analysis.avg_action_delta_norm:.4f}")
    print(f"Max action delta norm: {analysis.max_action_delta_norm:.4f}")

    print("\nPer-step results:")
    for r in analysis.results:
        print(f"  Step {r.inference_step}: original={r.original_action_norm:.3f}, "
              f"masked={r.masked_action_norm:.3f}, delta={r.action_delta_norm:.3f}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    print("-" * 60)
    if analysis.avg_action_delta_norm > 0.1:
        print("  ✓ SIGNIFICANT EFFECT: Masking the distractor changes actions substantially.")
        print("  → This supports the hypothesis that the distractor causes hallucination.")
        if analysis.results:
            # Check if masked actions have lower norms (less movement)
            avg_orig = np.mean([r.original_action_norm for r in analysis.results])
            avg_masked = np.mean([r.masked_action_norm for r in analysis.results])
            if avg_masked < avg_orig:
                print(f"  → Masked actions are smaller ({avg_masked:.3f} vs {avg_orig:.3f}), suggesting less reaching behavior.")
            else:
                print(f"  → Masked actions are similar/larger ({avg_masked:.3f} vs {avg_orig:.3f}), check if reaching direction changed.")
    else:
        print("  ✗ MINIMAL EFFECT: Masking the distractor has little impact on actions.")
        print("  → The distractor may not be the primary cause, or the bbox is incorrect.")

    # Generate visualizations
    print("\nGenerating visualizations...")

    plot_counterfactual_results(
        analysis,
        output_path=output_dir / "counterfactual_analysis.png"
    )

    # Visualize masking for first step
    if analysis.results:
        first_step = analysis.results[0].inference_step
        camera = analysis.camera
        visualize_masking(
            original_path=case_dir / "images" / f"step_{first_step:04d}_{camera}.jpg",
            masked_path=output_dir / "masked_images" / f"step_{first_step:04d}_{camera}_masked.jpg",
            bbox=distractor_bbox,
            camera=camera,
            output_path=output_dir / "masking_comparison.png"
        )

    # Save JSON results
    results_dict = {
        "case_dir": analysis.case_dir,
        "distractor_bbox": analysis.distractor_bbox,
        "camera": analysis.camera,
        "mask_type": analysis.mask_type,
        "timestamp": analysis.timestamp,
        "summary": {
            "avg_action_delta_norm": analysis.avg_action_delta_norm,
            "max_action_delta_norm": analysis.max_action_delta_norm,
        },
        "results": [asdict(r) for r in analysis.results],
    }

    with open(output_dir / "counterfactual_analysis.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nAnalysis saved to: {output_dir}")
    print(f"  - counterfactual_analysis.png")
    print(f"  - masking_comparison.png")
    print(f"  - counterfactual_analysis.json")
    print(f"  - masked_images/")


if __name__ == "__main__":
    main()
