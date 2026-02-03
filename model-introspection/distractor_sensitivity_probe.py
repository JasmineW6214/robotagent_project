#!/usr/bin/env python3
"""
Distractor Sensitivity Probe for SmolVLA Hallucination Investigation.

Tests if the model uses distractor objects (like the banana) as spurious landmarks
by masking/inpainting the distractor and measuring trajectory changes.

Based on BYOVLA (Bring Your Own VLA) methodology from Princeton AI.

Method:
1. Run normal inference with distractor present
2. Inpaint/mask distractor from image (using simple masking or inpainting)
3. Run inference again
4. Compare trajectories - if large change, distractor causally affects behavior

Key Questions to Answer:
- If we inpaint out the banana, does hallucination disappear?
- How much does trajectory shift when distractor is removed?
- Which camera is most sensitive to distractor removal?

Usage:
    python distractor_sensitivity_probe.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model \
        --case-dir logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --step 200 \
        --distractor-mask banana_mask.png \
        --output-dir outputs/distractor_probe
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
    import torch.nn.functional as F
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
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]

# Default distractor region (banana location in right wrist camera)
# These are approximate pixel coordinates [y_min, y_max, x_min, x_max] in 512x512
DEFAULT_DISTRACTOR_REGIONS = {
    "right_wrist": [200, 350, 150, 350],  # Approximate banana location
    "head": None,  # No distractor in head camera typically
    "left_wrist": None,  # No distractor in left wrist typically
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MaskingResult:
    """Result of masking a distractor in a single camera."""
    camera_name: str
    masked_region: List[int]  # [y_min, y_max, x_min, x_max]
    masked_pixels: int
    mask_method: str  # "blur", "mean", "inpaint"


@dataclass
class SensitivityResult:
    """Result of a single sensitivity probe."""
    camera_name: str
    mask_method: str

    # Trajectory comparison
    trajectory_l2_distance: float
    trajectory_cosine_similarity: float

    # Action statistics
    normal_action_delta: float
    masked_action_delta: float
    action_delta_change: float
    action_delta_change_pct: float

    # Shape analysis
    normal_trajectory_shape: str
    masked_trajectory_shape: str
    shape_changed: bool

    # Per-joint analysis (which joints changed most)
    max_joint_change_idx: int
    max_joint_change_value: float


@dataclass
class DistractorProbeAnalysis:
    """Full distractor sensitivity analysis."""
    case_name: str
    step: int

    # What was masked
    masked_cameras: List[str]
    mask_method: str

    # Baseline behavior
    normal_action_delta: float
    normal_trajectory_shape: str
    normal_trajectory: List[List[float]]

    # Per-camera sensitivity
    per_camera_results: Dict[str, SensitivityResult]

    # All-cameras masked result
    all_masked_result: Optional[SensitivityResult]

    # Summary
    most_sensitive_camera: str
    sensitivity_score: float  # Overall sensitivity to distractor
    hallucination_caused_by_distractor: bool  # Conclusion


# ============================================================================
# IMAGE MASKING
# ============================================================================

def apply_blur_mask(image: np.ndarray, region: List[int], blur_size: int = 31) -> np.ndarray:
    """Apply gaussian blur to mask a region."""
    y_min, y_max, x_min, x_max = region
    masked = image.copy()
    roi = masked[y_min:y_max, x_min:x_max]
    blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
    masked[y_min:y_max, x_min:x_max] = blurred
    return masked


def apply_mean_mask(image: np.ndarray, region: List[int]) -> np.ndarray:
    """Replace region with mean color."""
    y_min, y_max, x_min, x_max = region
    masked = image.copy()
    mean_color = image.mean(axis=(0, 1))
    masked[y_min:y_max, x_min:x_max] = mean_color
    return masked


def apply_inpaint_mask(image: np.ndarray, region: List[int]) -> np.ndarray:
    """Use OpenCV inpainting to remove the region."""
    y_min, y_max, x_min, x_max = region
    masked = image.copy()

    # Create binary mask
    inpaint_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    inpaint_mask[y_min:y_max, x_min:x_max] = 255

    # Convert to uint8 for inpainting
    img_uint8 = (image * 255).astype(np.uint8)

    # Inpaint
    inpainted = cv2.inpaint(img_uint8, inpaint_mask, 3, cv2.INPAINT_TELEA)

    # Convert back to float
    masked = inpainted.astype(np.float32) / 255.0

    return masked


def detect_distractor_region(image: np.ndarray, method: str = "yellow") -> Optional[List[int]]:
    """Automatically detect distractor region (e.g., yellow banana).

    Args:
        image: RGB image as float32 [0, 1]
        method: Detection method ("yellow" for banana detection)

    Returns:
        Region as [y_min, y_max, x_min, x_max] or None
    """
    if method == "yellow":
        # Convert to HSV for yellow detection
        img_uint8 = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

        # Yellow range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest)
                # Add padding
                padding = 20
                y_min = max(0, y - padding)
                y_max = min(image.shape[0], y + h + padding)
                x_min = max(0, x - padding)
                x_max = min(image.shape[1], x + w + padding)
                return [y_min, y_max, x_min, x_max]

    return None


# ============================================================================
# DISTRACTOR PROBE EXPERIMENT
# ============================================================================

class DistractorSensitivityProbe:
    """Runs distractor sensitivity experiments on SmolVLA."""

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

        images = []
        images_np = []  # Keep numpy versions for masking

        for camera_file_name in CAMERA_FILE_NAMES:
            image_path = case_path / "images" / f"step_{step:04d}_{camera_file_name}.jpg"
            if not image_path.exists():
                image_path = case_path / "images" / f"step_{step:04d}_{camera_file_name}.png"

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0

            images_np.append(img)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            images.append(img_tensor)

        # Load state
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

        # Load task
        metadata_path = case_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        task = metadata.get("task_original", metadata.get("task", ""))

        return {
            "images": images,
            "images_np": images_np,
            "state": state,
            "task": task,
            "case_name": case_path.name
        }

    def prepare_batch(self, case_data: dict, masked_images: List[np.ndarray] = None) -> dict:
        """Prepare batch for model inference."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

        images = []
        img_source = masked_images if masked_images else case_data["images_np"]

        for img in img_source:
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            else:
                img_tensor = img
            images.append(img_tensor.unsqueeze(0))

        state = torch.tensor(case_data["state"], dtype=torch.float32).unsqueeze(0)

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
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].bool().to(self.device),
        }

        for idx, camera_model_name in enumerate(CAMERA_MODEL_NAMES):
            key = f"observation.images.{camera_model_name}"
            batch[key] = images[idx].to(self.device)

        return batch

    def run_inference(self, batch: dict) -> np.ndarray:
        """Run inference and return action trajectory."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        with torch.no_grad():
            images, img_masks = self.model.prepare_images(batch)
            state = self.model.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = self.model.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.config.use_cache,
                fill_kv_cache=True,
            )

            # Run denoising loop
            bsize = state.shape[0]
            device = state.device
            actions_shape = (bsize, self.model.config.chunk_size, self.model.config.max_action_dim)
            noise = self.model.model.sample_noise(actions_shape, device)

            num_steps = self.model.config.num_steps
            dt = -1.0 / num_steps
            x_t = noise

            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
                v_t = self.model.model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t

            actions = x_t

            actions_np = actions[0].float().cpu().numpy()

        return actions_np

    def classify_trajectory_shape(self, actions: np.ndarray) -> str:
        """Classify trajectory shape."""
        velocities = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        first_half_vel = np.mean(velocities[:len(velocities)//2])
        second_half_vel = np.mean(velocities[len(velocities)//2:])
        overall_vel = np.mean(velocities)

        if overall_vel < 0.5:
            return "FLAT"
        elif second_half_vel > first_half_vel * 1.5:
            return "RAMP_UP"
        elif first_half_vel > second_half_vel * 1.5:
            return "RAMP_DOWN"
        else:
            return "IRREGULAR"

    def compute_action_delta(self, actions: np.ndarray) -> float:
        """Compute mean action delta."""
        deltas = np.diff(actions, axis=0)
        return float(np.mean(np.linalg.norm(deltas, axis=1)))

    def compare_trajectories(self, normal: np.ndarray, masked: np.ndarray) -> Dict:
        """Compare two trajectories."""
        l2_dist = float(np.linalg.norm(normal - masked))

        normal_flat = normal.flatten()
        masked_flat = masked.flatten()
        cosine_sim = float(np.dot(normal_flat, masked_flat) /
                          (np.linalg.norm(normal_flat) * np.linalg.norm(masked_flat) + 1e-8))

        # Per-joint change
        joint_changes = np.mean(np.abs(normal - masked), axis=0)
        max_joint_idx = int(np.argmax(joint_changes))
        max_joint_val = float(joint_changes[max_joint_idx])

        return {
            "l2_distance": l2_dist,
            "cosine_similarity": cosine_sim,
            "max_joint_idx": max_joint_idx,
            "max_joint_val": max_joint_val,
        }

    def run_probe(self, case_dir: str, step: int,
                  distractor_regions: Dict[str, List[int]] = None,
                  mask_method: str = "blur",
                  auto_detect: bool = True) -> DistractorProbeAnalysis:
        """Run full distractor sensitivity probe."""
        case_data = self.load_case_data(case_dir, step)

        # Use provided regions or defaults
        regions = distractor_regions or DEFAULT_DISTRACTOR_REGIONS.copy()

        # Auto-detect distractor if enabled
        if auto_detect:
            for cam_idx, cam_name in enumerate(CAMERA_FILE_NAMES):
                if regions.get(cam_name) is None:
                    detected = detect_distractor_region(case_data["images_np"][cam_idx])
                    if detected:
                        regions[cam_name] = detected
                        print(f"  Auto-detected distractor in {cam_name}: {detected}")

        # Run normal inference
        print("  Running normal inference...")
        normal_batch = self.prepare_batch(case_data)
        normal_actions = self.run_inference(normal_batch)
        normal_delta = self.compute_action_delta(normal_actions)
        normal_shape = self.classify_trajectory_shape(normal_actions)

        print(f"    Normal: delta={normal_delta:.3f}, shape={normal_shape}")

        # Select masking function
        mask_func = {
            "blur": apply_blur_mask,
            "mean": apply_mean_mask,
            "inpaint": apply_inpaint_mask,
        }.get(mask_method, apply_blur_mask)

        # Test each camera individually
        per_camera_results = {}
        sensitivity_scores = {}

        for cam_idx, cam_name in enumerate(CAMERA_FILE_NAMES):
            region = regions.get(cam_name)
            if region is None:
                print(f"  Skipping {cam_name} (no distractor region)")
                continue

            print(f"  Testing {cam_name} with {mask_method} masking...")

            # Create masked images (only mask this camera)
            masked_images = case_data["images_np"].copy()
            masked_images[cam_idx] = mask_func(case_data["images_np"][cam_idx], region)

            # Run inference
            masked_batch = self.prepare_batch(case_data, masked_images)
            masked_actions = self.run_inference(masked_batch)
            masked_delta = self.compute_action_delta(masked_actions)
            masked_shape = self.classify_trajectory_shape(masked_actions)

            # Compare
            comparison = self.compare_trajectories(normal_actions, masked_actions)

            delta_change = masked_delta - normal_delta
            delta_change_pct = (delta_change / (normal_delta + 1e-8)) * 100

            result = SensitivityResult(
                camera_name=cam_name,
                mask_method=mask_method,
                trajectory_l2_distance=comparison["l2_distance"],
                trajectory_cosine_similarity=comparison["cosine_similarity"],
                normal_action_delta=normal_delta,
                masked_action_delta=masked_delta,
                action_delta_change=delta_change,
                action_delta_change_pct=delta_change_pct,
                normal_trajectory_shape=normal_shape,
                masked_trajectory_shape=masked_shape,
                shape_changed=(normal_shape != masked_shape),
                max_joint_change_idx=comparison["max_joint_idx"],
                max_joint_change_value=comparison["max_joint_val"],
            )

            per_camera_results[cam_name] = result
            sensitivity_scores[cam_name] = comparison["l2_distance"]

            print(f"    Masked: delta={masked_delta:.3f}, shape={masked_shape}, "
                  f"L2={comparison['l2_distance']:.2f}, change={delta_change_pct:.1f}%")

        # Test all cameras masked
        all_masked_result = None
        active_regions = [(i, n, r) for i, (n, r) in enumerate(zip(CAMERA_FILE_NAMES, [regions.get(n) for n in CAMERA_FILE_NAMES])) if r]

        if len(active_regions) > 1:
            print("  Testing all cameras masked...")
            masked_images = case_data["images_np"].copy()
            for cam_idx, cam_name, region in active_regions:
                masked_images[cam_idx] = mask_func(masked_images[cam_idx], region)

            masked_batch = self.prepare_batch(case_data, masked_images)
            masked_actions = self.run_inference(masked_batch)
            masked_delta = self.compute_action_delta(masked_actions)
            masked_shape = self.classify_trajectory_shape(masked_actions)

            comparison = self.compare_trajectories(normal_actions, masked_actions)
            delta_change = masked_delta - normal_delta
            delta_change_pct = (delta_change / (normal_delta + 1e-8)) * 100

            all_masked_result = SensitivityResult(
                camera_name="all_cameras",
                mask_method=mask_method,
                trajectory_l2_distance=comparison["l2_distance"],
                trajectory_cosine_similarity=comparison["cosine_similarity"],
                normal_action_delta=normal_delta,
                masked_action_delta=masked_delta,
                action_delta_change=delta_change,
                action_delta_change_pct=delta_change_pct,
                normal_trajectory_shape=normal_shape,
                masked_trajectory_shape=masked_shape,
                shape_changed=(normal_shape != masked_shape),
                max_joint_change_idx=comparison["max_joint_idx"],
                max_joint_change_value=comparison["max_joint_val"],
            )

            print(f"    All masked: delta={masked_delta:.3f}, shape={masked_shape}, "
                  f"L2={comparison['l2_distance']:.2f}")

        # Summary
        most_sensitive = max(sensitivity_scores.keys(), key=lambda k: sensitivity_scores[k]) if sensitivity_scores else "none"
        overall_sensitivity = max(sensitivity_scores.values()) if sensitivity_scores else 0.0

        # Determine if hallucination is caused by distractor
        # Criteria: significant L2 change AND shape changes from RAMP to FLAT
        halluc_caused = False
        if per_camera_results:
            for result in per_camera_results.values():
                if result.trajectory_l2_distance > 5.0:  # Significant L2 threshold
                    if result.shape_changed and result.masked_trajectory_shape == "FLAT":
                        halluc_caused = True
                        break

        return DistractorProbeAnalysis(
            case_name=case_data["case_name"],
            step=step,
            masked_cameras=list(per_camera_results.keys()),
            mask_method=mask_method,
            normal_action_delta=normal_delta,
            normal_trajectory_shape=normal_shape,
            normal_trajectory=normal_actions.tolist(),
            per_camera_results={k: asdict(v) for k, v in per_camera_results.items()},
            all_masked_result=asdict(all_masked_result) if all_masked_result else None,
            most_sensitive_camera=most_sensitive,
            sensitivity_score=overall_sensitivity,
            hallucination_caused_by_distractor=halluc_caused,
        )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_masking(case_data: dict, regions: Dict[str, List[int]],
                     mask_method: str, output_dir: Path):
    """Visualize original and masked images."""
    mask_func = {
        "blur": apply_blur_mask,
        "mean": apply_mean_mask,
        "inpaint": apply_inpaint_mask,
    }.get(mask_method, apply_blur_mask)

    n_cams = len(CAMERA_FILE_NAMES)
    fig, axes = plt.subplots(2, n_cams, figsize=(5 * n_cams, 10))

    for cam_idx, cam_name in enumerate(CAMERA_FILE_NAMES):
        # Original
        axes[0, cam_idx].imshow(case_data["images_np"][cam_idx])
        axes[0, cam_idx].set_title(f"{cam_name} - Original")
        axes[0, cam_idx].axis('off')

        # Draw region rectangle if exists
        region = regions.get(cam_name)
        if region:
            y_min, y_max, x_min, x_max = region
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[0, cam_idx].add_patch(rect)

            # Masked
            masked = mask_func(case_data["images_np"][cam_idx], region)
            axes[1, cam_idx].imshow(masked)
            axes[1, cam_idx].set_title(f"{cam_name} - Masked ({mask_method})")
        else:
            axes[1, cam_idx].imshow(case_data["images_np"][cam_idx])
            axes[1, cam_idx].set_title(f"{cam_name} - No masking")

        axes[1, cam_idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "masking_visualization.png", dpi=150)
    plt.close()


def visualize_sensitivity_results(analysis: DistractorProbeAnalysis, output_dir: Path):
    """Visualize sensitivity probe results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cameras = list(analysis.per_camera_results.keys())
    if not cameras:
        print("No camera results to visualize")
        return

    # 1. Trajectory L2 distance by camera
    ax1 = axes[0, 0]
    l2_dists = [analysis.per_camera_results[c]["trajectory_l2_distance"] for c in cameras]
    colors = ['red' if d > 5 else 'blue' for d in l2_dists]
    ax1.bar(cameras, l2_dists, color=colors, alpha=0.7)
    ax1.axhline(y=5, color='red', linestyle='--', label='Significance threshold')
    ax1.set_xlabel("Camera")
    ax1.set_ylabel("Trajectory L2 Distance")
    ax1.set_title("How Much Does Trajectory Change When Distractor Masked?")
    ax1.legend()

    # 2. Action delta change by camera
    ax2 = axes[0, 1]
    delta_changes = [analysis.per_camera_results[c]["action_delta_change"] for c in cameras]
    colors = ['red' if d < 0 else 'blue' for d in delta_changes]
    ax2.bar(cameras, delta_changes, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Camera")
    ax2.set_ylabel("Action Delta Change")
    ax2.set_title("Movement Change When Distractor Masked\n(Negative = Less Movement)")

    # 3. Cosine similarity
    ax3 = axes[1, 0]
    cosines = [analysis.per_camera_results[c]["trajectory_cosine_similarity"] for c in cameras]
    ax3.bar(cameras, cosines, color='green', alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='--')
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("Camera")
    ax3.set_ylabel("Cosine Similarity")
    ax3.set_title("Trajectory Direction Similarity (1.0 = same direction)")

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = [["Camera", "L2 Dist", "Δ Change", "Shape Changed?"]]
    for c in cameras:
        r = analysis.per_camera_results[c]
        shape_changed = "YES" if r["shape_changed"] else "no"
        table_data.append([
            c,
            f"{r['trajectory_l2_distance']:.2f}",
            f"{r['action_delta_change_pct']:.1f}%",
            shape_changed
        ])

    # Add all-masked if available
    if analysis.all_masked_result:
        r = analysis.all_masked_result
        shape_changed = "YES" if r["shape_changed"] else "no"
        table_data.append([
            "ALL",
            f"{r['trajectory_l2_distance']:.2f}",
            f"{r['action_delta_change_pct']:.1f}%",
            shape_changed
        ])

    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    conclusion = "YES" if analysis.hallucination_caused_by_distractor else "NO"
    ax4.set_title(f"Summary\nMost Sensitive: {analysis.most_sensitive_camera}\n"
                 f"Hallucination Caused by Distractor: {conclusion}")

    plt.suptitle(f"Distractor Sensitivity Analysis: {analysis.case_name}\n"
                f"Step {analysis.step}, Normal shape: {analysis.normal_trajectory_shape}")
    plt.tight_layout()
    plt.savefig(output_dir / f"sensitivity_results_{analysis.case_name[:30]}.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: DistractorProbeAnalysis, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Distractor Sensitivity Probe Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
This probe tests if the model uses distractor objects (like the banana) as spurious
landmarks by masking the distractor and measuring trajectory changes.

If masking the distractor causes significant trajectory changes (especially reducing
movement in hallucination cases), this provides causal evidence that the distractor
drives the hallucination behavior.
""")

    report.append("\n## Experiment Setup")
    report.append(f"- Case: {analysis.case_name}")
    report.append(f"- Step: {analysis.step}")
    report.append(f"- Mask method: {analysis.mask_method}")
    report.append(f"- Cameras masked: {', '.join(analysis.masked_cameras)}")

    report.append("\n## Baseline Behavior")
    report.append(f"- Normal action delta: {analysis.normal_action_delta:.3f}")
    report.append(f"- Normal trajectory shape: **{analysis.normal_trajectory_shape}**")

    report.append("\n## Per-Camera Sensitivity Results")
    report.append("\n| Camera | L2 Distance | Delta Change | Shape Changed | New Shape |")
    report.append("|--------|-------------|--------------|---------------|-----------|")

    for cam in analysis.masked_cameras:
        r = analysis.per_camera_results[cam]
        shape_changed = "**YES**" if r["shape_changed"] else "no"
        new_shape = r["masked_trajectory_shape"]
        report.append(f"| {cam} | {r['trajectory_l2_distance']:.2f} | "
                     f"{r['action_delta_change_pct']:.1f}% | {shape_changed} | {new_shape} |")

    if analysis.all_masked_result:
        r = analysis.all_masked_result
        shape_changed = "**YES**" if r["shape_changed"] else "no"
        report.append(f"| ALL | {r['trajectory_l2_distance']:.2f} | "
                     f"{r['action_delta_change_pct']:.1f}% | {shape_changed} | {r['masked_trajectory_shape']} |")

    report.append("\n## Key Findings")

    report.append(f"\n### Most Sensitive Camera: **{analysis.most_sensitive_camera}**")
    if analysis.most_sensitive_camera in analysis.per_camera_results:
        r = analysis.per_camera_results[analysis.most_sensitive_camera]
        report.append(f"- Trajectory L2 distance: {r['trajectory_l2_distance']:.2f}")
        report.append(f"- Action delta change: {r['action_delta_change_pct']:.1f}%")
        if r["shape_changed"]:
            report.append(f"- Shape changed from {r['normal_trajectory_shape']} to {r['masked_trajectory_shape']}")

    report.append("\n### Conclusion")
    if analysis.hallucination_caused_by_distractor:
        report.append("""
**The hallucination IS caused by the distractor.**

Evidence:
- Masking the distractor caused significant trajectory change (L2 > 5)
- The trajectory shape changed from RAMP (movement) to FLAT (stay still)
- This proves the distractor is causally driving the hallucination behavior
""")
    else:
        report.append("""
**The hallucination is NOT primarily caused by the distractor.**

Evidence:
- Masking the distractor did not cause significant trajectory change
- The trajectory shape did not change to FLAT
- Other factors (e.g., training data distribution) may be the primary cause
""")

    report.append("\n## Interpretation Guide")
    report.append("""
- **High L2 distance** (>5): Masking this camera significantly changes the trajectory
- **Negative delta change**: Masking reduces movement → distractor was driving motion
- **Shape changed to FLAT**: Masking makes the robot stay still → distractor was causing hallucination

If masking the right wrist camera (where banana is visible) causes:
1. High L2 distance
2. Negative delta change
3. Shape change from RAMP_UP to FLAT

Then we have strong causal evidence that the banana is driving the hallucination.
""")

    report.append("\n## Visualizations")
    report.append("\n- `masking_visualization.png`: Original vs masked images")
    report.append("- `sensitivity_results_*.png`: Per-camera sensitivity metrics")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        get_default_checkpoint, CASE_HALLUC, get_output_dir, DEFAULT_STEP, DEVICE
    )

    parser = argparse.ArgumentParser(description="Run distractor sensitivity probe")
    parser.add_argument("--checkpoint", default=get_default_checkpoint(),
                       help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dir", default=str(CASE_HALLUC),
                       help="Path to case directory")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                       help="Inference step to analyze")
    parser.add_argument("--mask-method", default="blur",
                       choices=["blur", "mean", "inpaint"],
                       help="Masking method")
    parser.add_argument("--auto-detect", action="store_true", default=True,
                       help="Auto-detect distractor regions")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")
    parser.add_argument("--device", default=DEVICE, help="Device to use")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("distractor_probe")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize probe
    probe = DistractorSensitivityProbe(args.checkpoint, args.device)
    probe.load_model()

    # Load case data for visualization
    case_data = probe.load_case_data(args.case_dir, args.step)

    # Run probe
    print(f"\nRunning distractor sensitivity probe at step {args.step}...")
    analysis = probe.run_probe(
        args.case_dir,
        args.step,
        mask_method=args.mask_method,
        auto_detect=args.auto_detect
    )

    print(f"\nMost sensitive camera: {analysis.most_sensitive_camera}")
    print(f"Hallucination caused by distractor: {analysis.hallucination_caused_by_distractor}")

    # Save raw data
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    regions = {c: analysis.per_camera_results[c].get("masked_region") if c in analysis.per_camera_results else None
               for c in CAMERA_FILE_NAMES}
    # Use default regions for visualization
    regions = DEFAULT_DISTRACTOR_REGIONS
    visualize_masking(case_data, regions, args.mask_method, output_dir)
    visualize_sensitivity_results(analysis, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
