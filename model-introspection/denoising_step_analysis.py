#!/usr/bin/env python3
"""
Denoising Step Analysis Tool for SmolVLA Hallucination Investigation.

Traces the evolution of action trajectories through the 10-step denoising process
to identify exactly when the RAMP vs FLAT trajectory shape emerges.

This tool helps answer:
1. Does RAMP emerge at step 0 (KV cache driven) or gradually?
2. At which denoising step does halluc case "lock in" to wrong trajectory?
3. Does the velocity field point different directions from step 0?

Research References:
- FlowPolicy: Shows structure emerges during denoising
- Action Coherence Guidance: Shows how noisy patterns accumulate

Usage:
    python denoising_step_analysis.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir outputs/denoising_analysis
"""

import argparse
import json
import sys
import functools
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
IMAGE_SIZE = 512
# Camera names as they appear in case directories
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
# Camera names as expected by the model config
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]
CHUNK_SIZE = 50  # Action chunk size
ACTION_DIM = 32  # Padded action dimension for SmolVLA
NUM_DENOISING_STEPS = 10

# Joint indices of interest (for bimanual arm)
# Left arm: 0-5 (joints) + 6 (gripper)
# Right arm: 7-12 (joints) + 13 (gripper)
LEFT_SHOULDER_LIFT = 1  # Usually most visible in pick tasks
RIGHT_SHOULDER_LIFT = 8


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DenoiseStepMetrics:
    """Metrics captured at a single denoising step."""
    step: int  # 0-9
    time: float  # 1.0 -> 0.1

    # Trajectory shape metrics
    x_t_mean: List[float]  # Mean of x_t across action dim [50]
    x_t_std: List[float]
    v_t_mean: List[float]  # Velocity prediction mean [50]
    v_t_std: List[float]

    # Key joint trajectories
    left_shoulder_x_t: List[float]  # [50] values for left shoulder lift
    right_shoulder_x_t: List[float]  # [50] values for right shoulder lift
    left_shoulder_v_t: List[float]
    right_shoulder_v_t: List[float]

    # Shape classification
    trajectory_shape: str  # FLAT, RAMP_UP, RAMP_DOWN, IRREGULAR

    # Smoothness metrics
    action_delta: float  # |x_t[end] - x_t[start]| for key joint
    velocity_magnitude: float  # ||v_t|| average
    smoothness_score: float  # L2 norm of consecutive diffs


@dataclass
class CaseDenosingAnalysis:
    """Complete denoising analysis for one case."""
    case_name: str
    inference_step: int  # Which inference step (e.g., 200)

    # Per-step metrics
    steps: List[DenoiseStepMetrics]

    # Summary
    initial_shape: str  # Shape at step 0
    final_shape: str  # Shape at step 9
    shape_emergence_step: int  # When final shape first appears
    shape_stable_from_step: int  # When shape stops changing

    # Joint evolution matrix [10 steps x 50 positions]
    left_shoulder_evolution: List[List[float]]
    right_shoulder_evolution: List[List[float]]


@dataclass
class DenosingComparison:
    """Comparison of denoising process between two cases."""
    case_a: str
    case_b: str
    inference_step: int

    # Shape divergence
    shape_diverges_at_step: int  # When shapes first differ
    velocity_diverges_at_step: int  # When v_t directions differ significantly

    # Per-step comparison
    per_step_trajectory_similarity: List[float]  # Cosine sim of x_t
    per_step_velocity_similarity: List[float]  # Cosine sim of v_t

    # Key finding
    divergence_driven_by: str  # "initial_noise", "early_velocity", "late_denoising"


# ============================================================================
# TRAJECTORY SHAPE CLASSIFICATION
# ============================================================================

def classify_trajectory_shape(values: List[float], threshold: float = 0.1) -> str:
    """Classify trajectory shape based on values across 50 positions."""
    if len(values) < 2:
        return "UNKNOWN"

    values = np.array(values)
    start_mean = np.mean(values[:10])
    end_mean = np.mean(values[-10:])
    delta = end_mean - start_mean
    std = np.std(values)

    if std < threshold and abs(delta) < threshold:
        return "FLAT"
    elif delta > threshold:
        return "RAMP_UP"
    elif delta < -threshold:
        return "RAMP_DOWN"
    else:
        return "IRREGULAR"


def compute_smoothness(values: List[float]) -> float:
    """Compute smoothness as inverse of consecutive difference magnitude."""
    if len(values) < 2:
        return 0.0
    values = np.array(values)
    diffs = np.diff(values)
    return float(np.linalg.norm(diffs))


# ============================================================================
# DENOISING CAPTURE HOOK
# ============================================================================

class DenoisingCaptureHook:
    """Hooks into SmolVLA's denoising loop to capture intermediate states."""

    def __init__(self):
        self.captures = []  # List of DenoiseStepMetrics
        self.current_step = 0
        self.is_capturing = False

    def reset(self):
        """Reset captures for a new inference."""
        self.captures = []
        self.current_step = 0
        self.is_capturing = True

    def capture_step(self, step: int, time: float, x_t: torch.Tensor, v_t: torch.Tensor):
        """Capture metrics at a denoising step."""
        if not self.is_capturing:
            return

        # x_t and v_t shape: [batch, 50, 32]
        # Convert to float32 before numpy (BFloat16 not supported by numpy)
        x_t_np = x_t[0].float().cpu().numpy()  # [50, 32]
        v_t_np = v_t[0].float().cpu().numpy()

        # Compute per-position means
        x_t_mean = x_t_np.mean(axis=1).tolist()  # [50]
        x_t_std = x_t_np.std(axis=1).tolist()
        v_t_mean = v_t_np.mean(axis=1).tolist()
        v_t_std = v_t_np.std(axis=1).tolist()

        # Extract key joint trajectories
        left_shoulder_x_t = x_t_np[:, LEFT_SHOULDER_LIFT].tolist()
        right_shoulder_x_t = x_t_np[:, RIGHT_SHOULDER_LIFT].tolist()
        left_shoulder_v_t = v_t_np[:, LEFT_SHOULDER_LIFT].tolist()
        right_shoulder_v_t = v_t_np[:, RIGHT_SHOULDER_LIFT].tolist()

        # Classify shape based on left shoulder (usually most active)
        shape = classify_trajectory_shape(left_shoulder_x_t)

        # Compute metrics
        action_delta = abs(left_shoulder_x_t[-1] - left_shoulder_x_t[0])
        velocity_magnitude = float(np.mean(np.abs(v_t_np)))
        smoothness = compute_smoothness(left_shoulder_x_t)

        metrics = DenoiseStepMetrics(
            step=step,
            time=time,
            x_t_mean=x_t_mean,
            x_t_std=x_t_std,
            v_t_mean=v_t_mean,
            v_t_std=v_t_std,
            left_shoulder_x_t=left_shoulder_x_t,
            right_shoulder_x_t=right_shoulder_x_t,
            left_shoulder_v_t=left_shoulder_v_t,
            right_shoulder_v_t=right_shoulder_v_t,
            trajectory_shape=shape,
            action_delta=action_delta,
            velocity_magnitude=velocity_magnitude,
            smoothness_score=smoothness
        )
        self.captures.append(metrics)
        self.current_step = step

    def get_analysis(self, case_name: str, inference_step: int) -> CaseDenosingAnalysis:
        """Build analysis from captured data."""
        if not self.captures:
            raise ValueError("No captures available")

        # Determine shape emergence
        final_shape = self.captures[-1].trajectory_shape
        emergence_step = 0
        stable_from = 0

        for i, cap in enumerate(self.captures):
            if cap.trajectory_shape == final_shape:
                emergence_step = i
                break

        # Find when shape becomes stable
        for i in range(len(self.captures) - 1, -1, -1):
            if self.captures[i].trajectory_shape != final_shape:
                stable_from = i + 1
                break

        # Build evolution matrices
        left_evolution = [cap.left_shoulder_x_t for cap in self.captures]
        right_evolution = [cap.right_shoulder_x_t for cap in self.captures]

        return CaseDenosingAnalysis(
            case_name=case_name,
            inference_step=inference_step,
            steps=[asdict(cap) for cap in self.captures],
            initial_shape=self.captures[0].trajectory_shape,
            final_shape=final_shape,
            shape_emergence_step=emergence_step,
            shape_stable_from_step=stable_from,
            left_shoulder_evolution=left_evolution,
            right_shoulder_evolution=right_evolution
        )


# ============================================================================
# DENOISING EXTRACTOR
# ============================================================================

class DenoisingExtractor:
    """Extracts denoising trajectory data from SmolVLA model."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.capture_hook = DenoisingCaptureHook()

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

    def extract_denoising_trajectory(self, case_dir: str, step: int) -> CaseDenosingAnalysis:
        """Extract denoising trajectory for a case at a specific step."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        case_data = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(case_data)

        # Prepare inputs
        # prepare_images/prepare_state are on SmolVLAPolicy (self.model)
        images, img_masks = self.model.prepare_images(batch)
        state = self.model.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]

        # Reset capture hook
        self.capture_hook.reset()

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

            # Manual denoising loop with capture
            bsize = state.shape[0]
            device = state.device
            actions_shape = (bsize, CHUNK_SIZE, ACTION_DIM)
            x_t = self.model.model.sample_noise(actions_shape, device)

            num_steps = NUM_DENOISING_STEPS
            dt = -1.0 / num_steps

            for denoise_step in range(num_steps):
                time = 1.0 + denoise_step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

                v_t = self.model.model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )

                # Capture before update
                self.capture_hook.capture_step(denoise_step, time, x_t, v_t)

                # Euler update
                x_t = x_t + dt * v_t

        return self.capture_hook.get_analysis(case_data["case_name"], step)


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def cosine_similarity_list(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two lists."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compare_denoising(analysis_a: CaseDenosingAnalysis,
                      analysis_b: CaseDenosingAnalysis) -> DenosingComparison:
    """Compare denoising processes between two cases."""
    # Per-step trajectory and velocity similarity
    traj_similarity = []
    vel_similarity = []
    shape_diverges = -1
    vel_diverges = -1

    for i in range(len(analysis_a.steps)):
        step_a = analysis_a.steps[i]
        step_b = analysis_b.steps[i]

        # Trajectory similarity (x_t)
        traj_sim = cosine_similarity_list(
            step_a["left_shoulder_x_t"],
            step_b["left_shoulder_x_t"]
        )
        traj_similarity.append(traj_sim)

        # Velocity similarity (v_t)
        vel_sim = cosine_similarity_list(
            step_a["left_shoulder_v_t"],
            step_b["left_shoulder_v_t"]
        )
        vel_similarity.append(vel_sim)

        # Check for shape divergence
        if shape_diverges < 0 and step_a["trajectory_shape"] != step_b["trajectory_shape"]:
            shape_diverges = i

        # Check for velocity divergence (significant direction change)
        if vel_diverges < 0 and vel_sim < 0.5:
            vel_diverges = i

    # Determine what drives divergence
    if shape_diverges == 0 or (shape_diverges < 0 and traj_similarity[0] < 0.9):
        divergence_driven_by = "initial_noise"
    elif shape_diverges > 0 and shape_diverges <= 3:
        divergence_driven_by = "early_velocity"
    else:
        divergence_driven_by = "late_denoising"

    return DenosingComparison(
        case_a=analysis_a.case_name,
        case_b=analysis_b.case_name,
        inference_step=analysis_a.inference_step,
        shape_diverges_at_step=shape_diverges,
        velocity_diverges_at_step=vel_diverges,
        per_step_trajectory_similarity=traj_similarity,
        per_step_velocity_similarity=vel_similarity,
        divergence_driven_by=divergence_driven_by
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_trajectory_emergence(analysis: CaseDenosingAnalysis, output_dir: Path):
    """Visualize how trajectory shape emerges during denoising."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Left shoulder trajectory evolution
    ax1 = axes[0, 0]
    evolution = np.array(analysis.left_shoulder_evolution)  # [10, 50]
    for step_idx in range(len(analysis.steps)):
        alpha = 0.3 + 0.7 * (step_idx / 9)  # Fade in
        ax1.plot(evolution[step_idx], label=f"Step {step_idx}" if step_idx in [0, 4, 9] else None,
                alpha=alpha, color=plt.cm.viridis(step_idx / 9))
    ax1.set_xlabel("Position in Chunk (0-49)")
    ax1.set_ylabel("Left Shoulder Value")
    ax1.set_title(f"Trajectory Emergence: {analysis.case_name[:30]}")
    ax1.legend()

    # Top-right: Shape evolution
    ax2 = axes[0, 1]
    shapes = [s["trajectory_shape"] for s in analysis.steps]
    shape_to_num = {"FLAT": 0, "RAMP_UP": 1, "RAMP_DOWN": -1, "IRREGULAR": 0.5}
    shape_values = [shape_to_num.get(s, 0) for s in shapes]
    ax2.bar(range(10), shape_values)
    ax2.set_xlabel("Denoising Step")
    ax2.set_ylabel("Shape (0=FLAT, 1=RAMP_UP, -1=RAMP_DOWN)")
    ax2.set_title("Trajectory Shape Evolution")
    ax2.set_xticks(range(10))

    # Add shape labels
    for i, shape in enumerate(shapes):
        ax2.text(i, shape_values[i] + 0.1, shape[:4], ha='center', fontsize=8)

    # Bottom-left: Action delta evolution
    ax3 = axes[1, 0]
    action_deltas = [s["action_delta"] for s in analysis.steps]
    ax3.plot(action_deltas, 'b-o', linewidth=2)
    ax3.set_xlabel("Denoising Step")
    ax3.set_ylabel("Action Delta (|end - start|)")
    ax3.set_title("Action Delta Evolution")
    ax3.set_xticks(range(10))
    ax3.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
    ax3.legend()

    # Bottom-right: Velocity magnitude
    ax4 = axes[1, 1]
    vel_mags = [s["velocity_magnitude"] for s in analysis.steps]
    ax4.plot(vel_mags, 'g-o', linewidth=2)
    ax4.set_xlabel("Denoising Step")
    ax4.set_ylabel("Average |v_t|")
    ax4.set_title("Velocity Magnitude Evolution")
    ax4.set_xticks(range(10))

    plt.tight_layout()
    safe_name = analysis.case_name[:20].replace("/", "_")
    plt.savefig(output_dir / f"trajectory_emergence_{safe_name}.png", dpi=150)
    plt.close()


def visualize_denoising_comparison(comp: DenosingComparison,
                                    analysis_a: CaseDenosingAnalysis,
                                    analysis_b: CaseDenosingAnalysis,
                                    output_dir: Path):
    """Visualize comparison between two cases' denoising processes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Trajectory comparison at each step
    ax1 = axes[0, 0]
    for step_idx in [0, 4, 9]:  # Start, middle, end
        ax1.plot(analysis_a.steps[step_idx]["left_shoulder_x_t"],
                label=f"{analysis_a.case_name[:10]} step {step_idx}",
                linestyle='-', alpha=0.7)
        ax1.plot(analysis_b.steps[step_idx]["left_shoulder_x_t"],
                label=f"{analysis_b.case_name[:10]} step {step_idx}",
                linestyle='--', alpha=0.7)
    ax1.set_xlabel("Position in Chunk")
    ax1.set_ylabel("Left Shoulder Value")
    ax1.set_title("Trajectory Comparison at Steps 0, 4, 9")
    ax1.legend(fontsize=7)

    # Top-right: Similarity evolution
    ax2 = axes[0, 1]
    ax2.plot(comp.per_step_trajectory_similarity, 'b-o', label='Trajectory Cosine Sim')
    ax2.plot(comp.per_step_velocity_similarity, 'r-s', label='Velocity Cosine Sim')
    ax2.set_xlabel("Denoising Step")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Similarity Evolution During Denoising")
    ax2.legend()
    ax2.set_ylim(-0.5, 1.1)
    ax2.set_xticks(range(10))

    # Add divergence markers
    if comp.shape_diverges_at_step >= 0:
        ax2.axvline(x=comp.shape_diverges_at_step, color='green', linestyle='--',
                   label=f'Shape diverges: {comp.shape_diverges_at_step}')
    if comp.velocity_diverges_at_step >= 0:
        ax2.axvline(x=comp.velocity_diverges_at_step, color='orange', linestyle='--',
                   label=f'Velocity diverges: {comp.velocity_diverges_at_step}')

    # Bottom-left: Action delta comparison
    ax3 = axes[1, 0]
    deltas_a = [s["action_delta"] for s in analysis_a.steps]
    deltas_b = [s["action_delta"] for s in analysis_b.steps]
    ax3.plot(deltas_a, 'b-o', label=analysis_a.case_name[:15])
    ax3.plot(deltas_b, 'r-s', label=analysis_b.case_name[:15])
    ax3.set_xlabel("Denoising Step")
    ax3.set_ylabel("Action Delta")
    ax3.set_title("Action Delta Comparison")
    ax3.legend(fontsize=8)
    ax3.set_xticks(range(10))

    # Bottom-right: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    DENOISING COMPARISON SUMMARY

    Case A: {analysis_a.case_name[:35]}
    Case B: {analysis_b.case_name[:35]}

    Shape Diverges at Step: {comp.shape_diverges_at_step}
    Velocity Diverges at Step: {comp.velocity_diverges_at_step}

    Divergence Driven By: {comp.divergence_driven_by}

    Case A Final Shape: {analysis_a.final_shape}
    Case B Final Shape: {analysis_b.final_shape}

    Case A Shape Emergence: Step {analysis_a.shape_emergence_step}
    Case B Shape Emergence: Step {analysis_b.shape_emergence_step}

    Initial Trajectory Similarity: {comp.per_step_trajectory_similarity[0]:.3f}
    Final Trajectory Similarity: {comp.per_step_trajectory_similarity[-1]:.3f}
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    safe_name = f"{comp.case_a[:12]}_{comp.case_b[:12]}"
    plt.savefig(output_dir / f"denoising_comparison_{safe_name}.png", dpi=150)
    plt.close()


def visualize_joint_evolution_grid(analyses: List[CaseDenosingAnalysis], output_dir: Path):
    """Create grid showing joint evolution across denoising for all cases."""
    n_cases = len(analyses)
    fig, axes = plt.subplots(n_cases, 2, figsize=(12, 4 * n_cases))
    if n_cases == 1:
        axes = axes.reshape(1, -1)

    for case_idx, analysis in enumerate(analyses):
        # Left column: Left shoulder
        ax1 = axes[case_idx, 0]
        evolution = np.array(analysis.left_shoulder_evolution)
        im1 = ax1.imshow(evolution, cmap='RdBu', aspect='auto')
        ax1.set_xlabel("Position in Chunk (0-49)")
        ax1.set_ylabel("Denoising Step")
        ax1.set_title(f"{analysis.case_name[:25]}\nLeft Shoulder Evolution")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Right column: Right shoulder
        ax2 = axes[case_idx, 1]
        evolution_r = np.array(analysis.right_shoulder_evolution)
        im2 = ax2.imshow(evolution_r, cmap='RdBu', aspect='auto')
        ax2.set_xlabel("Position in Chunk (0-49)")
        ax2.set_ylabel("Denoising Step")
        ax2.set_title(f"{analysis.case_name[:25]}\nRight Shoulder Evolution")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_dir / "joint_evolution_grid.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analyses: List[CaseDenosingAnalysis],
                   comparisons: List[DenosingComparison],
                   output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Denoising Step Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"\n**Inference step analyzed**: {analyses[0].inference_step}")
    report.append(f"\n**Number of denoising steps**: {NUM_DENOISING_STEPS}")

    report.append("\n## Cases Analyzed")
    for analysis in analyses:
        report.append(f"\n### {analysis.case_name}")
        report.append(f"- Initial shape: **{analysis.initial_shape}**")
        report.append(f"- Final shape: **{analysis.final_shape}**")
        report.append(f"- Shape emergence step: **{analysis.shape_emergence_step}**")
        report.append(f"- Shape stable from step: **{analysis.shape_stable_from_step}**")

        # Per-step summary
        report.append("\n| Step | Time | Shape | Action Delta | Velocity Mag |")
        report.append("|------|------|-------|--------------|--------------|")
        for s in analysis.steps:
            report.append(f"| {s['step']} | {s['time']:.2f} | {s['trajectory_shape']} | "
                         f"{s['action_delta']:.3f} | {s['velocity_magnitude']:.3f} |")

    report.append("\n## Pairwise Comparisons")
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:25]} vs {comp.case_b[:25]}")
        report.append(f"- **Shape diverges at step**: {comp.shape_diverges_at_step}")
        report.append(f"- **Velocity diverges at step**: {comp.velocity_diverges_at_step}")
        report.append(f"- **Divergence driven by**: **{comp.divergence_driven_by}**")

        report.append("\n| Step | Traj Similarity | Vel Similarity |")
        report.append("|------|-----------------|----------------|")
        for i in range(len(comp.per_step_trajectory_similarity)):
            report.append(f"| {i} | {comp.per_step_trajectory_similarity[i]:.3f} | "
                         f"{comp.per_step_velocity_similarity[i]:.3f} |")

    report.append("\n## Key Findings")

    # Analyze patterns
    for comp in comparisons:
        report.append(f"\n### {comp.case_a[:20]} vs {comp.case_b[:20]}")

        if comp.divergence_driven_by == "initial_noise":
            report.append("- **Finding**: Trajectories differ from the very beginning (step 0)")
            report.append("- **Implication**: Divergence is driven by initial noise or KV cache difference")
            report.append("- This suggests the KV cache (derived from different visual input) "
                         "immediately conditions different trajectory sampling")
        elif comp.divergence_driven_by == "early_velocity":
            report.append("- **Finding**: Trajectories start similar but diverge in early steps (1-3)")
            report.append("- **Implication**: Velocity predictions cause divergence during denoising")
            report.append("- The KV cache influences velocity predictions, "
                         "pushing trajectories in different directions")
        else:
            report.append("- **Finding**: Trajectories remain similar until late denoising (steps 4+)")
            report.append("- **Implication**: Fine-grained trajectory details emerge late")
            report.append("- Main structure is similar, divergence is in trajectory details")

        # Interpret shape emergence
        for analysis in analyses:
            if analysis.case_name == comp.case_a or analysis.case_name == comp.case_b:
                if analysis.shape_emergence_step == 0:
                    report.append(f"- {analysis.case_name[:20]}: Shape emerges at step 0 "
                                 f"(KV-cache driven)")
                else:
                    report.append(f"- {analysis.case_name[:20]}: Shape emerges at step "
                                 f"{analysis.shape_emergence_step} (velocity-field driven)")

    report.append("\n## Visualizations")
    report.append("\n- `trajectory_emergence_*.png`: How trajectory shapes emerge per case")
    report.append("- `denoising_comparison_*.png`: Side-by-side comparison of denoising")
    report.append("- `joint_evolution_grid.png`: Heatmap of joint values across denoising")

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

    parser = argparse.ArgumentParser(description="Analyze denoising step evolution")
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
        output_dir = get_output_dir("denoising")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize extractor
    extractor = DenoisingExtractor(args.checkpoint, args.device)
    extractor.load_model()

    # Extract denoising trajectories
    print(f"\nExtracting denoising trajectories at step {args.step}...")
    all_analyses = []
    for case_dir in args.case_dirs:
        print(f"  Processing: {case_dir}")
        analysis = extractor.extract_denoising_trajectory(case_dir, args.step)
        all_analyses.append(analysis)
        print(f"    Initial shape: {analysis.initial_shape}, Final shape: {analysis.final_shape}")
        print(f"    Shape emergence: step {analysis.shape_emergence_step}")

    # Compare all pairs
    print("\nComparing denoising processes...")
    comparisons = []
    for i in range(len(all_analyses)):
        for j in range(i + 1, len(all_analyses)):
            comp = compare_denoising(all_analyses[i], all_analyses[j])
            comparisons.append(comp)
            print(f"  {comp.case_a[:25]} vs {comp.case_b[:25]}: "
                  f"shape_diverges={comp.shape_diverges_at_step}, "
                  f"driven_by={comp.divergence_driven_by}")

    # Save raw data
    analysis_data = [asdict(a) for a in all_analyses]
    with open(output_dir / "analyses.json", "w") as f:
        json.dump(analysis_data, f, indent=2)

    comparison_data = [asdict(c) for c in comparisons]
    with open(output_dir / "comparisons.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    for analysis in all_analyses:
        visualize_trajectory_emergence(analysis, output_dir)

    for i, comp in enumerate(comparisons):
        visualize_denoising_comparison(comp, all_analyses[i], all_analyses[i+1], output_dir)

    visualize_joint_evolution_grid(all_analyses, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_analyses, comparisons, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
