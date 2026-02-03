#!/usr/bin/env python3
"""
Denoising Process Analyzer for SmolVLA Hallucination Investigation.

Captures and visualizes the 10-step flow-matching denoising process to identify
where hallucination-causing actions emerge in the trajectory from noise to action.

Key Questions:
- At which denoising step does hallucination action emerge?
- Is the velocity field (v_t) anomalous in hallucination cases?
- Does the denoising trajectory differ between normal and hallucination cases?

Usage:
    # Analyze a single inference run
    python analyze_denoising.py \
        --checkpoint outputs/smolvla_bimanual/checkpoints/080000/pretrained_model \
        --task-key left_yogurt_bin \
        --output-dir ../reports/denoising_analysis

    # Compare two cases
    python analyze_denoising.py \
        --compare \
        --case1 ../cases/hallucination/case_001 \
        --case2 ../cases/normal/case_002 \
        --output-dir ../reports/comparison

Output:
    - Denoising trajectory plots (12D joint space evolution)
    - Velocity field analysis (direction and magnitude)
    - Step-by-step action emergence visualization
    - Comparison plots (if comparing cases)
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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
class DenoiseStep:
    """Data from a single denoising step."""
    step: int               # 0-9
    time: float             # 1.0 -> 0.0
    x_t: np.ndarray         # Current noisy action [chunk_size, action_dim]
    v_t: np.ndarray         # Velocity (direction of change)
    x_t_mean: np.ndarray    # Mean across chunk [action_dim]
    v_t_mean: np.ndarray    # Mean velocity
    x_t_std: np.ndarray     # Std across chunk
    v_t_norm: float         # Velocity magnitude


@dataclass
class DenoiseTrace:
    """Complete trace of one action chunk generation."""
    chunk_index: int
    timestamp: float
    task: str
    steps: list[DenoiseStep] = field(default_factory=list)
    final_action: Optional[np.ndarray] = None
    initial_noise: Optional[np.ndarray] = None


# ============================================================================
# DENOISING CAPTURE HOOKS
# ============================================================================

class DenoisingCapture:
    """Captures denoising steps during SmolVLA inference."""

    def __init__(self):
        self.traces: list[DenoiseTrace] = []
        self.current_trace: Optional[DenoiseTrace] = None
        self._hook_handle = None

    def start_trace(self, chunk_index: int, task: str, initial_noise: np.ndarray = None):
        """Start capturing a new denoising trace."""
        self.current_trace = DenoiseTrace(
            chunk_index=chunk_index,
            timestamp=datetime.now().timestamp(),
            task=task,
            initial_noise=initial_noise,
        )

    def record_step(
        self,
        step: int,
        time: float,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
    ):
        """Record a single denoising step."""
        if self.current_trace is None:
            return

        x_t_np = x_t.detach().cpu().numpy()
        v_t_np = v_t.detach().cpu().numpy()

        # x_t shape: [batch, chunk_size, action_dim] -> take first batch
        if x_t_np.ndim == 3:
            x_t_np = x_t_np[0]
        if v_t_np.ndim == 3:
            v_t_np = v_t_np[0]

        denoise_step = DenoiseStep(
            step=step,
            time=time,
            x_t=x_t_np,
            v_t=v_t_np,
            x_t_mean=x_t_np.mean(axis=0),
            v_t_mean=v_t_np.mean(axis=0),
            x_t_std=x_t_np.std(axis=0),
            v_t_norm=float(np.linalg.norm(v_t_np.mean(axis=0))),
        )

        self.current_trace.steps.append(denoise_step)

    def end_trace(self, final_action: np.ndarray = None):
        """End current trace and save it."""
        if self.current_trace is not None:
            self.current_trace.final_action = final_action
            self.traces.append(self.current_trace)
            self.current_trace = None

    def get_traces(self) -> list[DenoiseTrace]:
        return self.traces

    def clear(self):
        self.traces = []
        self.current_trace = None


def create_denoising_hook(capture: DenoisingCapture, step_counter: list):
    """Create a hook function for the denoising loop.

    This patches into SmolVLA's sample_actions method to capture each step.
    """
    def hook(x_t: torch.Tensor, v_t: torch.Tensor, time: float):
        capture.record_step(
            step=step_counter[0],
            time=time,
            x_t=x_t,
            v_t=v_t,
        )
        step_counter[0] += 1

    return hook


# ============================================================================
# MODIFIED INFERENCE FOR DENOISING CAPTURE
# ============================================================================

def run_denoising_capture(
    policy,
    preprocessor,
    observation: dict,
    capture: DenoisingCapture,
    chunk_index: int,
    task: str,
) -> np.ndarray:
    """
    Run a single action chunk generation with denoising capture.

    This manually calls into the model's denoising loop to capture each step.
    """
    # Preprocess observation
    batch = preprocessor(observation)

    # Prepare inputs for model.sample_actions
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch["observation.language_tokens"]
    lang_masks = batch["observation.language_attention_mask"]

    device = state.device
    bsize = state.shape[0]

    # Generate initial noise
    actions_shape = (bsize, policy.config.chunk_size, policy.config.max_action_dim)
    noise = policy.model.sample_noise(actions_shape, device)

    # Start trace
    capture.start_trace(
        chunk_index=chunk_index,
        task=task,
        initial_noise=noise.cpu().numpy()[0] if noise is not None else None,
    )

    # Manually run denoising loop (mirrors VLAFlowMatching.sample_actions)
    model = policy.model

    # Compute prefix embeddings and KV cache
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Compute KV cache
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    # Denoising loop
    num_steps = model.config.num_steps
    dt = -1.0 / num_steps

    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        # Denoise step
        v_t = model.denoise_step(
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_tensor,
        )

        # Capture
        capture.record_step(step=step, time=time, x_t=x_t, v_t=v_t)

        # Update
        x_t = x_t + dt * v_t

    # End trace
    final_action = x_t.cpu().numpy()[0]
    capture.end_trace(final_action=final_action)

    return final_action


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_denoising_trajectory(
    trace: DenoiseTrace,
    output_path: Path,
    joint_names: list[str] = None,
):
    """
    Plot the denoising trajectory showing how actions evolve from noise to final.

    Creates a grid showing each joint's trajectory through denoising steps.
    """
    if joint_names is None:
        joint_names = [
            "L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
            "R_pan", "R_lift", "R_elbow", "R_wflex", "R_wroll", "R_grip",
        ]

    num_joints = min(len(joint_names), 12)
    num_steps = len(trace.steps)

    # Extract trajectory data
    times = [s.time for s in trace.steps]
    x_t_means = np.array([s.x_t_mean[:num_joints] for s in trace.steps])

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:num_joints], joint_names)):
        ax.plot(times, x_t_means[:, i], 'b-o', markersize=4, linewidth=1.5)

        # Mark start (noise) and end (action)
        ax.scatter([times[0]], [x_t_means[0, i]], c='red', s=100, zorder=5, label='Noise')
        ax.scatter([times[-1]], [x_t_means[-1, i]], c='green', s=100, zorder=5, label='Action')

        ax.set_xlabel('Time (1→0)')
        ax.set_ylabel('Value')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # Time goes from 1 to 0

        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for ax in axes[num_joints:]:
        ax.set_visible(False)

    plt.suptitle(f'Denoising Trajectory - Chunk {trace.chunk_index}\nTask: {trace.task[:50]}...',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {output_path}")


def plot_velocity_field(
    trace: DenoiseTrace,
    output_path: Path,
):
    """
    Plot velocity field showing direction and magnitude at each step.
    """
    num_steps = len(trace.steps)

    # Extract velocity data
    v_norms = [s.v_t_norm for s in trace.steps]
    v_means = np.array([s.v_t_mean[:12] for s in trace.steps])
    times = [s.time for s in trace.steps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Velocity magnitude over time
    ax1 = axes[0]
    ax1.bar(range(num_steps), v_norms, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Denoising Step')
    ax1.set_ylabel('Velocity Magnitude ||v_t||')
    ax1.set_title('Velocity Magnitude per Step')
    ax1.set_xticks(range(num_steps))

    # Velocity direction (PCA or selected joints)
    ax2 = axes[1]
    # Show velocity for first 4 joints as example
    for i, name in enumerate(['L_pan', 'L_lift', 'L_elbow', 'L_wflex'][:4]):
        ax2.plot(range(num_steps), v_means[:, i], '-o', label=name, markersize=4)
    ax2.set_xlabel('Denoising Step')
    ax2.set_ylabel('Velocity Component')
    ax2.set_title('Velocity Direction (Selected Joints)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Velocity Field Analysis - Chunk {trace.chunk_index}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved velocity plot: {output_path}")


def plot_comparison(
    trace1: DenoiseTrace,
    trace2: DenoiseTrace,
    output_path: Path,
    label1: str = "Hallucination",
    label2: str = "Normal",
):
    """
    Compare denoising trajectories between two cases.
    """
    joint_names = [
        "L_pan", "L_lift", "L_elbow", "L_wflex", "L_wroll", "L_grip",
        "R_pan", "R_lift", "R_elbow", "R_wflex", "R_wroll", "R_grip",
    ]

    num_joints = 12
    num_steps = min(len(trace1.steps), len(trace2.steps))

    times1 = [s.time for s in trace1.steps[:num_steps]]
    times2 = [s.time for s in trace2.steps[:num_steps]]
    x1 = np.array([s.x_t_mean[:num_joints] for s in trace1.steps[:num_steps]])
    x2 = np.array([s.x_t_mean[:num_joints] for s in trace2.steps[:num_steps]])

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:num_joints], joint_names)):
        ax.plot(range(num_steps), x1[:, i], 'r-o', markersize=3, linewidth=1.5, label=label1, alpha=0.8)
        ax.plot(range(num_steps), x2[:, i], 'b-s', markersize=3, linewidth=1.5, label=label2, alpha=0.8)

        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=8)

    for ax in axes[num_joints:]:
        ax.set_visible(False)

    plt.suptitle(f'Denoising Comparison: {label1} vs {label2}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def plot_action_emergence(
    trace: DenoiseTrace,
    output_path: Path,
):
    """
    Visualize how the final action "emerges" through denoising.

    Shows the difference between each step's prediction and the final action.
    """
    num_steps = len(trace.steps)
    final = trace.final_action[:12] if trace.final_action is not None else trace.steps[-1].x_t_mean[:12]

    # Calculate distance to final action at each step
    distances = []
    for step in trace.steps:
        diff = step.x_t_mean[:12] - final
        distances.append(np.linalg.norm(diff))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(num_steps), distances, 'g-o', markersize=6, linewidth=2)
    ax.fill_between(range(num_steps), 0, distances, alpha=0.3, color='green')

    ax.set_xlabel('Denoising Step', fontsize=12)
    ax.set_ylabel('Distance to Final Action', fontsize=12)
    ax.set_title(f'Action Emergence - Chunk {trace.chunk_index}\n'
                 f'How quickly does the action emerge from noise?',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_steps))
    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(f'Initial: {distances[0]:.2f}', xy=(0, distances[0]),
                xytext=(0.5, distances[0] * 1.1), fontsize=10)
    ax.annotate(f'Final: {distances[-1]:.2f}', xy=(num_steps-1, distances[-1]),
                xytext=(num_steps-1.5, distances[-1] + 0.5), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved emergence plot: {output_path}")


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_trace(trace: DenoiseTrace) -> dict:
    """
    Analyze a single denoising trace for anomalies.

    Returns dict with analysis results.
    """
    if not trace.steps:
        return {"error": "No steps in trace"}

    num_steps = len(trace.steps)

    # Extract data
    v_norms = [s.v_t_norm for s in trace.steps]
    x_means = np.array([s.x_t_mean[:12] for s in trace.steps])

    # Velocity analysis
    velocity_mean = np.mean(v_norms)
    velocity_std = np.std(v_norms)
    velocity_max_step = int(np.argmax(v_norms))

    # Trajectory analysis
    total_movement = np.linalg.norm(x_means[-1] - x_means[0])
    per_step_movement = [np.linalg.norm(x_means[i+1] - x_means[i])
                         for i in range(num_steps - 1)]

    # Identify which step has largest movement
    if per_step_movement:
        max_movement_step = int(np.argmax(per_step_movement))
        max_movement_value = per_step_movement[max_movement_step]
    else:
        max_movement_step = 0
        max_movement_value = 0

    # Final action analysis
    final_action = trace.final_action[:12] if trace.final_action is not None else x_means[-1]
    gripper_left = final_action[5]
    gripper_right = final_action[11]

    return {
        "chunk_index": trace.chunk_index,
        "task": trace.task,
        "num_steps": num_steps,
        "velocity": {
            "mean": float(velocity_mean),
            "std": float(velocity_std),
            "max_step": velocity_max_step,
            "max_value": float(v_norms[velocity_max_step]),
        },
        "trajectory": {
            "total_movement": float(total_movement),
            "max_movement_step": max_movement_step,
            "max_movement_value": float(max_movement_value),
            "per_step_movement": [float(m) for m in per_step_movement],
        },
        "final_action": {
            "left_arm_mean": float(np.mean(final_action[:5])),
            "right_arm_mean": float(np.mean(final_action[6:11])),
            "gripper_left": float(gripper_left),
            "gripper_right": float(gripper_right),
        },
    }


def save_trace_data(trace: DenoiseTrace, output_path: Path):
    """Save trace data to JSON for later analysis."""
    data = {
        "chunk_index": trace.chunk_index,
        "timestamp": trace.timestamp,
        "task": trace.task,
        "steps": [
            {
                "step": s.step,
                "time": s.time,
                "x_t_mean": s.x_t_mean.tolist(),
                "v_t_mean": s.v_t_mean.tolist(),
                "x_t_std": s.x_t_std.tolist(),
                "v_t_norm": s.v_t_norm,
            }
            for s in trace.steps
        ],
        "final_action": trace.final_action.tolist() if trace.final_action is not None else None,
        "initial_noise_mean": trace.initial_noise.mean(axis=0).tolist() if trace.initial_noise is not None else None,
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved trace data: {output_path}")


def load_trace_data(path: Path) -> DenoiseTrace:
    """Load trace data from JSON."""
    with open(path) as f:
        data = json.load(f)

    trace = DenoiseTrace(
        chunk_index=data["chunk_index"],
        timestamp=data["timestamp"],
        task=data["task"],
    )

    for s in data["steps"]:
        step = DenoiseStep(
            step=s["step"],
            time=s["time"],
            x_t=np.array([]),  # Full tensor not saved
            v_t=np.array([]),
            x_t_mean=np.array(s["x_t_mean"]),
            v_t_mean=np.array(s["v_t_mean"]),
            x_t_std=np.array(s["x_t_std"]),
            v_t_norm=s["v_t_norm"],
        )
        trace.steps.append(step)

    trace.final_action = np.array(data["final_action"]) if data["final_action"] else None

    return trace


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Denoising Process Analyzer for SmolVLA Hallucination Investigation"
    )

    # Mode selection
    parser.add_argument("--compare", action="store_true", help="Compare two cases")

    # Single run mode
    parser.add_argument("--checkpoint", "-c", help="Path to SmolVLA checkpoint")
    parser.add_argument("--task-key", "-k", help="Task key")
    parser.add_argument("--task", "-t", help="Task description")

    # Compare mode
    parser.add_argument("--case1", type=Path, help="First case directory (for comparison)")
    parser.add_argument("--case2", type=Path, help="Second case directory (for comparison)")
    parser.add_argument("--label1", default="Case 1", help="Label for case 1")
    parser.add_argument("--label2", default="Case 2", help="Label for case 2")

    # Output
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")

    # Load from trace file
    parser.add_argument("--trace-file", type=Path, help="Load existing trace file for visualization")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SmolVLA Denoising Process Analyzer")
    print("=" * 60)

    if args.compare:
        # Comparison mode - load two traces and compare
        if not args.case1 or not args.case2:
            print("ERROR: --case1 and --case2 required for comparison mode")
            sys.exit(1)

        trace1_file = args.case1 / "denoising_trace.json"
        trace2_file = args.case2 / "denoising_trace.json"

        if not trace1_file.exists() or not trace2_file.exists():
            print(f"ERROR: Trace files not found:")
            print(f"  {trace1_file}")
            print(f"  {trace2_file}")
            print("\nRun denoising analysis on each case first.")
            sys.exit(1)

        print(f"Loading case 1: {args.case1}")
        trace1 = load_trace_data(trace1_file)

        print(f"Loading case 2: {args.case2}")
        trace2 = load_trace_data(trace2_file)

        # Generate comparison plots
        plot_comparison(
            trace1, trace2,
            args.output_dir / "comparison_trajectory.png",
            label1=args.label1,
            label2=args.label2,
        )

        # Analyze both
        analysis1 = analyze_trace(trace1)
        analysis2 = analyze_trace(trace2)

        comparison_results = {
            "case1": {
                "path": str(args.case1),
                "label": args.label1,
                "analysis": analysis1,
            },
            "case2": {
                "path": str(args.case2),
                "label": args.label2,
                "analysis": analysis2,
            },
            "comparison": {
                "velocity_diff": analysis1["velocity"]["mean"] - analysis2["velocity"]["mean"],
                "movement_diff": analysis1["trajectory"]["total_movement"] - analysis2["trajectory"]["total_movement"],
            }
        }

        with open(args.output_dir / "comparison_analysis.json", 'w') as f:
            json.dump(comparison_results, f, indent=2)

        print(f"\nResults saved to: {args.output_dir}")

    elif args.trace_file:
        # Visualization mode - load and visualize existing trace
        print(f"Loading trace: {args.trace_file}")
        trace = load_trace_data(args.trace_file)

        # Generate all visualizations
        plot_denoising_trajectory(trace, args.output_dir / "trajectory.png")
        plot_velocity_field(trace, args.output_dir / "velocity.png")
        plot_action_emergence(trace, args.output_dir / "emergence.png")

        # Run analysis
        analysis = analyze_trace(trace)
        with open(args.output_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nAnalysis results:")
        print(json.dumps(analysis, indent=2))
        print(f"\nResults saved to: {args.output_dir}")

    else:
        # Live capture mode - run inference and capture denoising
        if not args.checkpoint:
            print("ERROR: --checkpoint required for live capture mode")
            print("Use --trace-file to analyze existing traces")
            sys.exit(1)

        print(f"Checkpoint: {args.checkpoint}")
        print("\nNOTE: Live capture mode requires hardware setup.")
        print("For offline analysis, use --trace-file with existing traces.")
        print("\nTo capture denoising data during inference:")
        print("1. Use trace_inference.py with --capture-denoising flag")
        print("2. Then run this tool with --trace-file")

        # TODO: Implement live capture when hardware is available


if __name__ == "__main__":
    main()
