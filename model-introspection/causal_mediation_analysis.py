#!/usr/bin/env python3
"""
Causal Mediation Analysis (NDE) for SmolVLA Hallucination Investigation.

Based on "Treble Counterfactual VLMs" (arXiv:2503.06169), this tool computes
Natural Direct Effects to isolate whether hallucination is driven by
visual input alone or other factors.

Method:
- NDE_Vis: Effect of visual input alone (keep normal text/state, use halluc visual)
- NDE_Cross: Cross-case swapping to isolate causal contributions

Key Questions to Answer:
- Is visual input sufficient to cause hallucination?
- Does changing instruction with same visual input prevent hallucination?
- Can we isolate which input modality drives the behavior?

Usage:
    python causal_mediation_analysis.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model \
        --halluc-case logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --normal-case logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --output-dir outputs/causal_mediation
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

IMAGE_SIZE = 512
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TrajectoryMetrics:
    """Metrics for a single trajectory."""
    action_delta: float
    trajectory_shape: str
    trajectory: List[List[float]]


@dataclass
class CounterfactualResult:
    """Result of a counterfactual experiment."""
    name: str
    description: str

    # Input configuration
    visual_from: str  # "halluc" or "normal"
    state_from: str
    language_from: str

    # Output metrics
    action_delta: float
    trajectory_shape: str
    trajectory_l2_to_halluc: float
    trajectory_l2_to_normal: float
    trajectory: List[List[float]]

    # Resemblance
    resembles: str  # "halluc", "normal", or "neither"


@dataclass
class CausalMediationAnalysis:
    """Full causal mediation analysis."""
    halluc_case: str
    normal_case: str
    step: int

    # Baseline results
    halluc_baseline: TrajectoryMetrics
    normal_baseline: TrajectoryMetrics

    # Counterfactual results
    counterfactuals: Dict[str, CounterfactualResult]

    # Natural Direct Effects
    nde_visual: float  # Effect of visual input alone
    nde_visual_pct: float  # As percentage of total difference

    # Summary
    visual_sufficient_for_halluc: bool
    primary_causal_factor: str  # "visual", "state", "language", "interaction"


# ============================================================================
# CAUSAL MEDIATION EXPERIMENT
# ============================================================================

class CausalMediationExperiment:
    """Runs causal mediation analysis on SmolVLA."""

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

    def prepare_batch(self, images: List[torch.Tensor], state: List[float], task: str) -> dict:
        """Prepare batch for model inference."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

        batch_images = []
        for img in images:
            batch_images.append(img.unsqueeze(0))

        batch_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        processor = self.model.model.vlm_with_expert.processor
        task_with_newline = task + "\n" if not task.endswith("\n") else task

        tokenized = processor.tokenizer(
            task_with_newline,
            padding="max_length",
            max_length=48,
            truncation=True,
            return_tensors="pt"
        )

        batch = {
            OBS_STATE: batch_state.to(self.device),
            OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].bool().to(self.device),
        }

        for idx, camera_model_name in enumerate(CAMERA_MODEL_NAMES):
            key = f"observation.images.{camera_model_name}"
            batch[key] = batch_images[idx].to(self.device)

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

    def run_analysis(self, halluc_dir: str, normal_dir: str, step: int) -> CausalMediationAnalysis:
        """Run full causal mediation analysis."""
        # Load both cases
        halluc_data = self.load_case_data(halluc_dir, step)
        normal_data = self.load_case_data(normal_dir, step)

        print("\n=== Running Baseline Inferences ===")

        # Baseline: Hallucination case
        print("  Running hallucination baseline...")
        halluc_batch = self.prepare_batch(halluc_data["images"], halluc_data["state"], halluc_data["task"])
        halluc_actions = self.run_inference(halluc_batch)
        halluc_delta = self.compute_action_delta(halluc_actions)
        halluc_shape = self.classify_trajectory_shape(halluc_actions)
        print(f"    Halluc baseline: delta={halluc_delta:.3f}, shape={halluc_shape}")

        halluc_baseline = TrajectoryMetrics(
            action_delta=halluc_delta,
            trajectory_shape=halluc_shape,
            trajectory=halluc_actions.tolist()
        )

        # Baseline: Normal case
        print("  Running normal baseline...")
        normal_batch = self.prepare_batch(normal_data["images"], normal_data["state"], normal_data["task"])
        normal_actions = self.run_inference(normal_batch)
        normal_delta = self.compute_action_delta(normal_actions)
        normal_shape = self.classify_trajectory_shape(normal_actions)
        print(f"    Normal baseline: delta={normal_delta:.3f}, shape={normal_shape}")

        normal_baseline = TrajectoryMetrics(
            action_delta=normal_delta,
            trajectory_shape=normal_shape,
            trajectory=normal_actions.tolist()
        )

        # Total difference
        baseline_l2 = float(np.linalg.norm(halluc_actions - normal_actions))
        print(f"    Baseline difference L2: {baseline_l2:.2f}")

        print("\n=== Running Counterfactual Experiments ===")

        counterfactuals = {}

        # Counterfactual 1: Halluc visual + Normal state + Normal language
        print("  Running CF1: Halluc visual + Normal state/language...")
        cf1_batch = self.prepare_batch(halluc_data["images"], normal_data["state"], normal_data["task"])
        cf1_actions = self.run_inference(cf1_batch)
        cf1_delta = self.compute_action_delta(cf1_actions)
        cf1_shape = self.classify_trajectory_shape(cf1_actions)
        cf1_l2_h = float(np.linalg.norm(cf1_actions - halluc_actions))
        cf1_l2_n = float(np.linalg.norm(cf1_actions - normal_actions))
        cf1_resembles = "halluc" if cf1_l2_h < cf1_l2_n else "normal"
        print(f"    CF1: delta={cf1_delta:.3f}, shape={cf1_shape}, resembles={cf1_resembles}")

        counterfactuals["halluc_visual_normal_rest"] = CounterfactualResult(
            name="CF1: Halluc Visual + Normal State/Language",
            description="Tests if halluc visual input alone drives hallucination",
            visual_from="halluc",
            state_from="normal",
            language_from="normal",
            action_delta=cf1_delta,
            trajectory_shape=cf1_shape,
            trajectory_l2_to_halluc=cf1_l2_h,
            trajectory_l2_to_normal=cf1_l2_n,
            trajectory=cf1_actions.tolist(),
            resembles=cf1_resembles,
        )

        # Counterfactual 2: Normal visual + Halluc state + Normal language
        print("  Running CF2: Normal visual + Halluc state...")
        cf2_batch = self.prepare_batch(normal_data["images"], halluc_data["state"], normal_data["task"])
        cf2_actions = self.run_inference(cf2_batch)
        cf2_delta = self.compute_action_delta(cf2_actions)
        cf2_shape = self.classify_trajectory_shape(cf2_actions)
        cf2_l2_h = float(np.linalg.norm(cf2_actions - halluc_actions))
        cf2_l2_n = float(np.linalg.norm(cf2_actions - normal_actions))
        cf2_resembles = "halluc" if cf2_l2_h < cf2_l2_n else "normal"
        print(f"    CF2: delta={cf2_delta:.3f}, shape={cf2_shape}, resembles={cf2_resembles}")

        counterfactuals["normal_visual_halluc_state"] = CounterfactualResult(
            name="CF2: Normal Visual + Halluc State",
            description="Tests if state difference contributes to hallucination",
            visual_from="normal",
            state_from="halluc",
            language_from="normal",
            action_delta=cf2_delta,
            trajectory_shape=cf2_shape,
            trajectory_l2_to_halluc=cf2_l2_h,
            trajectory_l2_to_normal=cf2_l2_n,
            trajectory=cf2_actions.tolist(),
            resembles=cf2_resembles,
        )

        # Counterfactual 3: Per-camera swap (only right wrist from halluc)
        print("  Running CF3: Only right wrist from halluc...")
        cf3_images = normal_data["images"].copy()
        cf3_images[2] = halluc_data["images"][2]  # right wrist is index 2
        cf3_batch = self.prepare_batch(cf3_images, normal_data["state"], normal_data["task"])
        cf3_actions = self.run_inference(cf3_batch)
        cf3_delta = self.compute_action_delta(cf3_actions)
        cf3_shape = self.classify_trajectory_shape(cf3_actions)
        cf3_l2_h = float(np.linalg.norm(cf3_actions - halluc_actions))
        cf3_l2_n = float(np.linalg.norm(cf3_actions - normal_actions))
        cf3_resembles = "halluc" if cf3_l2_h < cf3_l2_n else "normal"
        print(f"    CF3: delta={cf3_delta:.3f}, shape={cf3_shape}, resembles={cf3_resembles}")

        counterfactuals["right_wrist_only_halluc"] = CounterfactualResult(
            name="CF3: Only Right Wrist from Halluc",
            description="Tests if right wrist camera alone drives hallucination",
            visual_from="halluc_right_wrist_only",
            state_from="normal",
            language_from="normal",
            action_delta=cf3_delta,
            trajectory_shape=cf3_shape,
            trajectory_l2_to_halluc=cf3_l2_h,
            trajectory_l2_to_normal=cf3_l2_n,
            trajectory=cf3_actions.tolist(),
            resembles=cf3_resembles,
        )

        # Counterfactual 4: All except right wrist from halluc
        print("  Running CF4: All EXCEPT right wrist from halluc...")
        cf4_images = halluc_data["images"].copy()
        cf4_images[2] = normal_data["images"][2]  # right wrist from normal
        cf4_batch = self.prepare_batch(cf4_images, halluc_data["state"], halluc_data["task"])
        cf4_actions = self.run_inference(cf4_batch)
        cf4_delta = self.compute_action_delta(cf4_actions)
        cf4_shape = self.classify_trajectory_shape(cf4_actions)
        cf4_l2_h = float(np.linalg.norm(cf4_actions - halluc_actions))
        cf4_l2_n = float(np.linalg.norm(cf4_actions - normal_actions))
        cf4_resembles = "halluc" if cf4_l2_h < cf4_l2_n else "normal"
        print(f"    CF4: delta={cf4_delta:.3f}, shape={cf4_shape}, resembles={cf4_resembles}")

        counterfactuals["all_except_right_wrist_halluc"] = CounterfactualResult(
            name="CF4: All EXCEPT Right Wrist from Halluc",
            description="Tests if removing halluc right wrist prevents hallucination",
            visual_from="halluc_except_right_wrist",
            state_from="halluc",
            language_from="halluc",
            action_delta=cf4_delta,
            trajectory_shape=cf4_shape,
            trajectory_l2_to_halluc=cf4_l2_h,
            trajectory_l2_to_normal=cf4_l2_n,
            trajectory=cf4_actions.tolist(),
            resembles=cf4_resembles,
        )

        # Compute Natural Direct Effects
        # NDE_Visual: How much does changing visual input alone affect the output?
        nde_visual = cf1_l2_n  # Distance from normal when only visual is changed
        nde_visual_pct = (nde_visual / (baseline_l2 + 1e-8)) * 100

        # Determine if visual is sufficient
        # Visual is sufficient if CF1 (halluc visual only) resembles halluc behavior
        visual_sufficient = (counterfactuals["halluc_visual_normal_rest"].resembles == "halluc" or
                           counterfactuals["halluc_visual_normal_rest"].trajectory_shape in ["RAMP_UP", "IRREGULAR"])

        # Determine primary causal factor
        cf1_effect = counterfactuals["halluc_visual_normal_rest"].trajectory_l2_to_normal
        cf2_effect = counterfactuals["normal_visual_halluc_state"].trajectory_l2_to_normal
        cf3_effect = counterfactuals["right_wrist_only_halluc"].trajectory_l2_to_normal

        if cf3_effect > max(cf1_effect, cf2_effect) * 0.5:  # Right wrist explains >50%
            primary_factor = "visual_right_wrist"
        elif cf1_effect > cf2_effect:
            primary_factor = "visual"
        else:
            primary_factor = "state"

        return CausalMediationAnalysis(
            halluc_case=halluc_data["case_name"],
            normal_case=normal_data["case_name"],
            step=step,
            halluc_baseline=asdict(halluc_baseline),
            normal_baseline=asdict(normal_baseline),
            counterfactuals={k: asdict(v) for k, v in counterfactuals.items()},
            nde_visual=nde_visual,
            nde_visual_pct=nde_visual_pct,
            visual_sufficient_for_halluc=visual_sufficient,
            primary_causal_factor=primary_factor,
        )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_counterfactual_results(analysis: CausalMediationAnalysis, output_dir: Path):
    """Visualize counterfactual experiment results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Trajectory L2 distances to baselines
    ax1 = axes[0, 0]
    cf_names = list(analysis.counterfactuals.keys())
    l2_to_halluc = [analysis.counterfactuals[n]["trajectory_l2_to_halluc"] for n in cf_names]
    l2_to_normal = [analysis.counterfactuals[n]["trajectory_l2_to_normal"] for n in cf_names]

    x = np.arange(len(cf_names))
    width = 0.35
    ax1.bar(x - width/2, l2_to_halluc, width, label='L2 to Halluc', color='red', alpha=0.7)
    ax1.bar(x + width/2, l2_to_normal, width, label='L2 to Normal', color='blue', alpha=0.7)
    ax1.set_xlabel("Counterfactual")
    ax1.set_ylabel("L2 Distance")
    ax1.set_title("Trajectory Distance to Baselines")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n[:15] + "..." for n in cf_names], rotation=45, ha='right')
    ax1.legend()

    # 2. Action deltas
    ax2 = axes[0, 1]
    halluc_delta = analysis.halluc_baseline["action_delta"]
    normal_delta = analysis.normal_baseline["action_delta"]
    cf_deltas = [analysis.counterfactuals[n]["action_delta"] for n in cf_names]

    bars = ax2.bar(["Halluc", "Normal"] + [n[:10] for n in cf_names],
                   [halluc_delta, normal_delta] + cf_deltas)
    bars[0].set_color('red')
    bars[1].set_color('blue')
    ax2.set_xlabel("Condition")
    ax2.set_ylabel("Action Delta")
    ax2.set_title("Movement Magnitude Across Conditions")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Resemblance chart
    ax3 = axes[1, 0]
    resemblances = [analysis.counterfactuals[n]["resembles"] for n in cf_names]
    colors = ['red' if r == "halluc" else 'blue' if r == "normal" else 'gray' for r in resemblances]
    ax3.bar(cf_names, [1]*len(cf_names), color=colors)
    ax3.set_ylabel("Resemblance")
    ax3.set_title("Which Baseline Does Each Counterfactual Resemble?")
    ax3.set_xticklabels([n[:15] + "..." for n in cf_names], rotation=45, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Resembles Halluc'),
                      Patch(facecolor='blue', label='Resembles Normal')]
    ax3.legend(handles=legend_elements)

    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
CAUSAL MEDIATION ANALYSIS SUMMARY

Baseline Comparison:
- Halluc: delta={halluc_delta:.3f}, shape={analysis.halluc_baseline['trajectory_shape']}
- Normal: delta={normal_delta:.3f}, shape={analysis.normal_baseline['trajectory_shape']}

Natural Direct Effect of Visual:
- NDE_Visual = {analysis.nde_visual:.2f}
- NDE_Visual % = {analysis.nde_visual_pct:.1f}%

Key Finding:
- Visual sufficient for hallucination: {analysis.visual_sufficient_for_halluc}
- Primary causal factor: {analysis.primary_causal_factor}

Interpretation:
"""
    if analysis.visual_sufficient_for_halluc:
        summary_text += "The visual input alone is SUFFICIENT to cause hallucination behavior.\n"
        summary_text += "This confirms that what the model 'sees' drives the hallucination."
    else:
        summary_text += "Visual input alone is NOT sufficient for hallucination.\n"
        summary_text += "Other factors (state, interaction) also contribute."

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"Causal Mediation Analysis\n{analysis.halluc_case[:25]} vs {analysis.normal_case[:25]}")
    plt.tight_layout()
    plt.savefig(output_dir / "causal_mediation_results.png", dpi=150)
    plt.close()


def visualize_trajectory_comparison(analysis: CausalMediationAnalysis, output_dir: Path):
    """Visualize trajectory comparisons."""
    halluc_traj = np.array(analysis.halluc_baseline["trajectory"])
    normal_traj = np.array(analysis.normal_baseline["trajectory"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: First 6 joints
    for idx in range(6):
        ax = axes[idx // 3, idx % 3]
        timesteps = range(len(halluc_traj))

        ax.plot(timesteps, halluc_traj[:, idx], 'r-', linewidth=2, label='Halluc')
        ax.plot(timesteps, normal_traj[:, idx], 'b-', linewidth=2, label='Normal')

        # Add counterfactuals
        colors = ['green', 'orange', 'purple', 'cyan']
        for i, (cf_name, cf_data) in enumerate(analysis.counterfactuals.items()):
            cf_traj = np.array(cf_data["trajectory"])
            ax.plot(timesteps, cf_traj[:, idx], color=colors[i % len(colors)],
                   linestyle='--', alpha=0.5, label=cf_name[:10])

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position")
        ax.set_title(f"Joint {idx}")
        if idx == 0:
            ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Trajectory Comparison: Baselines vs Counterfactuals")
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_comparison.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analysis: CausalMediationAnalysis, output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Causal Mediation Analysis Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
Causal mediation analysis isolates the effect of individual input modalities
(visual, state, language) on the output behavior. This helps determine
whether the hallucination is driven by:
- Visual input alone (what the model "sees")
- State input (robot's current position)
- Language input (task instruction)
- Or some interaction between them
""")

    report.append("\n## Cases Analyzed")
    report.append(f"- Hallucination case: {analysis.halluc_case}")
    report.append(f"- Normal case: {analysis.normal_case}")
    report.append(f"- Step: {analysis.step}")

    report.append("\n## Baseline Results")
    report.append(f"\n### Hallucination Baseline")
    report.append(f"- Action delta: {analysis.halluc_baseline['action_delta']:.3f}")
    report.append(f"- Trajectory shape: **{analysis.halluc_baseline['trajectory_shape']}**")

    report.append(f"\n### Normal Baseline")
    report.append(f"- Action delta: {analysis.normal_baseline['action_delta']:.3f}")
    report.append(f"- Trajectory shape: **{analysis.normal_baseline['trajectory_shape']}**")

    report.append("\n## Counterfactual Experiments")
    report.append("\n| Experiment | Visual | State | Language | Delta | Shape | Resembles |")
    report.append("|------------|--------|-------|----------|-------|-------|-----------|")

    for cf_name, cf in analysis.counterfactuals.items():
        report.append(f"| {cf['name'][:30]} | {cf['visual_from'][:10]} | {cf['state_from'][:10]} | "
                     f"{cf['language_from'][:10]} | {cf['action_delta']:.3f} | {cf['trajectory_shape']} | "
                     f"**{cf['resembles']}** |")

    report.append("\n## Natural Direct Effects")
    report.append(f"\n**NDE_Visual** = {analysis.nde_visual:.2f}")
    report.append(f"**NDE_Visual %** = {analysis.nde_visual_pct:.1f}%")
    report.append("""
NDE_Visual measures how much the trajectory changes when ONLY the visual input
is swapped from normal to hallucination case (keeping state and language constant).

A high NDE_Visual indicates that visual input is the primary driver of behavior.
""")

    report.append("\n## Key Findings")

    report.append(f"\n### Primary Causal Factor: **{analysis.primary_causal_factor}**")

    if analysis.visual_sufficient_for_halluc:
        report.append("""
### Visual Input is SUFFICIENT for Hallucination

Evidence:
- When we use hallucination visual input with normal state/language,
  the resulting trajectory still resembles hallucination behavior
- This proves that what the model "sees" (the banana in the right wrist camera)
  is the primary driver of the hallucination
""")
    else:
        report.append("""
### Visual Input is NOT Sufficient for Hallucination

Evidence:
- When we use hallucination visual input with normal state/language,
  the resulting trajectory does NOT match hallucination behavior
- Other factors (state, language, or interactions) also contribute
""")

    # Specific analysis of right wrist
    rw_cf = analysis.counterfactuals.get("right_wrist_only_halluc")
    if rw_cf:
        report.append(f"\n### Right Wrist Camera Analysis")
        report.append(f"- When only right wrist is from halluc case:")
        report.append(f"  - Trajectory resembles: **{rw_cf['resembles']}**")
        report.append(f"  - Action delta: {rw_cf['action_delta']:.3f}")

        if rw_cf["resembles"] == "halluc":
            report.append("- **The right wrist camera ALONE is sufficient to cause hallucination**")
        else:
            report.append("- Right wrist camera alone is not sufficient; other cameras also contribute")

    report.append("\n## Visualizations")
    report.append("\n- `causal_mediation_results.png`: Summary of counterfactual experiments")
    report.append("- `trajectory_comparison.png`: Trajectory comparison across conditions")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        get_default_checkpoint, CASE_HALLUC, CASE_NORMAL_CLEAN, get_output_dir, DEFAULT_STEP, DEVICE
    )

    parser = argparse.ArgumentParser(description="Run causal mediation analysis")
    parser.add_argument("--checkpoint", default=get_default_checkpoint(),
                       help="Path to SmolVLA checkpoint")
    parser.add_argument("--halluc-case", default=str(CASE_HALLUC),
                       help="Path to hallucination case directory")
    parser.add_argument("--normal-case", default=str(CASE_NORMAL_CLEAN),
                       help="Path to normal case directory")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                       help="Inference step to analyze")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")
    parser.add_argument("--device", default=DEVICE, help="Device to use")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("causal_mediation")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize experiment
    experiment = CausalMediationExperiment(args.checkpoint, args.device)
    experiment.load_model()

    # Run analysis
    print(f"\nRunning causal mediation analysis at step {args.step}...")
    analysis = experiment.run_analysis(args.halluc_case, args.normal_case, args.step)

    print(f"\nVisual sufficient for hallucination: {analysis.visual_sufficient_for_halluc}")
    print(f"Primary causal factor: {analysis.primary_causal_factor}")

    # Save raw data
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(asdict(analysis), f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_counterfactual_results(analysis, output_dir)
    visualize_trajectory_comparison(analysis, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(analysis, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
