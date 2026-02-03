#!/usr/bin/env python3
"""
Attention Knockout Experiment for SmolVLA Hallucination Investigation.

This tool tests CAUSAL importance of attention to specific token regions
by blocking attention to those regions and measuring the change in action output.

Key insight: Our previous attention analysis shows correlation (what model attends to)
but not causation (whether that attention actually drives the action). This tool
provides causal evidence.

Method:
1. Run normal forward pass, record action output
2. Modify attention mask to block target tokens (e.g., right wrist camera)
3. Run forward pass again, record new action output
4. Causal effect = difference in action outputs

Key Questions to Answer:
- If we knock out right wrist camera attention, does hallucination disappear?
- Which token regions have highest causal effect on movement?

Usage:
    python attention_knockout.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/080000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
        --step 200 \
        --output-dir outputs/attention_knockout
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
    "state": (STATE_INDEX, STATE_INDEX + 1),
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class KnockoutResult:
    """Result of a single knockout experiment."""
    region_name: str
    knocked_out_tokens: Tuple[int, int]

    # Normal output
    normal_action_delta: float  # Mean velocity/movement magnitude
    normal_trajectory_shape: str  # FLAT, RAMP_UP, RAMP_DOWN, etc.

    # Knockout output
    knockout_action_delta: float
    knockout_trajectory_shape: str

    # Causal effect
    action_delta_change: float  # knockout - normal
    action_delta_change_pct: float  # percentage change
    trajectory_l2_distance: float  # L2 distance between normal and knockout trajectories
    cosine_similarity: float  # similarity between trajectories

    # Is the effect significant?
    is_significant: bool


@dataclass
class CaseKnockoutAnalysis:
    """Full knockout analysis for a case."""
    case_name: str
    step: int

    # Normal baseline
    normal_action_delta: float
    normal_trajectory_shape: str
    normal_trajectory: List[List[float]]  # 50x32 actions

    # Per-region knockouts
    region_results: Dict[str, KnockoutResult]

    # Summary
    most_causal_region: str
    most_causal_effect: float
    regions_by_effect: List[str]


# ============================================================================
# KNOCKOUT EXPERIMENT
# ============================================================================

class AttentionKnockoutExperiment:
    """Runs attention knockout experiments on SmolVLA."""

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

    def prepare_batch(self, case_data: dict) -> dict:
        """Prepare batch for model inference."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

        images = []
        for img in case_data["images"]:
            images.append(img.unsqueeze(0))

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

    def classify_trajectory_shape(self, actions: np.ndarray) -> str:
        """Classify trajectory shape based on action deltas."""
        # Compute per-timestep velocity magnitude
        velocities = np.linalg.norm(np.diff(actions, axis=0), axis=1)

        # Analyze trend
        first_half_vel = np.mean(velocities[:len(velocities)//2])
        second_half_vel = np.mean(velocities[len(velocities)//2:])
        overall_vel = np.mean(velocities)

        # Classify
        if overall_vel < 0.5:
            return "FLAT"
        elif second_half_vel > first_half_vel * 1.5:
            return "RAMP_UP"
        elif first_half_vel > second_half_vel * 1.5:
            return "RAMP_DOWN"
        else:
            return "IRREGULAR"

    def compute_action_delta(self, actions: np.ndarray) -> float:
        """Compute mean action delta (movement magnitude)."""
        deltas = np.diff(actions, axis=0)
        return float(np.mean(np.linalg.norm(deltas, axis=1)))

    def run_normal_inference(self, batch: dict) -> Tuple[np.ndarray, dict]:
        """Run normal inference without knockout."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        with torch.no_grad():
            # Prepare inputs
            images, img_masks = self.model.prepare_images(batch)
            state = self.model.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]

            # Get prefix embeddings
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Fill KV cache
            _, past_key_values = self.model.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.config.use_cache,
                fill_kv_cache=True,
            )

            # Run denoising loop manually
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

            actions = x_t  # Final denoised actions

            # Convert to numpy
            actions_np = actions[0].float().cpu().numpy()  # [50, 32]

        return actions_np, {"past_key_values": past_key_values, "prefix_pad_masks": prefix_pad_masks}

    def run_knockout_inference(self, batch: dict, knockout_region: Tuple[int, int]) -> np.ndarray:
        """Run inference with attention to specified region knocked out.

        This works by modifying the KV cache to zero out the keys/values for
        the knocked-out region, preventing the action expert from attending to them.
        """
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        with torch.no_grad():
            # Prepare inputs
            images, img_masks = self.model.prepare_images(batch)
            state = self.model.prepare_state(batch)
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]

            # Get prefix embeddings
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            # Fill KV cache
            _, past_key_values = self.model.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.model.config.use_cache,
                fill_kv_cache=True,
            )

            # KNOCKOUT: Zero out the KV cache for the specified region
            knockout_start, knockout_end = knockout_region
            for layer_idx in range(len(past_key_values)):
                # past_key_values[layer_idx] is dict with "key_states" and "value_states"
                # Shape: [batch, seq_len, num_kv_heads, head_dim]
                past_key_values[layer_idx]["key_states"][:, knockout_start:knockout_end, :, :] = 0.0
                past_key_values[layer_idx]["value_states"][:, knockout_start:knockout_end, :, :] = 0.0

            # Run denoising with knocked-out KV cache
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

            actions = x_t  # Final denoised actions

            actions_np = actions[0].float().cpu().numpy()

        return actions_np

    def run_experiment(self, case_dir: str, step: int) -> CaseKnockoutAnalysis:
        """Run full knockout experiment for a case."""
        case_data = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(case_data)

        # Run normal inference
        print(f"  Running normal inference...")
        normal_actions, _ = self.run_normal_inference(batch)
        normal_delta = self.compute_action_delta(normal_actions)
        normal_shape = self.classify_trajectory_shape(normal_actions)

        print(f"    Normal: delta={normal_delta:.3f}, shape={normal_shape}")

        # Run knockouts for each region
        region_results = {}
        effects = {}

        for region_name, (start, end) in TOKEN_REGIONS.items():
            print(f"  Running knockout for {region_name} [{start}:{end}]...")

            knockout_actions = self.run_knockout_inference(batch, (start, end))
            knockout_delta = self.compute_action_delta(knockout_actions)
            knockout_shape = self.classify_trajectory_shape(knockout_actions)

            # Compute causal effect
            delta_change = knockout_delta - normal_delta
            delta_change_pct = (delta_change / (normal_delta + 1e-8)) * 100

            # Trajectory comparison
            traj_l2 = float(np.linalg.norm(knockout_actions - normal_actions))

            # Cosine similarity
            normal_flat = normal_actions.flatten()
            knockout_flat = knockout_actions.flatten()
            cosine_sim = float(np.dot(normal_flat, knockout_flat) /
                              (np.linalg.norm(normal_flat) * np.linalg.norm(knockout_flat) + 1e-8))

            # Significance threshold: >10% change in action delta
            is_significant = abs(delta_change_pct) > 10

            result = KnockoutResult(
                region_name=region_name,
                knocked_out_tokens=(start, end),
                normal_action_delta=normal_delta,
                normal_trajectory_shape=normal_shape,
                knockout_action_delta=knockout_delta,
                knockout_trajectory_shape=knockout_shape,
                action_delta_change=delta_change,
                action_delta_change_pct=delta_change_pct,
                trajectory_l2_distance=traj_l2,
                cosine_similarity=cosine_sim,
                is_significant=is_significant,
            )

            region_results[region_name] = result
            effects[region_name] = abs(delta_change)

            print(f"    Knockout: delta={knockout_delta:.3f}, shape={knockout_shape}, "
                  f"change={delta_change_pct:.1f}%, L2={traj_l2:.2f}")

        # Rank regions by causal effect
        regions_by_effect = sorted(effects.keys(), key=lambda k: effects[k], reverse=True)
        most_causal = regions_by_effect[0]

        return CaseKnockoutAnalysis(
            case_name=case_data["case_name"],
            step=step,
            normal_action_delta=normal_delta,
            normal_trajectory_shape=normal_shape,
            normal_trajectory=normal_actions.tolist(),
            region_results={k: asdict(v) for k, v in region_results.items()},
            most_causal_region=most_causal,
            most_causal_effect=effects[most_causal],
            regions_by_effect=regions_by_effect,
        )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_knockout_effects(analysis: CaseKnockoutAnalysis, output_dir: Path):
    """Visualize the causal effects of knockouts."""
    regions = list(TOKEN_REGIONS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Action delta change by region
    ax1 = axes[0, 0]
    deltas = [analysis.region_results[r]["action_delta_change"] for r in regions]
    colors = ['red' if d > 0 else 'blue' for d in deltas]
    bars = ax1.bar(regions, deltas, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel("Token Region")
    ax1.set_ylabel("Action Delta Change")
    ax1.set_title("Causal Effect: Change in Movement When Region Knocked Out")
    ax1.tick_params(axis='x', rotation=45)

    # Add significance markers
    for i, (r, d) in enumerate(zip(regions, deltas)):
        if analysis.region_results[r]["is_significant"]:
            ax1.annotate('*', (i, d), ha='center', fontsize=16, color='red')

    # 2. Trajectory L2 distance by region
    ax2 = axes[0, 1]
    l2_distances = [analysis.region_results[r]["trajectory_l2_distance"] for r in regions]
    ax2.bar(regions, l2_distances, color='purple', alpha=0.7)
    ax2.set_xlabel("Token Region")
    ax2.set_ylabel("Trajectory L2 Distance")
    ax2.set_title("How Much Does Trajectory Change When Region Knocked Out?")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Cosine similarity by region
    ax3 = axes[1, 0]
    cosines = [analysis.region_results[r]["cosine_similarity"] for r in regions]
    ax3.bar(regions, cosines, color='green', alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=0.5, label='Perfect similarity')
    ax3.set_xlabel("Token Region")
    ax3.set_ylabel("Cosine Similarity")
    ax3.set_title("Trajectory Similarity After Knockout (1.0 = no change)")
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1.1)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create table data
    table_data = [
        ["Region", "Knockout Δ", "% Change", "Significant?"]
    ]
    for r in analysis.regions_by_effect:
        result = analysis.region_results[r]
        sig = "YES" if result["is_significant"] else "no"
        table_data.append([
            r,
            f"{result['action_delta_change']:.3f}",
            f"{result['action_delta_change_pct']:.1f}%",
            sig
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
    ax4.set_title(f"Regions Ranked by Causal Effect\nMost Causal: {analysis.most_causal_region}")

    plt.suptitle(f"Attention Knockout Analysis: {analysis.case_name}\n"
                 f"Step {analysis.step}, Normal shape: {analysis.normal_trajectory_shape}")
    plt.tight_layout()
    plt.savefig(output_dir / f"knockout_effects_{analysis.case_name[:30]}.png", dpi=150)
    plt.close()


def visualize_trajectory_comparison(analysis: CaseKnockoutAnalysis, output_dir: Path):
    """Visualize normal vs knockout trajectories."""
    normal_traj = np.array(analysis.normal_trajectory)  # [50, 32]

    # Select key joints to visualize (first 6 for one arm)
    joint_names = ["Joint 0", "Joint 1", "Joint 2", "Joint 3", "Joint 4", "Joint 5"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    timesteps = range(normal_traj.shape[0])

    for idx, (ax, joint_name) in enumerate(zip(axes, joint_names)):
        ax.plot(timesteps, normal_traj[:, idx], 'k-', linewidth=2, label='Normal')

        # Plot most causal region knockout
        most_causal = analysis.most_causal_region
        # We need to re-run to get knockout trajectories, but for now just show normal

        ax.set_xlabel("Timestep")
        ax.set_ylabel("Position")
        ax.set_title(f"{joint_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory Analysis: {analysis.case_name}\n"
                 f"Shape: {analysis.normal_trajectory_shape}, "
                 f"Most Causal Region: {analysis.most_causal_region}")
    plt.tight_layout()
    plt.savefig(output_dir / f"trajectory_{analysis.case_name[:30]}.png", dpi=150)
    plt.close()


def visualize_comparison_across_cases(analyses: List[CaseKnockoutAnalysis], output_dir: Path):
    """Compare knockout effects across multiple cases."""
    if len(analyses) < 2:
        return

    regions = list(TOKEN_REGIONS.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(regions))
    width = 0.35

    for idx, analysis in enumerate(analyses):
        effects = [abs(analysis.region_results[r]["action_delta_change"]) for r in regions]
        offset = (idx - len(analyses)/2 + 0.5) * width
        bars = ax.bar(x + offset, effects, width, label=analysis.case_name[:25])

    ax.set_xlabel("Token Region")
    ax.set_ylabel("Absolute Action Delta Change")
    ax.set_title("Causal Effect Comparison Across Cases")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "knockout_comparison_across_cases.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(analyses: List[CaseKnockoutAnalysis], output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Attention Knockout Experiment Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")
    report.append(f"\n**Purpose**: Test CAUSAL importance of attention to specific token regions")

    report.append("\n## Method")
    report.append("""
The attention knockout experiment works by:
1. Running normal inference to establish baseline action output
2. Zeroing out the KV cache for a specific token region (e.g., right wrist camera)
3. Running inference with the modified KV cache
4. Measuring the change in action output

If knocking out a region causes large changes in the action, that region has
HIGH CAUSAL IMPORTANCE for the action. If there's little change, the model
doesn't depend on that region for its decision.
""")

    report.append("\n## Cases Analyzed")
    for analysis in analyses:
        report.append(f"\n### {analysis.case_name}")
        report.append(f"- Step: {analysis.step}")
        report.append(f"- Normal trajectory shape: **{analysis.normal_trajectory_shape}**")
        report.append(f"- Normal action delta: {analysis.normal_action_delta:.3f}")
        report.append(f"- Most causal region: **{analysis.most_causal_region}**")

        report.append(f"\n#### Knockout Results")
        report.append("\n| Region | Normal Δ | Knockout Δ | Change | % Change | L2 Dist | Cosine | Significant? |")
        report.append("|--------|----------|------------|--------|----------|---------|--------|--------------|")

        for region in analysis.regions_by_effect:
            r = analysis.region_results[region]
            sig = "**YES**" if r["is_significant"] else "no"
            report.append(f"| {region} | {r['normal_action_delta']:.3f} | "
                         f"{r['knockout_action_delta']:.3f} | {r['action_delta_change']:.3f} | "
                         f"{r['action_delta_change_pct']:.1f}% | {r['trajectory_l2_distance']:.2f} | "
                         f"{r['cosine_similarity']:.3f} | {sig} |")

    report.append("\n## Key Findings")

    # Analyze patterns
    halluc_analyses = [a for a in analyses if "ha_bana" in a.case_name or "halluc" in a.case_name.lower()]
    normal_analyses = [a for a in analyses if "no_ha" in a.case_name]

    if halluc_analyses and normal_analyses:
        report.append("\n### Hallucination vs Normal Cases")

        halluc = halluc_analyses[0]
        normal = normal_analyses[0]

        report.append(f"\n**Hallucination case ({halluc.case_name[:25]}):**")
        report.append(f"- Most causal region: {halluc.most_causal_region}")
        report.append(f"- Right wrist effect: {halluc.region_results['right_wrist']['action_delta_change_pct']:.1f}%")

        report.append(f"\n**Normal case ({normal.case_name[:25]}):**")
        report.append(f"- Most causal region: {normal.most_causal_region}")
        report.append(f"- Right wrist effect: {normal.region_results['right_wrist']['action_delta_change_pct']:.1f}%")

        # Key question: does knocking out right wrist in halluc case reduce movement?
        rw_effect = halluc.region_results['right_wrist']['action_delta_change']
        if rw_effect < 0:
            report.append(f"\n**CRITICAL FINDING**: Knocking out right wrist camera attention **REDUCES** "
                         f"action delta by {abs(rw_effect):.3f} in hallucination case. This suggests "
                         f"the right wrist camera is causally driving the hallucination behavior.")
        else:
            report.append(f"\n**Finding**: Knocking out right wrist camera attention changes "
                         f"action delta by {rw_effect:.3f} in hallucination case.")

    report.append("\n## Interpretation Guide")
    report.append("""
- **Negative delta change**: Knocking out this region REDUCES movement → region drives motion
- **Positive delta change**: Knocking out this region INCREASES movement → region inhibits motion
- **High L2 distance**: Trajectory changes significantly when region knocked out
- **Low cosine similarity**: Trajectory direction changes when region knocked out

If the hallucination is caused by the banana in the right wrist camera, we expect:
- Knocking out right wrist → significant reduction in movement (negative delta change)
- High trajectory L2 distance for right wrist knockout
""")

    report.append("\n## Visualizations")
    report.append("\n- `knockout_effects_*.png`: Per-region causal effects")
    report.append("- `trajectory_*.png`: Trajectory shape analysis")
    report.append("- `knockout_comparison_across_cases.png`: Cross-case comparison")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        get_default_checkpoint, get_all_case_dirs, get_output_dir, DEFAULT_STEP, DEVICE
    )

    parser = argparse.ArgumentParser(description="Run attention knockout experiments")
    parser.add_argument("--checkpoint", default=get_default_checkpoint(),
                       help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dirs", nargs="+", default=get_all_case_dirs(),
                       help="Paths to case directories")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                       help="Inference step to analyze")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")
    parser.add_argument("--device", default=DEVICE, help="Device to use")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("attention_knockout")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize experiment
    experiment = AttentionKnockoutExperiment(args.checkpoint, args.device)
    experiment.load_model()

    # Run experiments
    print(f"\nRunning attention knockout experiments at step {args.step}...")
    all_analyses = []

    for case_dir in args.case_dirs:
        print(f"\nProcessing: {case_dir}")
        analysis = experiment.run_experiment(case_dir, args.step)
        all_analyses.append(analysis)

        print(f"  Most causal region: {analysis.most_causal_region}")
        print(f"  Most causal effect: {analysis.most_causal_effect:.3f}")

    # Save raw data
    analyses_data = [asdict(a) for a in all_analyses]
    with open(output_dir / "analyses.json", "w") as f:
        json.dump(analyses_data, f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    for analysis in all_analyses:
        visualize_knockout_effects(analysis, output_dir)
        visualize_trajectory_comparison(analysis, output_dir)

    visualize_comparison_across_cases(all_analyses, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_analyses, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
