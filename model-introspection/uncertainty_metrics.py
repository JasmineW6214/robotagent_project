#!/usr/bin/env python3
"""
Uncertainty Quantification Metrics for SmolVLA Hallucination Investigation.

Based on "Evaluating Uncertainty and Quality of VLA" (arXiv:2507.17049), this tool
computes various uncertainty metrics to quantify model confidence during
hallucination vs normal cases.

Metrics:
- Action Position Instability (A-PI): Variance across multiple inference runs
- Trajectory Variance (TV): Deviation from smooth trajectory
- Denoising Entropy: Uncertainty in the denoising process
- Visual Grounding Score (VGS): Attention alignment with objects

Key Questions to Answer:
- Is the model more uncertain during hallucination?
- Does high uncertainty correlate with hallucination likelihood?
- Can uncertainty be used as a detection signal?

Usage:
    python uncertainty_metrics.py \
        --checkpoint outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_133142_no_ha_no_other_obj \
        --step 200 \
        --n-samples 10 \
        --output-dir outputs/uncertainty
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
class UncertaintyMetrics:
    """Uncertainty metrics for a single case."""
    case_name: str
    step: int

    # Action Position Instability (A-PI)
    # Variance across multiple inference runs
    api_mean: float  # Mean variance across all joints
    api_per_joint: List[float]  # Per-joint variance
    api_max_joint: int  # Joint with highest variance

    # Trajectory Variance (TV)
    # Deviation from smooth spline fit
    tv_mean: float
    tv_per_timestep: List[float]
    tv_max_timestep: int

    # Trajectory Jerk
    # Third derivative of position (smoothness indicator)
    jerk_mean: float
    jerk_max: float

    # Denoising Consistency
    # How consistent are intermediate denoising steps
    denoising_consistency: float

    # Overall uncertainty score (normalized 0-1)
    uncertainty_score: float

    # Classification
    uncertainty_level: str  # "low", "medium", "high"


@dataclass
class CaseComparison:
    """Comparison of uncertainty between cases."""
    halluc_case: str
    normal_case: str

    halluc_uncertainty: float
    normal_uncertainty: float
    uncertainty_ratio: float  # halluc / normal

    # Which metrics differ most
    most_discriminative_metric: str
    discrimination_power: float

    # Can we detect hallucination from uncertainty?
    halluc_detectable_by_uncertainty: bool


# ============================================================================
# UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyQuantifier:
    """Computes uncertainty metrics for SmolVLA."""

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

    def run_inference_with_noise_seed(self, batch: dict, seed: int) -> np.ndarray:
        """Run inference with a specific noise seed."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        # Set seed for noise generation
        torch.manual_seed(seed)
        np.random.seed(seed)

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

            # Run denoising loop with seeded noise
            bsize = state.shape[0]
            device = state.device
            actions_shape = (bsize, self.model.config.chunk_size, self.model.config.max_action_dim)

            # Generate noise with the specified seed
            torch.manual_seed(seed)
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

    def compute_api(self, trajectories: List[np.ndarray]) -> Tuple[float, List[float], int]:
        """Compute Action Position Instability.

        API measures variance across multiple inference runs with different noise seeds.
        High API indicates the model is uncertain and produces different outputs.
        """
        # Stack trajectories: (n_samples, timesteps, joints)
        stacked = np.stack(trajectories)

        # Compute variance across samples for each timestep and joint
        variance = np.var(stacked, axis=0)  # (timesteps, joints)

        # Per-joint variance (average over timesteps)
        per_joint = np.mean(variance, axis=0).tolist()

        # Mean variance
        mean_var = float(np.mean(variance))

        # Joint with max variance
        max_joint = int(np.argmax(per_joint))

        return mean_var, per_joint, max_joint

    def compute_trajectory_variance(self, trajectory: np.ndarray) -> Tuple[float, List[float], int]:
        """Compute Trajectory Variance.

        Measures deviation from a smooth trajectory by fitting a low-order
        polynomial and computing residuals.
        """
        timesteps = np.arange(len(trajectory))
        residuals = []

        for joint in range(trajectory.shape[1]):
            # Fit 3rd order polynomial
            poly = np.polyfit(timesteps, trajectory[:, joint], 3)
            fitted = np.polyval(poly, timesteps)
            residual = np.abs(trajectory[:, joint] - fitted)
            residuals.append(residual)

        residuals = np.array(residuals).T  # (timesteps, joints)

        # Per-timestep variance (mean across joints)
        per_timestep = np.mean(residuals, axis=1).tolist()

        # Mean variance
        mean_var = float(np.mean(residuals))

        # Timestep with max variance
        max_timestep = int(np.argmax(per_timestep))

        return mean_var, per_timestep, max_timestep

    def compute_jerk(self, trajectory: np.ndarray) -> Tuple[float, float]:
        """Compute trajectory jerk (third derivative).

        High jerk indicates jerky, non-smooth motion which can indicate
        uncertainty or out-of-distribution behavior.
        """
        # First derivative (velocity)
        vel = np.diff(trajectory, axis=0)

        # Second derivative (acceleration)
        acc = np.diff(vel, axis=0)

        # Third derivative (jerk)
        jerk = np.diff(acc, axis=0)

        # Compute norms
        jerk_norms = np.linalg.norm(jerk, axis=1)

        mean_jerk = float(np.mean(jerk_norms))
        max_jerk = float(np.max(jerk_norms))

        return mean_jerk, max_jerk

    def compute_denoising_consistency(self, batch: dict, n_runs: int = 5) -> float:
        """Compute consistency across denoising runs.

        Measures how similar the denoising trajectories are across different
        noise seeds. Low consistency indicates high uncertainty.
        """
        trajectories = []
        for seed in range(n_runs):
            traj = self.run_inference_with_noise_seed(batch, seed * 1000)
            trajectories.append(traj)

        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                flat_i = trajectories[i].flatten()
                flat_j = trajectories[j].flatten()
                sim = np.dot(flat_i, flat_j) / (np.linalg.norm(flat_i) * np.linalg.norm(flat_j) + 1e-8)
                similarities.append(sim)

        return float(np.mean(similarities))

    def compute_uncertainty_metrics(self, case_dir: str, step: int,
                                   n_samples: int = 10) -> UncertaintyMetrics:
        """Compute all uncertainty metrics for a case."""
        case_data = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(case_data)

        print(f"  Running {n_samples} inference samples...")
        trajectories = []
        for i in range(n_samples):
            traj = self.run_inference_with_noise_seed(batch, i * 1000)
            trajectories.append(traj)

        # Use first trajectory as reference
        ref_traj = trajectories[0]

        # Compute metrics
        print("  Computing API...")
        api_mean, api_per_joint, api_max_joint = self.compute_api(trajectories)

        print("  Computing trajectory variance...")
        tv_mean, tv_per_timestep, tv_max_timestep = self.compute_trajectory_variance(ref_traj)

        print("  Computing jerk...")
        jerk_mean, jerk_max = self.compute_jerk(ref_traj)

        print("  Computing denoising consistency...")
        denoising_consistency = self.compute_denoising_consistency(batch, n_runs=5)

        # Compute overall uncertainty score (normalized)
        # Higher API, higher TV, higher jerk, lower consistency = higher uncertainty
        api_normalized = min(api_mean / 0.1, 1.0)  # Normalize by expected max
        tv_normalized = min(tv_mean / 0.05, 1.0)
        jerk_normalized = min(jerk_mean / 0.1, 1.0)
        consistency_inverted = 1.0 - denoising_consistency

        uncertainty_score = (api_normalized + tv_normalized + jerk_normalized + consistency_inverted) / 4

        # Classify uncertainty level
        if uncertainty_score < 0.3:
            uncertainty_level = "low"
        elif uncertainty_score < 0.6:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "high"

        return UncertaintyMetrics(
            case_name=case_data["case_name"],
            step=step,
            api_mean=api_mean,
            api_per_joint=api_per_joint,
            api_max_joint=api_max_joint,
            tv_mean=tv_mean,
            tv_per_timestep=tv_per_timestep,
            tv_max_timestep=tv_max_timestep,
            jerk_mean=jerk_mean,
            jerk_max=jerk_max,
            denoising_consistency=denoising_consistency,
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level,
        )


def compare_cases(halluc_metrics: UncertaintyMetrics, normal_metrics: UncertaintyMetrics) -> CaseComparison:
    """Compare uncertainty metrics between hallucination and normal cases."""
    ratio = halluc_metrics.uncertainty_score / (normal_metrics.uncertainty_score + 1e-8)

    # Find most discriminative metric
    metrics = {
        "api": abs(halluc_metrics.api_mean - normal_metrics.api_mean) / (max(halluc_metrics.api_mean, normal_metrics.api_mean) + 1e-8),
        "tv": abs(halluc_metrics.tv_mean - normal_metrics.tv_mean) / (max(halluc_metrics.tv_mean, normal_metrics.tv_mean) + 1e-8),
        "jerk": abs(halluc_metrics.jerk_mean - normal_metrics.jerk_mean) / (max(halluc_metrics.jerk_mean, normal_metrics.jerk_mean) + 1e-8),
        "consistency": abs(halluc_metrics.denoising_consistency - normal_metrics.denoising_consistency),
    }

    most_discriminative = max(metrics.keys(), key=lambda k: metrics[k])
    discrimination_power = metrics[most_discriminative]

    # Can we detect hallucination by uncertainty?
    # If halluc has significantly higher uncertainty, it's detectable
    detectable = (ratio > 1.5 or halluc_metrics.uncertainty_level == "high" and
                  normal_metrics.uncertainty_level in ["low", "medium"])

    return CaseComparison(
        halluc_case=halluc_metrics.case_name,
        normal_case=normal_metrics.case_name,
        halluc_uncertainty=halluc_metrics.uncertainty_score,
        normal_uncertainty=normal_metrics.uncertainty_score,
        uncertainty_ratio=ratio,
        most_discriminative_metric=most_discriminative,
        discrimination_power=discrimination_power,
        halluc_detectable_by_uncertainty=detectable,
    )


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_uncertainty_metrics(metrics_list: List[UncertaintyMetrics], output_dir: Path):
    """Visualize uncertainty metrics across cases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    case_names = [m.case_name[:20] for m in metrics_list]

    # 1. Overall uncertainty score
    ax1 = axes[0, 0]
    scores = [m.uncertainty_score for m in metrics_list]
    colors = ['red' if m.uncertainty_level == 'high' else 'orange' if m.uncertainty_level == 'medium' else 'green'
              for m in metrics_list]
    ax1.bar(case_names, scores, color=colors)
    ax1.set_ylabel("Uncertainty Score")
    ax1.set_title("Overall Uncertainty Score (0=certain, 1=uncertain)")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    # 2. API comparison
    ax2 = axes[0, 1]
    apis = [m.api_mean for m in metrics_list]
    ax2.bar(case_names, apis, color='blue', alpha=0.7)
    ax2.set_ylabel("API (Action Position Instability)")
    ax2.set_title("Variance Across Multiple Inference Runs")
    ax2.tick_params(axis='x', rotation=45)

    # 3. Denoising consistency
    ax3 = axes[1, 0]
    consistencies = [m.denoising_consistency for m in metrics_list]
    ax3.bar(case_names, consistencies, color='green', alpha=0.7)
    ax3.set_ylabel("Denoising Consistency")
    ax3.set_title("Consistency Across Denoising Runs (1=perfect)")
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Jerk comparison
    ax4 = axes[1, 1]
    jerks = [m.jerk_mean for m in metrics_list]
    ax4.bar(case_names, jerks, color='purple', alpha=0.7)
    ax4.set_ylabel("Mean Jerk")
    ax4.set_title("Trajectory Smoothness (lower=smoother)")
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_comparison.png", dpi=150)
    plt.close()


def visualize_per_joint_api(metrics: UncertaintyMetrics, output_dir: Path):
    """Visualize per-joint API for a single case."""
    fig, ax = plt.subplots(figsize=(12, 6))

    joints = range(len(metrics.api_per_joint))
    ax.bar(joints, metrics.api_per_joint, color='blue', alpha=0.7)
    ax.axhline(y=metrics.api_mean, color='red', linestyle='--', label=f'Mean: {metrics.api_mean:.4f}')
    ax.axvline(x=metrics.api_max_joint, color='green', linestyle='--',
               label=f'Max joint: {metrics.api_max_joint}')

    ax.set_xlabel("Joint Index")
    ax.set_ylabel("Position Variance (API)")
    ax.set_title(f"Per-Joint Action Position Instability: {metrics.case_name[:30]}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f"api_per_joint_{metrics.case_name[:20]}.png", dpi=150)
    plt.close()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(metrics_list: List[UncertaintyMetrics],
                   comparison: Optional[CaseComparison], output_dir: Path):
    """Generate markdown report."""
    report = []
    report.append("# Uncertainty Quantification Report")
    report.append(f"\n**Generated**: {datetime.now().isoformat()}")

    report.append("\n## Purpose")
    report.append("""
Uncertainty quantification helps determine if the model is confident in its
predictions or if it's operating in an uncertain/out-of-distribution regime.

Key metrics:
- **API (Action Position Instability)**: Variance across multiple inference runs
- **TV (Trajectory Variance)**: Deviation from smooth polynomial fit
- **Jerk**: Third derivative of position (smoothness indicator)
- **Denoising Consistency**: Similarity across different noise seeds
""")

    report.append("\n## Metrics Summary")
    report.append("\n| Case | Uncertainty | API | TV | Jerk | Consistency | Level |")
    report.append("|------|-------------|-----|----|----- |-------------|-------|")

    for m in metrics_list:
        report.append(f"| {m.case_name[:25]} | {m.uncertainty_score:.3f} | {m.api_mean:.4f} | "
                     f"{m.tv_mean:.4f} | {m.jerk_mean:.4f} | {m.denoising_consistency:.3f} | "
                     f"**{m.uncertainty_level}** |")

    for m in metrics_list:
        report.append(f"\n### {m.case_name}")
        report.append(f"- Overall uncertainty score: **{m.uncertainty_score:.3f}** ({m.uncertainty_level})")
        report.append(f"- API (mean): {m.api_mean:.4f}")
        report.append(f"  - Max variance joint: {m.api_max_joint}")
        report.append(f"- Trajectory Variance: {m.tv_mean:.4f}")
        report.append(f"  - Max variance timestep: {m.tv_max_timestep}")
        report.append(f"- Jerk: mean={m.jerk_mean:.4f}, max={m.jerk_max:.4f}")
        report.append(f"- Denoising consistency: {m.denoising_consistency:.3f}")

    if comparison:
        report.append("\n## Case Comparison")
        report.append(f"\n**Hallucination case**: {comparison.halluc_case[:30]}")
        report.append(f"**Normal case**: {comparison.normal_case[:30]}")

        report.append(f"\n### Uncertainty Comparison")
        report.append(f"- Halluc uncertainty: {comparison.halluc_uncertainty:.3f}")
        report.append(f"- Normal uncertainty: {comparison.normal_uncertainty:.3f}")
        report.append(f"- Ratio (halluc/normal): **{comparison.uncertainty_ratio:.2f}x**")

        report.append(f"\n### Most Discriminative Metric: **{comparison.most_discriminative_metric}**")
        report.append(f"- Discrimination power: {comparison.discrimination_power:.3f}")

        report.append(f"\n### Hallucination Detectable by Uncertainty: **{comparison.halluc_detectable_by_uncertainty}**")

        if comparison.halluc_detectable_by_uncertainty:
            report.append("""
**Finding**: Hallucination case shows significantly higher uncertainty than normal case.
This suggests uncertainty metrics could be used as a runtime detection signal.
""")
        else:
            report.append("""
**Finding**: Hallucination case does NOT show significantly higher uncertainty.
The model is confident in its (incorrect) prediction, which means uncertainty
alone cannot detect this type of hallucination.
""")

    report.append("\n## Interpretation Guide")
    report.append("""
| Metric | Low Value | High Value |
|--------|-----------|------------|
| Uncertainty Score | Confident prediction | Uncertain, possibly OOD |
| API | Consistent across runs | Varies with noise seed |
| TV | Smooth trajectory | Jerky, non-smooth |
| Jerk | Smooth motion | Jerky, discontinuous |
| Consistency | Different each run | Same output regardless of seed |

**Hallucination Detection**:
- If hallucination cases consistently show higher uncertainty → can use as detection signal
- If hallucination cases show similar uncertainty to normal → model is confidently wrong
""")

    report.append("\n## Visualizations")
    report.append("\n- `uncertainty_comparison.png`: Cross-case metric comparison")
    report.append("- `api_per_joint_*.png`: Per-joint uncertainty for each case")

    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(report))


# ============================================================================
# MAIN
# ============================================================================

def main():
    from investigation_config import (
        get_default_checkpoint, get_all_case_dirs, get_output_dir, DEFAULT_STEP, DEVICE
    )

    parser = argparse.ArgumentParser(description="Compute uncertainty metrics")
    parser.add_argument("--checkpoint", default=get_default_checkpoint(),
                       help="Path to SmolVLA checkpoint")
    parser.add_argument("--case-dirs", nargs="+", default=get_all_case_dirs(),
                       help="Paths to case directories")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP,
                       help="Inference step to analyze")
    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of inference samples for API computation")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory")
    parser.add_argument("--device", default=DEVICE, help="Device to use")

    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = get_output_dir("uncertainty")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize quantifier
    quantifier = UncertaintyQuantifier(args.checkpoint, args.device)
    quantifier.load_model()

    # Compute metrics for each case
    print(f"\nComputing uncertainty metrics at step {args.step} with {args.n_samples} samples...")
    all_metrics = []

    for case_dir in args.case_dirs:
        print(f"\nProcessing: {case_dir}")
        metrics = quantifier.compute_uncertainty_metrics(case_dir, args.step, args.n_samples)
        all_metrics.append(metrics)
        print(f"  Uncertainty: {metrics.uncertainty_score:.3f} ({metrics.uncertainty_level})")

    # Compare halluc vs normal if we have both
    comparison = None
    halluc_metrics = [m for m in all_metrics if "ha_bana" in m.case_name or "halluc" in m.case_name.lower()]
    normal_metrics = [m for m in all_metrics if "no_ha" in m.case_name]

    if halluc_metrics and normal_metrics:
        comparison = compare_cases(halluc_metrics[0], normal_metrics[0])
        print(f"\nComparison: halluc/normal uncertainty ratio = {comparison.uncertainty_ratio:.2f}x")
        print(f"Hallucination detectable by uncertainty: {comparison.halluc_detectable_by_uncertainty}")

    # Save raw data
    metrics_data = [asdict(m) for m in all_metrics]
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    if comparison:
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(asdict(comparison), f, indent=2)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_uncertainty_metrics(all_metrics, output_dir)
    for metrics in all_metrics:
        visualize_per_joint_api(metrics, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(all_metrics, comparison, output_dir)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
