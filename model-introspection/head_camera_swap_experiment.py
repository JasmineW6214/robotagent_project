#!/usr/bin/env python3
"""
Head Camera Swap Experiment

Purpose: Directly test if head camera alone can flip behavior between
hallucination and normal cases.

Method:
- Swap A: Halluc case + Normal's head camera → Does hallucination disappear?
- Swap B: Normal case + Halluc's head camera → Does hallucination appear?

If swapping head camera flips the behavior, this proves head camera is the
primary causal driver of the hallucination.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import cv2
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

# Constants
IMAGE_SIZE = 512
CAMERA_FILE_NAMES = ["head", "left_wrist", "right_wrist"]
CAMERA_MODEL_NAMES = ["camera1", "camera2", "camera3"]


class HeadCameraSwapExperiment:
    """Experiment to test head camera's causal role via swapping."""

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

    def load_case_images(self, case_dir: str, step: int) -> dict:
        """Load images from case directory."""
        case_path = Path(case_dir)
        images = {}

        for camera_file_name in CAMERA_FILE_NAMES:
            # Try different path patterns
            candidates = [
                case_path / f"step_{step:05d}_{camera_file_name}.png",
                case_path / f"step_{step:04d}_{camera_file_name}.png",
                case_path / f"step_{step:05d}_{camera_file_name}.jpg",
                case_path / f"step_{step:04d}_{camera_file_name}.jpg",
                case_path / "images" / f"step_{step:05d}_{camera_file_name}.png",
                case_path / "images" / f"step_{step:04d}_{camera_file_name}.png",
                case_path / "images" / f"step_{step:05d}_{camera_file_name}.jpg",
                case_path / "images" / f"step_{step:04d}_{camera_file_name}.jpg",
            ]

            image_path = None
            for candidate in candidates:
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                raise FileNotFoundError(f"Could not find {camera_file_name} image for step {step} in {case_dir}")

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            images[camera_file_name] = img_tensor

        return images

    def load_case_state(self, case_dir: str, step: int):
        """Load state from trace."""
        case_path = Path(case_dir)
        trace_path = case_path / "trace.jsonl"

        state = None
        state_raw = None
        with open(trace_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["step"] == step:
                    state = data.get("state_normalized", data.get("state"))
                    state_raw = data.get("state_raw", data.get("state"))
                    break

        if state is None:
            raise ValueError(f"Could not find step {step} in trace")

        return state, state_raw

    def load_case_task(self, case_dir: str) -> str:
        """Load task from metadata."""
        case_path = Path(case_dir)
        metadata_path = case_path / "metadata.json"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("task_original", metadata.get("task", ""))

    def prepare_batch(self, images: dict, state: list, task: str) -> dict:
        """Prepare batch for model inference."""
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

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
            OBS_STATE: state_tensor.to(self.device),
            OBS_LANGUAGE_TOKENS: tokenized["input_ids"].to(self.device),
            OBS_LANGUAGE_ATTENTION_MASK: tokenized["attention_mask"].bool().to(self.device),
        }

        # Map camera file names to model names
        for file_name, model_name in zip(CAMERA_FILE_NAMES, CAMERA_MODEL_NAMES):
            key = f"observation.images.{model_name}"
            batch[key] = images[file_name].unsqueeze(0).to(self.device)

        return batch

    def run_inference(self, batch: dict) -> np.ndarray:
        """Run model inference and return action."""
        with torch.no_grad():
            action = self.model.select_action(batch)
        return action.cpu().numpy()

    def compute_metrics(self, action: np.ndarray, state_normalized: list) -> dict:
        """Compute movement metrics from action.

        Note: Model outputs normalized actions, so compare with state_normalized.
        The threshold for FLAT vs RAMP is based on normalized scale:
        - Normalized actions are in [-1, 1] range
        - Delta < 0.05 means stay still (FLAT)
        - Delta > 0.05 means movement (RAMP)
        """
        state_norm = np.array(state_normalized)

        # Action delta (how much robot would move) - in normalized space
        action_first = action[0, :len(state_norm)]
        delta = np.abs(action_first - state_norm)
        delta_max = delta.max()
        delta_mean = delta.mean()

        # Trajectory shape analysis
        trajectory = action[:, :len(state_norm)]
        if trajectory.shape[0] > 1:
            velocities = np.diff(trajectory, axis=0)
            velocity_norms = np.linalg.norm(velocities, axis=1)
            mean_velocity = velocity_norms.mean() if len(velocity_norms) > 0 else 0.0
        else:
            mean_velocity = 0.0

        # Is it FLAT (stay still) or RAMP (movement)?
        # Threshold 0.05 in normalized space ≈ 3° in raw space
        shape = "FLAT" if delta_max < 0.05 else "RAMP"

        return {
            'delta_max': float(delta_max),
            'delta_mean': float(delta_mean),
            'mean_velocity': float(mean_velocity),
            'shape': shape,
        }

    def run_swap_experiment(
        self,
        halluc_dir: str,
        normal_dir: str,
        step: int,
        output_dir: str,
    ):
        """Run the complete swap experiment."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("HEAD CAMERA SWAP EXPERIMENT")
        print(f"{'='*60}")
        print(f"Halluc case: {halluc_dir}")
        print(f"Normal case: {normal_dir}")
        print(f"Step: {step}")

        # Load model
        self.load_model()

        # Load data from both cases
        print("\nLoading case data...")
        halluc_images = self.load_case_images(halluc_dir, step)
        halluc_state, halluc_state_raw = self.load_case_state(halluc_dir, step)
        halluc_task = self.load_case_task(halluc_dir)

        normal_images = self.load_case_images(normal_dir, step)
        normal_state, normal_state_raw = self.load_case_state(normal_dir, step)
        normal_task = self.load_case_task(normal_dir)

        results = {}

        # Baseline: Original halluc case
        print("\n--- Baseline: Original Halluc Case ---")
        batch = self.prepare_batch(halluc_images, halluc_state, halluc_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, halluc_state)
        results['baseline_halluc'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")

        # Baseline: Original normal case
        print("\n--- Baseline: Original Normal Case ---")
        batch = self.prepare_batch(normal_images, normal_state, normal_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, normal_state)
        results['baseline_normal'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")

        # Swap A: Halluc + Normal's head camera
        print("\n--- Swap A: Halluc case + Normal's HEAD camera ---")
        swap_a_images = {
            'head': normal_images['head'],  # SWAPPED
            'left_wrist': halluc_images['left_wrist'],
            'right_wrist': halluc_images['right_wrist'],  # Still has banana
        }
        batch = self.prepare_batch(swap_a_images, halluc_state, halluc_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, halluc_state)
        results['swap_a_halluc_with_normal_head'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")
        print(f"  Question: Does hallucination disappear? → {'YES' if metrics['shape'] == 'FLAT' else 'NO'}")

        # Swap B: Normal + Halluc's head camera
        print("\n--- Swap B: Normal case + Halluc's HEAD camera ---")
        swap_b_images = {
            'head': halluc_images['head'],  # SWAPPED
            'left_wrist': normal_images['left_wrist'],
            'right_wrist': normal_images['right_wrist'],  # No banana
        }
        batch = self.prepare_batch(swap_b_images, normal_state, normal_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, normal_state)
        results['swap_b_normal_with_halluc_head'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")
        print(f"  Question: Does hallucination appear? → {'YES' if metrics['shape'] == 'RAMP' else 'NO'}")

        # Swap C: Halluc + Normal's right wrist (remove banana)
        print("\n--- Swap C: Halluc case + Normal's RIGHT WRIST camera ---")
        swap_c_images = {
            'head': halluc_images['head'],
            'left_wrist': halluc_images['left_wrist'],
            'right_wrist': normal_images['right_wrist'],  # SWAPPED - banana removed
        }
        batch = self.prepare_batch(swap_c_images, halluc_state, halluc_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, halluc_state)
        results['swap_c_halluc_with_normal_rightwrist'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")
        print(f"  Question: Does removing banana fix hallucination? → {'YES' if metrics['shape'] == 'FLAT' else 'NO'}")

        # Swap D: Normal + Halluc's right wrist (add banana)
        print("\n--- Swap D: Normal case + Halluc's RIGHT WRIST camera ---")
        swap_d_images = {
            'head': normal_images['head'],
            'left_wrist': normal_images['left_wrist'],
            'right_wrist': halluc_images['right_wrist'],  # SWAPPED - banana added
        }
        batch = self.prepare_batch(swap_d_images, normal_state, normal_task)
        action = self.run_inference(batch)
        metrics = self.compute_metrics(action, normal_state)
        results['swap_d_normal_with_halluc_rightwrist'] = metrics
        print(f"  Delta max: {metrics['delta_max']:.3f}")
        print(f"  Shape: {metrics['shape']}")
        print(f"  Question: Does adding banana cause hallucination? → {'YES' if metrics['shape'] == 'RAMP' else 'NO'}")

        # Analysis
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)

        baseline_diff = results['baseline_halluc']['delta_max'] - results['baseline_normal']['delta_max']
        print(f"\nBaseline difference: {baseline_diff:.3f}")
        print(f"  Halluc: {results['baseline_halluc']['delta_max']:.3f} ({results['baseline_halluc']['shape']})")
        print(f"  Normal: {results['baseline_normal']['delta_max']:.3f} ({results['baseline_normal']['shape']})")

        # Head camera swap effect
        head_swap_a_effect = results['baseline_halluc']['delta_max'] - results['swap_a_halluc_with_normal_head']['delta_max']
        head_swap_b_effect = results['swap_b_normal_with_halluc_head']['delta_max'] - results['baseline_normal']['delta_max']

        print(f"\nHead Camera Swap Effects:")
        print(f"  Swap A (add normal head to halluc): delta change = {-head_swap_a_effect:.3f}")
        print(f"    {results['baseline_halluc']['shape']} → {results['swap_a_halluc_with_normal_head']['shape']}")
        print(f"  Swap B (add halluc head to normal): delta change = {head_swap_b_effect:.3f}")
        print(f"    {results['baseline_normal']['shape']} → {results['swap_b_normal_with_halluc_head']['shape']}")

        # Right wrist swap effect
        rw_swap_c_effect = results['baseline_halluc']['delta_max'] - results['swap_c_halluc_with_normal_rightwrist']['delta_max']
        rw_swap_d_effect = results['swap_d_normal_with_halluc_rightwrist']['delta_max'] - results['baseline_normal']['delta_max']

        print(f"\nRight Wrist (Banana) Swap Effects:")
        print(f"  Swap C (remove banana from halluc): delta change = {-rw_swap_c_effect:.3f}")
        print(f"    {results['baseline_halluc']['shape']} → {results['swap_c_halluc_with_normal_rightwrist']['shape']}")
        print(f"  Swap D (add banana to normal): delta change = {rw_swap_d_effect:.3f}")
        print(f"    {results['baseline_normal']['shape']} → {results['swap_d_normal_with_halluc_rightwrist']['shape']}")

        # Conclusion
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)

        head_causal = abs(head_swap_a_effect) + abs(head_swap_b_effect)
        rw_causal = abs(rw_swap_c_effect) + abs(rw_swap_d_effect)

        print(f"\nTotal Head Camera Causal Effect: {head_causal:.3f}")
        print(f"Total Right Wrist Causal Effect: {rw_causal:.3f}")

        if head_causal > rw_causal * 1.5:
            conclusion = "HEAD_CAMERA_DOMINANT"
            print(f"\n→ HEAD CAMERA is the primary causal driver ({head_causal:.1f} > {rw_causal:.1f})")
        elif rw_causal > head_causal * 1.5:
            conclusion = "RIGHT_WRIST_DOMINANT"
            print(f"\n→ RIGHT WRIST (banana) is the primary causal driver ({rw_causal:.1f} > {head_causal:.1f})")
        else:
            conclusion = "MIXED_CAUSALITY"
            print(f"\n→ MIXED causality - both contribute ({head_causal:.1f} vs {rw_causal:.1f})")

        results['conclusion'] = {
            'head_causal_effect': head_causal,
            'right_wrist_causal_effect': rw_causal,
            'dominant_factor': conclusion,
            'head_swap_a_flipped': results['swap_a_halluc_with_normal_head']['shape'] != results['baseline_halluc']['shape'],
            'head_swap_b_flipped': results['swap_b_normal_with_halluc_head']['shape'] != results['baseline_normal']['shape'],
            'rw_swap_c_flipped': results['swap_c_halluc_with_normal_rightwrist']['shape'] != results['baseline_halluc']['shape'],
            'rw_swap_d_flipped': results['swap_d_normal_with_halluc_rightwrist']['shape'] != results['baseline_normal']['shape'],
        }

        # Save results
        results_path = Path(output_dir) / "swap_experiment_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Generate report
        self.generate_report(results, output_dir)

        return results

    def generate_report(self, results: dict, output_dir: str):
        """Generate markdown report."""
        report_path = Path(output_dir) / "report.md"

        with open(report_path, 'w') as f:
            f.write("# Head Camera Swap Experiment Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            f.write("## Purpose\n\n")
            f.write("Test if swapping the head camera between hallucination and normal cases\n")
            f.write("can flip the behavior, proving head camera is the causal driver.\n\n")

            f.write("## Results Summary\n\n")
            f.write("| Experiment | Delta Max | Shape | Behavior Flipped? |\n")
            f.write("|------------|-----------|-------|-------------------|\n")

            f.write(f"| Baseline Halluc | {results['baseline_halluc']['delta_max']:.3f} | {results['baseline_halluc']['shape']} | - |\n")
            f.write(f"| Baseline Normal | {results['baseline_normal']['delta_max']:.3f} | {results['baseline_normal']['shape']} | - |\n")
            f.write(f"| Swap A: Halluc + Normal HEAD | {results['swap_a_halluc_with_normal_head']['delta_max']:.3f} | {results['swap_a_halluc_with_normal_head']['shape']} | {results['conclusion']['head_swap_a_flipped']} |\n")
            f.write(f"| Swap B: Normal + Halluc HEAD | {results['swap_b_normal_with_halluc_head']['delta_max']:.3f} | {results['swap_b_normal_with_halluc_head']['shape']} | {results['conclusion']['head_swap_b_flipped']} |\n")
            f.write(f"| Swap C: Halluc + Normal RW | {results['swap_c_halluc_with_normal_rightwrist']['delta_max']:.3f} | {results['swap_c_halluc_with_normal_rightwrist']['shape']} | {results['conclusion']['rw_swap_c_flipped']} |\n")
            f.write(f"| Swap D: Normal + Halluc RW | {results['swap_d_normal_with_halluc_rightwrist']['delta_max']:.3f} | {results['swap_d_normal_with_halluc_rightwrist']['shape']} | {results['conclusion']['rw_swap_d_flipped']} |\n")

            f.write("\n## Causal Effect Comparison\n\n")
            f.write(f"- **Head Camera Total Effect**: {results['conclusion']['head_causal_effect']:.3f}\n")
            f.write(f"- **Right Wrist Total Effect**: {results['conclusion']['right_wrist_causal_effect']:.3f}\n")
            f.write(f"- **Dominant Factor**: {results['conclusion']['dominant_factor']}\n")

            f.write("\n## Interpretation\n\n")
            if results['conclusion']['dominant_factor'] == 'HEAD_CAMERA_DOMINANT':
                f.write("The head camera swap experiment confirms that **head camera is the primary causal driver**.\n")
                f.write("Swapping head camera has a larger effect on behavior than swapping right wrist (banana).\n")
            elif results['conclusion']['dominant_factor'] == 'RIGHT_WRIST_DOMINANT':
                f.write("The experiment shows **right wrist (banana) is the primary causal driver**.\n")
                f.write("This contradicts the attention knockout findings and requires further investigation.\n")
            else:
                f.write("The experiment shows **mixed causality** - both head camera and right wrist contribute.\n")
                f.write("The hallucination may be driven by a combination of factors.\n")

        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Head Camera Swap Experiment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--halluc-dir", type=str, required=True, help="Hallucination case directory")
    parser.add_argument("--normal-dir", type=str, required=True, help="Normal case directory")
    parser.add_argument("--step", type=int, default=200, help="Inference step to analyze")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    experiment = HeadCameraSwapExperiment(args.checkpoint, args.device)
    experiment.run_swap_experiment(
        halluc_dir=args.halluc_dir,
        normal_dir=args.normal_dir,
        step=args.step,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
