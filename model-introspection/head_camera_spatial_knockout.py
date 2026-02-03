#!/usr/bin/env python3
"""
Head Camera Spatial Knockout Experiment

Purpose: Identify which spatial region of the head camera (64 tokens in 8x8 grid)
has the highest causal importance for action generation.

Method:
1. Divide head camera's 64 tokens into spatial regions (quadrants, rows, etc.)
2. Knock out each region by zeroing its KV cache
3. Measure which knockout causes the largest action change
4. This reveals which part of the scene the model relies on

Expected Insight: If bottom rows (table surface) matter more than top rows (background),
the model is using table configuration for decisions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


class HeadCameraSpatialKnockout:
    """Experiment to test which spatial region of head camera matters most."""

    # Token layout for head camera (8x8 grid = 64 tokens)
    # Tokens 0-63 are head camera in the prefix
    HEAD_CAM_START = 0
    HEAD_CAM_END = 64

    # Define spatial regions
    REGIONS = {
        # Quadrants (16 tokens each)
        'Q1_top_left': list(range(0, 4)) + list(range(8, 12)) + list(range(16, 20)) + list(range(24, 28)),
        'Q2_top_right': list(range(4, 8)) + list(range(12, 16)) + list(range(20, 24)) + list(range(28, 32)),
        'Q3_bottom_left': list(range(32, 36)) + list(range(40, 44)) + list(range(48, 52)) + list(range(56, 60)),
        'Q4_bottom_right': list(range(36, 40)) + list(range(44, 48)) + list(range(52, 56)) + list(range(60, 64)),

        # Horizontal halves
        'top_half': list(range(0, 32)),
        'bottom_half': list(range(32, 64)),

        # Vertical halves
        'left_half': [i for i in range(64) if (i % 8) < 4],
        'right_half': [i for i in range(64) if (i % 8) >= 4],

        # Individual rows (8 tokens each)
        'row_0': list(range(0, 8)),
        'row_1': list(range(8, 16)),
        'row_2': list(range(16, 24)),
        'row_3': list(range(24, 32)),
        'row_4': list(range(32, 40)),
        'row_5': list(range(40, 48)),
        'row_6': list(range(48, 56)),
        'row_7': list(range(56, 64)),

        # Center region
        'center': list(range(18, 22)) + list(range(26, 30)) + list(range(34, 38)) + list(range(42, 46)),

        # Edge regions
        'edges': [i for i in range(64) if (i % 8) == 0 or (i % 8) == 7 or i < 8 or i >= 56],
    }

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path

        print(f"Loading model from {checkpoint_path}...")
        self.config = SmolVLAConfig.from_pretrained(checkpoint_path)
        self.model = SmolVLAForActionPrediction.from_pretrained(
            checkpoint_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()

        self.processor = self.model.model.get_processor()

        self.camera_map = {
            'head': 'observation.images.camera1',
            'left_wrist': 'observation.images.camera2',
            'right_wrist': 'observation.images.camera3',
        }

    def load_case_data(self, case_dir: str, step: int = 200):
        """Load images and state from a case directory."""
        case_path = Path(case_dir)

        trace_path = case_path / "trace.jsonl"
        state = None
        with open(trace_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('step') == step:
                    state = data.get('state_raw') or data.get('state')
                    break

        if state is None:
            raise ValueError(f"Could not find step {step} in trace")

        images = {}
        for cam_name in ['head', 'left_wrist', 'right_wrist']:
            img_path = None
            patterns = [
                f"step_{step:05d}_{cam_name}.png",
                f"step_{step:04d}_{cam_name}.png",
                f"step_{step:05d}_{cam_name}.jpg",
                f"step_{step:04d}_{cam_name}.jpg",
            ]
            for pattern in patterns:
                candidate = case_path / pattern
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                raise FileNotFoundError(f"Could not find {cam_name} image for step {step}")

            images[cam_name] = Image.open(img_path).convert('RGB')

        return images, np.array(state)

    def prepare_batch(self, images: dict, state: np.ndarray, language: str):
        """Prepare model input batch."""
        observation = {}
        for cam_name, model_key in self.camera_map.items():
            img = images[cam_name]
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            observation[model_key] = img_tensor.unsqueeze(0).to(self.device)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        observation['observation.state'] = state_tensor

        task_tokens = self.processor.tokenizer(
            language,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
        )

        batch = {
            **observation,
            'task': language,
            'task_tokens': task_tokens['input_ids'].to(self.device),
            'task_attention_mask': task_tokens['attention_mask'].to(self.device),
        }

        return batch

    def get_kv_cache(self, batch: dict):
        """Get KV cache from model."""
        with torch.no_grad():
            images = self.model.model.prepare_images(batch)
            state = self.model.model.prepare_state(batch)
            task_tokens = batch['task_tokens']
            task_attention_mask = batch['task_attention_mask']

            prefix_embeddings, prefix_pad_masks = self.model.model.embed_prefix(
                images, state, task_tokens, task_attention_mask
            )

            past_key_values = self.model.model.fill_kv_cache(prefix_embeddings, prefix_pad_masks)

        return past_key_values, prefix_pad_masks, state

    def knockout_region(self, past_key_values, token_indices: List[int]):
        """Zero out KV cache for specified token indices."""
        modified_kv = []

        for layer_kv in past_key_values:
            key_states, value_states = layer_kv

            # Clone to avoid modifying original
            key_mod = key_states.clone()
            value_mod = value_states.clone()

            # Zero out specified tokens
            for idx in token_indices:
                key_mod[:, :, idx, :] = 0
                value_mod[:, :, idx, :] = 0

            modified_kv.append((key_mod, value_mod))

        return modified_kv

    def run_denoising(self, past_key_values, prefix_pad_masks, state) -> np.ndarray:
        """Run denoising with given KV cache."""
        with torch.no_grad():
            bsize = state.shape[0]
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.model.model.sample_noise(actions_shape, self.device)

            num_steps = self.config.num_steps
            dt = -1.0 / num_steps
            x_t = noise

            for step in range(num_steps):
                time = 1.0 + step * dt
                time_tensor = torch.tensor(time, dtype=torch.float32, device=self.device).expand(bsize)
                v_t = self.model.model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                x_t = x_t + dt * v_t

            actions = x_t

        return actions[0].float().cpu().numpy()

    def compute_metrics(self, action: np.ndarray, baseline_action: np.ndarray, state: np.ndarray) -> dict:
        """Compute change metrics."""
        # Delta from state
        action_first = action[0, :len(state)]
        baseline_first = baseline_action[0, :len(state)]

        delta = np.abs(action_first - state).max()
        baseline_delta = np.abs(baseline_first - state).max()

        # Change from baseline
        l2_dist = np.linalg.norm(action - baseline_action)
        cosine_sim = np.dot(action.flatten(), baseline_action.flatten()) / (
            np.linalg.norm(action) * np.linalg.norm(baseline_action) + 1e-8
        )

        delta_change = delta - baseline_delta
        pct_change = (delta_change / (baseline_delta + 1e-8)) * 100

        return {
            'delta': float(delta),
            'baseline_delta': float(baseline_delta),
            'delta_change': float(delta_change),
            'pct_change': float(pct_change),
            'l2_distance': float(l2_dist),
            'cosine_similarity': float(cosine_sim),
        }

    def run_experiment(
        self,
        case_dir: str,
        step: int,
        language: str,
        output_dir: str,
        regions_to_test: List[str] = None,
    ):
        """Run spatial knockout experiment."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print("HEAD CAMERA SPATIAL KNOCKOUT EXPERIMENT")
        print(f"{'='*60}")
        print(f"Case: {case_dir}")
        print(f"Step: {step}")

        # Load data
        images, state = self.load_case_data(case_dir, step)
        batch = self.prepare_batch(images, state, language)

        # Get baseline KV cache and action
        print("\nGetting baseline...")
        kv_cache, prefix_pad_masks, state_tensor = self.get_kv_cache(batch)
        baseline_action = self.run_denoising(kv_cache, prefix_pad_masks, state_tensor)

        baseline_delta = np.abs(baseline_action[0, :len(state)] - state).max()
        print(f"Baseline delta max: {baseline_delta:.3f}")

        # Select regions to test
        if regions_to_test is None:
            regions_to_test = list(self.REGIONS.keys())

        results = {
            'baseline_delta': float(baseline_delta),
            'regions': {},
        }

        print(f"\nTesting {len(regions_to_test)} regions...")

        for region_name in regions_to_test:
            token_indices = self.REGIONS[region_name]
            print(f"\n  Knocking out {region_name} ({len(token_indices)} tokens)...")

            # Knockout and run inference
            modified_kv = self.knockout_region(kv_cache, token_indices)
            knockout_action = self.run_denoising(modified_kv, prefix_pad_masks, state_tensor)

            # Compute metrics
            metrics = self.compute_metrics(knockout_action, baseline_action, state)
            metrics['num_tokens'] = len(token_indices)
            metrics['token_indices'] = token_indices

            results['regions'][region_name] = metrics
            print(f"    Delta change: {metrics['delta_change']:.3f} ({metrics['pct_change']:.1f}%)")
            print(f"    L2 distance: {metrics['l2_distance']:.3f}")

        # Rank regions by causal effect
        ranked = sorted(
            results['regions'].items(),
            key=lambda x: abs(x[1]['pct_change']),
            reverse=True
        )

        print("\n" + "="*60)
        print("RANKED BY CAUSAL EFFECT")
        print("="*60)
        for i, (name, metrics) in enumerate(ranked[:10]):
            print(f"{i+1}. {name}: {metrics['pct_change']:.1f}% change, L2={metrics['l2_distance']:.3f}")

        results['ranked_regions'] = [(name, metrics['pct_change']) for name, metrics in ranked]
        results['most_causal_region'] = ranked[0][0] if ranked else None

        # Save results
        results_path = Path(output_dir) / "spatial_knockout_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Generate visualizations
        self.visualize_results(results, images['head'], output_dir)

        # Generate report
        self.generate_report(results, output_dir)

        return results

    def visualize_results(self, results: dict, head_image: Image.Image, output_dir: str):
        """Generate visualization of spatial knockout effects."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Panel 1: Head camera image with grid overlay
        ax1 = axes[0, 0]
        ax1.imshow(head_image)

        # Draw 8x8 grid
        img_w, img_h = head_image.size
        for i in range(9):
            ax1.axhline(y=i * img_h / 8, color='white', linewidth=0.5, alpha=0.5)
            ax1.axvline(x=i * img_w / 8, color='white', linewidth=0.5, alpha=0.5)

        ax1.set_title("Head Camera with Token Grid (8x8)")
        ax1.axis('off')

        # Panel 2: Quadrant effects
        ax2 = axes[0, 1]
        quadrant_names = ['Q1_top_left', 'Q2_top_right', 'Q3_bottom_left', 'Q4_bottom_right']
        quadrant_effects = [results['regions'].get(q, {}).get('pct_change', 0) for q in quadrant_names]

        colors = ['red' if e > 0 else 'blue' for e in quadrant_effects]
        bars = ax2.bar(range(4), [abs(e) for e in quadrant_effects], color=colors, alpha=0.7)
        ax2.set_xticks(range(4))
        ax2.set_xticklabels(['Top-Left', 'Top-Right', 'Bottom-Left', 'Bottom-Right'], rotation=45)
        ax2.set_ylabel('|% Change| in Action')
        ax2.set_title('Quadrant Knockout Effects')
        ax2.axhline(y=0, color='black', linewidth=0.5)

        for bar, effect in zip(bars, quadrant_effects):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{effect:.1f}%', ha='center', va='bottom', fontsize=9)

        # Panel 3: Row effects (top to bottom)
        ax3 = axes[1, 0]
        row_names = [f'row_{i}' for i in range(8)]
        row_effects = [results['regions'].get(r, {}).get('pct_change', 0) for r in row_names]

        colors = ['red' if e > 0 else 'blue' for e in row_effects]
        bars = ax3.barh(range(8), [abs(e) for e in row_effects], color=colors, alpha=0.7)
        ax3.set_yticks(range(8))
        ax3.set_yticklabels([f'Row {i} ({"top" if i < 4 else "bottom"})' for i in range(8)])
        ax3.set_xlabel('|% Change| in Action')
        ax3.set_title('Row-by-Row Knockout Effects')
        ax3.invert_yaxis()  # Top row at top

        for bar, effect in zip(bars, row_effects):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{effect:.1f}%', ha='left', va='center', fontsize=9)

        # Panel 4: Heatmap of causal effect by token position
        ax4 = axes[1, 1]

        # Create heatmap data
        heatmap = np.zeros((8, 8))
        for region_name, metrics in results['regions'].items():
            if region_name.startswith('row_'):
                continue  # Skip rows for heatmap, use quadrants
            for idx in metrics.get('token_indices', []):
                row = idx // 8
                col = idx % 8
                # Average effect across overlapping regions
                current = heatmap[row, col]
                new_effect = abs(metrics['pct_change']) / len(metrics['token_indices'])
                heatmap[row, col] = max(current, new_effect)

        im = ax4.imshow(heatmap, cmap='Reds', aspect='auto')
        ax4.set_title('Causal Effect Heatmap (Token Position)')
        ax4.set_xlabel('Column')
        ax4.set_ylabel('Row')
        plt.colorbar(im, ax=ax4, label='Effect Magnitude')

        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'spatial_knockout_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_dir}/spatial_knockout_visualization.png")

    def generate_report(self, results: dict, output_dir: str):
        """Generate markdown report."""
        report_path = Path(output_dir) / "report.md"

        with open(report_path, 'w') as f:
            f.write("# Head Camera Spatial Knockout Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")

            f.write("## Purpose\n\n")
            f.write("Identify which spatial region of the head camera has the highest causal\n")
            f.write("importance for action generation by knocking out different token regions.\n\n")

            f.write("## Token Layout\n\n")
            f.write("```\n")
            f.write("Head Camera 8x8 Token Grid:\n")
            f.write("┌────┬────┬────┬────┬────┬────┬────┬────┐\n")
            for row in range(8):
                f.write("│")
                for col in range(8):
                    idx = row * 8 + col
                    f.write(f" {idx:2d} │")
                f.write(f"  ← Row {row}\n")
                if row < 7:
                    f.write("├────┼────┼────┼────┼────┼────┼────┼────┤\n")
            f.write("└────┴────┴────┴────┴────┴────┴────┴────┘\n")
            f.write("```\n\n")

            f.write(f"## Baseline\n\n")
            f.write(f"- **Baseline Delta Max**: {results['baseline_delta']:.3f}\n\n")

            f.write("## Quadrant Results\n\n")
            f.write("| Quadrant | % Change | L2 Distance | Interpretation |\n")
            f.write("|----------|----------|-------------|----------------|\n")
            quadrants = ['Q1_top_left', 'Q2_top_right', 'Q3_bottom_left', 'Q4_bottom_right']
            for q in quadrants:
                if q in results['regions']:
                    m = results['regions'][q]
                    interp = "HIGH" if abs(m['pct_change']) > 50 else "MEDIUM" if abs(m['pct_change']) > 20 else "LOW"
                    f.write(f"| {q} | {m['pct_change']:.1f}% | {m['l2_distance']:.3f} | {interp} |\n")

            f.write("\n## Row Results\n\n")
            f.write("| Row | Position | % Change | L2 Distance |\n")
            f.write("|-----|----------|----------|-------------|\n")
            for i in range(8):
                row_name = f'row_{i}'
                if row_name in results['regions']:
                    m = results['regions'][row_name]
                    pos = "Top (background)" if i < 2 else "Middle" if i < 6 else "Bottom (table)"
                    f.write(f"| Row {i} | {pos} | {m['pct_change']:.1f}% | {m['l2_distance']:.3f} |\n")

            f.write("\n## Ranked by Causal Effect\n\n")
            f.write("| Rank | Region | % Change |\n")
            f.write("|------|--------|----------|\n")
            for i, (name, pct) in enumerate(results['ranked_regions'][:10]):
                f.write(f"| {i+1} | {name} | {pct:.1f}% |\n")

            f.write(f"\n## Conclusion\n\n")
            f.write(f"**Most Causal Region**: {results['most_causal_region']}\n\n")

            if results['most_causal_region']:
                if 'bottom' in results['most_causal_region'].lower():
                    f.write("The bottom region (table surface) has highest causal effect.\n")
                    f.write("This suggests the model uses **table configuration** for decisions.\n")
                elif 'top' in results['most_causal_region'].lower():
                    f.write("The top region (background) has highest causal effect.\n")
                    f.write("This suggests the model uses **scene/background context** for decisions.\n")
                else:
                    f.write("The causal effect is distributed across regions.\n")

        print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Head Camera Spatial Knockout Experiment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--case-dir", type=str, required=True, help="Case directory")
    parser.add_argument("--step", type=int, default=200, help="Inference step to analyze")
    parser.add_argument("--language", type=str,
                        default="Pick up the yogurt bottle with your left arm and place it on the plate",
                        help="Task language instruction")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    experiment = HeadCameraSpatialKnockout(args.checkpoint, args.device)
    experiment.run_experiment(
        case_dir=args.case_dir,
        step=args.step,
        language=args.language,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
