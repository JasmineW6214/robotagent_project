#!/usr/bin/env python3
"""
Attention Flow Analysis for SmolVLA Hallucination Investigation.

Analyzes how action tokens attend to different prefix regions during denoising,
and how this differs between hallucination and normal cases.

Key Questions:
- Which layers show the most attention difference between cases?
- Does the model attend differently to language vs visual tokens?
- Is there a language-visual interaction pattern that explains hallucination?

Usage:
    python attention_flow_analysis.py \
        --case-dirs logs/yogurt_banana_leftarm/case_20260119_131914_ha_bana_table \
                    logs/yogurt_banana_leftarm/case_20260119_132946_no_ha_plate \
        --step 200 \
        --output-dir logs/investigation/attention_flow
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not installed")
    sys.exit(1)

# Add project src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Token regions
TOKEN_REGIONS = {
    "head_camera": (0, 64),
    "left_wrist": (64, 128),
    "right_wrist": (128, 192),
    "language": (192, 240),
    "state": (240, 241),
}


@dataclass
class AttentionPattern:
    """Attention pattern from action tokens to prefix."""
    case_name: str
    step: int
    # Per-layer, per-region attention weights
    # Shape: {layer_idx: {region_name: mean_attention}}
    layer_region_attention: Dict[int, Dict[str, float]]
    # Per-region total attention (summed across layers)
    region_total_attention: Dict[str, float]
    # Most attended region per layer
    most_attended_per_layer: Dict[int, str]


@dataclass
class AttentionFlowAnalysis:
    """Analysis of attention flow differences."""
    cases: List[AttentionPattern]
    attention_difference: Dict[str, float]  # Per-region difference
    layer_differences: Dict[int, Dict[str, float]]  # Per-layer, per-region
    most_different_region: str
    most_different_layer: int
    findings: List[str]


class AttentionFlowAnalyzer:
    """Analyzes attention flow from action tokens to prefix."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(checkpoint_path)
        self.attention_weights = {}
        self.hooks = []

    def load_model(self, checkpoint_path: str):
        """Load SmolVLA model."""
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        print(f"Loading model from {checkpoint_path}...")
        model = SmolVLAPolicy.from_pretrained(checkpoint_path)
        model.to(self.device)
        model.eval()
        print("Model loaded successfully.")
        return model

    def register_attention_hooks(self):
        """Register hooks to capture attention weights."""
        self.attention_weights = {}
        self.hooks = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is (attn_output, attn_weights) when output_attentions=True
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]  # [batch, heads, seq, seq]
                    if attn_weights is not None:
                        self.attention_weights[layer_idx] = attn_weights.detach().cpu()
            return hook

        # Register hooks on attention layers (text model layers)
        text_model = self.model.model.vlm_with_expert.vlm.model.text_model
        for layer_idx, layer in enumerate(text_model.layers):
            hook = layer.self_attn.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}

    def load_case_data(self, case_dir: str, step: int) -> dict:
        """Load case data at a specific step."""
        case_path = Path(case_dir)

        # Load trace
        trace_path = case_path / "trace.jsonl"
        step_data = None
        with open(trace_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if data["step"] == step:
                    step_data = data
                    break

        if step_data is None:
            raise ValueError(f"Step {step} not found in trace")

        # Load images (try both PNG and JPG formats)
        images = {}
        for cam in ["head", "left_wrist", "right_wrist"]:
            # Try different file naming conventions
            img_paths = [
                case_path / "images" / f"step_{step:05d}_{cam}.png",
                case_path / "images" / f"step_{step:04d}_{cam}.png",
                case_path / "images" / f"step_{step:05d}_{cam}.jpg",
                case_path / "images" / f"step_{step:04d}_{cam}.jpg",
            ]
            import cv2
            for img_path in img_paths:
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images[f"observation.images.{cam}"] = img
                    break

        return {
            "step": step,
            "state": step_data.get("state_normalized", step_data.get("state_raw")),
            "task": step_data.get("task", step_data.get("task_modified")),
            "images": images,
            "case_name": case_path.name,
        }

    def prepare_batch(self, case_data: dict) -> dict:
        """Prepare batch for model inference."""
        processor = self.model.model.vlm_with_expert.processor

        # Camera mapping: image files -> model expected keys
        camera_mapping = {
            "head": "camera1",
            "left_wrist": "camera2",
            "right_wrist": "camera3",
        }

        # Prepare images
        images_dict = {}
        for file_cam, model_cam in camera_mapping.items():
            file_key = f"observation.images.{file_cam}"
            model_key = f"observation.images.{model_cam}"
            if file_key in case_data["images"]:
                img = case_data["images"][file_key]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                images_dict[model_key] = img_tensor.unsqueeze(0).to(self.device)

        # Prepare state
        state = torch.tensor(case_data["state"], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Prepare language tokens
        task = case_data["task"]
        text_inputs = processor.tokenizer(
            task,
            return_tensors="pt",
            padding="max_length",
            max_length=48,
            truncation=True,
        )

        batch = {
            **images_dict,
            "observation.state": state,
            "observation.language.tokens": text_inputs["input_ids"].to(self.device),
            "observation.language.attention_mask": text_inputs["attention_mask"].to(self.device),
        }

        return batch

    def run_inference_with_attention(self, batch: dict) -> Tuple[np.ndarray, Dict]:
        """Run inference and capture attention weights."""
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        # Register hooks
        self.register_attention_hooks()

        try:
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

                # Store config for output_attentions
                original_output_attentions = self.model.model.vlm_with_expert.config.output_attentions
                self.model.model.vlm_with_expert.config.output_attentions = True

                # Fill KV cache (this captures attention)
                _, past_key_values = self.model.model.vlm_with_expert.forward(
                    attention_mask=prefix_att_2d_masks,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=self.model.config.use_cache,
                    fill_kv_cache=True,
                )

                # Capture attention from first denoising step
                bsize = state.shape[0]
                device = state.device
                actions_shape = (bsize, self.model.config.chunk_size, self.model.config.max_action_dim)
                noise = self.model.model.sample_noise(actions_shape, device)

                time_tensor = torch.tensor(1.0, dtype=torch.float32, device=device).expand(bsize)

                # Run one denoising step to capture cross-attention
                suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.model.embed_suffix(
                    noise, time_tensor
                )

                suffix_len = suffix_pad_masks.shape[1]
                prefix_len = prefix_pad_masks.shape[1]
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
                suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

                # This forward captures cross-attention from action tokens to prefix
                outputs_embeds, _ = self.model.model.vlm_with_expert.forward(
                    attention_mask=full_att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=self.model.config.use_cache,
                    fill_kv_cache=False,
                )

                # Restore config
                self.model.model.vlm_with_expert.config.output_attentions = original_output_attentions

                # Get attention weights (copy before hooks are removed)
                attention_copy = {k: v.clone() for k, v in self.attention_weights.items()}

        finally:
            self.remove_hooks()

        return attention_copy

    def analyze_attention_pattern(self, attention_weights: Dict, case_name: str, step: int) -> AttentionPattern:
        """Analyze attention pattern from action tokens to prefix regions."""

        layer_region_attention = {}
        prefix_len = 241  # Standard prefix length

        for layer_idx, attn in attention_weights.items():
            # attn shape: [batch, heads, query_seq, key_seq]
            # We want attention from action tokens (suffix) to prefix tokens

            if attn.shape[-1] < prefix_len:
                continue  # This is prefix self-attention, skip

            # Extract attention to prefix (first 241 tokens of key dimension)
            # For suffix tokens attending to prefix
            attn_to_prefix = attn[:, :, :, :prefix_len]  # [batch, heads, suffix_len, prefix_len]

            # Average across batch, heads, and suffix tokens
            mean_attn = attn_to_prefix.mean(dim=(0, 1, 2))  # [prefix_len]

            # Compute per-region attention
            region_attn = {}
            for region_name, (start, end) in TOKEN_REGIONS.items():
                region_attn[region_name] = float(mean_attn[start:end].sum())

            layer_region_attention[layer_idx] = region_attn

        # Compute total per-region attention
        region_total = {region: 0.0 for region in TOKEN_REGIONS.keys()}
        for layer_attn in layer_region_attention.values():
            for region, val in layer_attn.items():
                region_total[region] += val

        # Normalize
        total = sum(region_total.values())
        if total > 0:
            region_total = {k: v / total for k, v in region_total.items()}

        # Most attended per layer
        most_attended = {}
        for layer_idx, region_attn in layer_region_attention.items():
            most_attended[layer_idx] = max(region_attn, key=region_attn.get)

        return AttentionPattern(
            case_name=case_name,
            step=step,
            layer_region_attention=layer_region_attention,
            region_total_attention=region_total,
            most_attended_per_layer=most_attended,
        )

    def compare_patterns(self, patterns: List[AttentionPattern]) -> AttentionFlowAnalysis:
        """Compare attention patterns between cases."""

        if len(patterns) < 2:
            return None

        halluc = patterns[0]  # Assume first is hallucination
        normal = patterns[1]  # Assume second is normal

        # Compute per-region difference
        attention_diff = {}
        for region in TOKEN_REGIONS.keys():
            diff = halluc.region_total_attention.get(region, 0) - normal.region_total_attention.get(region, 0)
            attention_diff[region] = diff

        # Compute per-layer, per-region difference
        layer_diffs = {}
        all_layers = set(halluc.layer_region_attention.keys()) | set(normal.layer_region_attention.keys())

        for layer_idx in all_layers:
            h_attn = halluc.layer_region_attention.get(layer_idx, {})
            n_attn = normal.layer_region_attention.get(layer_idx, {})

            layer_diffs[layer_idx] = {}
            for region in TOKEN_REGIONS.keys():
                diff = h_attn.get(region, 0) - n_attn.get(region, 0)
                layer_diffs[layer_idx][region] = diff

        # Find most different
        most_diff_region = max(attention_diff.keys(), key=lambda k: abs(attention_diff[k]))

        # Most different layer
        layer_total_diff = {}
        for layer_idx, region_diffs in layer_diffs.items():
            layer_total_diff[layer_idx] = sum(abs(v) for v in region_diffs.values())

        most_diff_layer = max(layer_total_diff.keys(), key=layer_total_diff.get) if layer_total_diff else 0

        # Generate findings
        findings = []

        # Check if language attention differs
        lang_diff = attention_diff.get("language", 0)
        if abs(lang_diff) > 0.01:
            direction = "more" if lang_diff > 0 else "less"
            findings.append(f"Hallucination case attends {direction} to language tokens ({lang_diff:.3f})")

        # Check head camera
        head_diff = attention_diff.get("head_camera", 0)
        if abs(head_diff) > 0.01:
            direction = "more" if head_diff > 0 else "less"
            findings.append(f"Hallucination case attends {direction} to head camera ({head_diff:.3f})")

        # Check right wrist
        rw_diff = attention_diff.get("right_wrist", 0)
        if abs(rw_diff) > 0.01:
            direction = "more" if rw_diff > 0 else "less"
            findings.append(f"Hallucination case attends {direction} to right wrist ({rw_diff:.3f})")

        return AttentionFlowAnalysis(
            cases=patterns,
            attention_difference=attention_diff,
            layer_differences=layer_diffs,
            most_different_region=most_diff_region,
            most_different_layer=most_diff_layer,
            findings=findings,
        )

    def visualize(self, analysis: AttentionFlowAnalysis, output_dir: Path):
        """Generate visualizations."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Region attention comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        regions = list(TOKEN_REGIONS.keys())
        x = np.arange(len(regions))
        width = 0.35

        # Per-case attention
        for i, pattern in enumerate(analysis.cases):
            values = [pattern.region_total_attention.get(r, 0) for r in regions]
            axes[0].bar(x + i * width, values, width, label=pattern.case_name[:30])

        axes[0].set_xlabel("Token Region")
        axes[0].set_ylabel("Normalized Attention")
        axes[0].set_title("Attention to Prefix Regions")
        axes[0].set_xticks(x + width / 2)
        axes[0].set_xticklabels(regions, rotation=45, ha='right')
        axes[0].legend()

        # Difference
        diff_values = [analysis.attention_difference.get(r, 0) for r in regions]
        colors = ['green' if v > 0 else 'red' for v in diff_values]
        axes[1].bar(x, diff_values, color=colors)
        axes[1].set_xlabel("Token Region")
        axes[1].set_ylabel("Attention Difference (Halluc - Normal)")
        axes[1].set_title("Attention Difference by Region")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(regions, rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "attention_comparison.png", dpi=150)
        plt.close()

        # 2. Per-layer heatmap
        if analysis.layer_differences:
            layers = sorted(analysis.layer_differences.keys())

            data = np.zeros((len(layers), len(regions)))
            for i, layer_idx in enumerate(layers):
                for j, region in enumerate(regions):
                    data[i, j] = analysis.layer_differences[layer_idx].get(region, 0)

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))

            ax.set_xticks(range(len(regions)))
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"Layer {l}" for l in layers])
            ax.set_xlabel("Token Region")
            ax.set_ylabel("Layer")
            ax.set_title("Attention Difference by Layer and Region\n(Hallucination - Normal)")

            plt.colorbar(im, ax=ax, label="Attention Diff")
            plt.tight_layout()
            plt.savefig(output_dir / "layer_region_heatmap.png", dpi=150)
            plt.close()

    def generate_report(self, analysis: AttentionFlowAnalysis, output_dir: Path):
        """Generate markdown report."""

        report = f"""# Attention Flow Analysis Report

**Generated**: {datetime.now().isoformat()}

## Purpose

This analysis examines how action tokens attend to different prefix regions
during the denoising process, comparing hallucination vs normal cases.

Key questions:
- Which regions receive more attention in hallucination cases?
- Does the model attend differently to language vs visual tokens?
- Which layers show the most attention difference?

## Cases Analyzed

"""
        for pattern in analysis.cases:
            report += f"### {pattern.case_name}\n"
            report += f"- Step: {pattern.step}\n"
            report += "\n**Region Attention:**\n\n"
            report += "| Region | Attention |\n"
            report += "|--------|----------|\n"
            for region, attn in pattern.region_total_attention.items():
                report += f"| {region} | {attn:.4f} |\n"
            report += "\n"

        report += """## Attention Differences

### Per-Region Difference (Hallucination - Normal)

| Region | Difference | Direction |
|--------|------------|-----------|
"""
        for region, diff in sorted(analysis.attention_difference.items(), key=lambda x: -abs(x[1])):
            direction = "↑ MORE" if diff > 0 else "↓ LESS" if diff < 0 else "="
            report += f"| {region} | {diff:+.4f} | {direction} |\n"

        report += f"""
## Key Findings

**Most Different Region**: {analysis.most_different_region}
**Most Different Layer**: {analysis.most_different_layer}

### Observations

"""
        for finding in analysis.findings:
            report += f"- {finding}\n"

        if not analysis.findings:
            report += "- No significant attention differences detected\n"

        report += """
## Interpretation

- **Positive difference**: Hallucination case attends MORE to this region
- **Negative difference**: Hallucination case attends LESS to this region

High attention difference in a region suggests that region contributes
differently to the hallucination behavior.

## Visualizations

- `attention_comparison.png`: Per-region attention comparison
- `layer_region_heatmap.png`: Per-layer attention differences
"""

        with open(output_dir / "report.md", "w") as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze attention flow")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "outputs/smolvla_bimanual_20260103_200201/checkpoints/040000/pretrained_model"))
    parser.add_argument("--case-dirs", nargs="+", required=True)
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument("--output-dir", default="logs/investigation/attention_flow")
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Output directory: {output_dir}")

    analyzer = AttentionFlowAnalyzer(args.checkpoint, args.device)

    patterns = []
    for case_dir in args.case_dirs:
        print(f"\nAnalyzing: {case_dir}")
        case_data = analyzer.load_case_data(case_dir, args.step)
        batch = analyzer.prepare_batch(case_data)

        print("  Running inference with attention capture...")
        attention_weights = analyzer.run_inference_with_attention(batch)

        print(f"  Captured attention from {len(attention_weights)} layers")

        pattern = analyzer.analyze_attention_pattern(
            attention_weights, case_data["case_name"], args.step
        )
        patterns.append(pattern)

        print(f"  Region attention: {pattern.region_total_attention}")

    print("\nComparing patterns...")
    analysis = analyzer.compare_patterns(patterns)

    if analysis:
        print(f"Most different region: {analysis.most_different_region}")
        print(f"Most different layer: {analysis.most_different_layer}")

        for finding in analysis.findings:
            print(f"  - {finding}")

        print("\nGenerating visualizations...")
        analyzer.visualize(analysis, output_dir)

        print("Generating report...")
        analyzer.generate_report(analysis, output_dir)

        # Save JSON
        with open(output_dir / "analysis.json", "w") as f:
            json.dump({
                "attention_difference": analysis.attention_difference,
                "most_different_region": analysis.most_different_region,
                "most_different_layer": analysis.most_different_layer,
                "findings": analysis.findings,
            }, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
