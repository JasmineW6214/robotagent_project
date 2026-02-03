#!/usr/bin/env python3


import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# ============================================================================
# KEY CONFIGURATION
# ============================================================================
# Default checkpoint path (override with --checkpoint)
DEFAULT_CHECKPOINT = "outputs/so101/checkpoint-10000"

# Dataset path (override with --dataset)
DEFAULT_DATASET = "/home/jrobot/project/datasets/so101_pick_place"

# Modality config path
MODALITY_CONFIG_PATH = "custom/scripts/ver1_6/so101_config_1_6.py"

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Evaluation settings
NUM_TRAJECTORIES = 5      # Number of episodes to evaluate (or -1 for all)
STEPS_PER_TRAJ = 300      # Max steps per trajectory
ACTION_HORIZON = 16       # Must match training

# Output settings
OUTPUT_DIR = "eval_outputs/openloop"
SAVE_PLOTS = True
SAVE_METRICS = True
# ============================================================================

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def import_modality_config(config_path: str):
    """Import the modality config module to register NEW_EMBODIMENT."""
    full_path = PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Modality config not found: {full_path}")

    import importlib.util
    spec = importlib.util.spec_from_file_location("so101_config", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def load_base_model_policy(model_path: str, device: str):
    """
    Load base model (e.g., nvidia/GR00T-N1.6-3B) with custom modality config.

    The base model's processor doesn't have NEW_EMBODIMENT config, so we need to
    override it manually.
    """
    import torch
    from transformers import AutoModel, AutoProcessor
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.types import VLAStepData, MessageType
    import gr00t.model  # noqa: F401 - Register model classes

    # Load model
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)

    # Load processor with modality config override
    modality_configs_override = {
        EmbodimentTag.NEW_EMBODIMENT.value: MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value]
    }
    processor = AutoProcessor.from_pretrained(
        model_path,
        modality_configs=modality_configs_override
    )
    processor.eval()

    # Create a policy wrapper that matches Gr00tPolicy interface
    class BaseModelPolicy:
        def __init__(self, model, processor, embodiment_tag):
            self.model = model
            self.processor = processor
            self.embodiment_tag = embodiment_tag
            self.modality_configs = processor.get_modality_configs()[embodiment_tag.value]
            self.collate_fn = processor.collator
            self.device = device

            # Extract language key
            language_keys = self.modality_configs["language"].modality_keys
            assert len(language_keys) == 1, "Only one language key is supported"
            self.language_key = language_keys[0]

        def get_modality_config(self):
            return self.modality_configs

        def get_action(self, observation):
            """Run inference and return action."""
            # Unbatch observation
            unbatched_obs = []
            batch_size = observation["video"][list(observation["video"].keys())[0]].shape[0]
            for i in range(batch_size):
                unbatched_value = {
                    "video": {k: v[i] for k, v in observation["video"].items()},
                    "state": {k: v[i] for k, v in observation["state"].items()},
                    "language": {k: v[i] for k, v in observation["language"].items()},
                }
                unbatched_obs.append(unbatched_value)

            # Convert to VLAStepData and process each
            processed_list = []
            states_list = []
            for obs in unbatched_obs:
                step_data = VLAStepData(
                    images=obs["video"],
                    states=obs["state"],
                    actions={},  # No ground truth actions during inference
                    text=obs["language"][self.language_key][0],
                    embodiment=self.embodiment_tag,
                )
                states_list.append(step_data.states)
                # Processor expects messages format
                messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
                processed = self.processor(messages)
                processed_list.append(processed)

            # Collate processed features
            batch = self.collate_fn(processed_list)
            # Move to device and convert to bfloat16
            batch = {
                k: v.to(self.device, dtype=torch.bfloat16) if hasattr(v, 'to') and v.dtype in [torch.float32, torch.float64]
                else v.to(self.device) if hasattr(v, 'to') else v
                for k, v in batch.items()
            }

            # Run inference
            with torch.no_grad():
                output = self.model.get_action(**batch)

            # Get normalized action predictions
            normalized_action = output["action_pred"].float()

            # Stack states for decoding
            batched_states = {}
            for k in self.modality_configs["state"].modality_keys:
                batched_states[k] = np.stack([s[k] for s in states_list], axis=0)

            # Decode actions
            decoded = self.processor.decode_action(
                normalized_action.cpu().numpy(),
                self.embodiment_tag,
                batched_states,
            )

            # Cast to float32
            decoded = {k: v.astype(np.float32) for k, v in decoded.items()}

            return decoded, {}

    return BaseModelPolicy(model, processor, EmbodimentTag.NEW_EMBODIMENT)


def parse_observation(obs: dict[str, Any], modality_configs: dict[str, Any]) -> dict[str, Any]:
    """Parse raw observation to expected policy input format."""
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"

            arr = obs[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]

    return new_obs


def parse_action(action: dict[str, Any]) -> dict[str, Any]:
    """Unbatch and add prefix to action."""
    return {f"action.{key}": action[key][0] for key in action}


def extract_state_joints(traj: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Extract state joints from trajectory DataFrame."""
    np_dict = {}
    for column in columns:
        np_dict[column] = np.vstack([arr for arr in traj[column]])
    return np.concatenate([np_dict[column] for column in columns], axis=-1)


def plot_trajectory_results(
    state_joints: np.ndarray,
    gt_action: np.ndarray,
    pred_action: np.ndarray,
    traj_id: int,
    action_horizon: int,
    save_path: str,
    state_keys: list[str],
    action_keys: list[str],
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Matches the format shown in media/open_loop_eval_so100.png reference image.
    """
    actual_steps = len(gt_action)
    action_dim = gt_action.shape[1]

    indices_to_plot = list(range(action_dim))
    num_plots = len(indices_to_plot)

    if num_plots == 0:
        logger.warning("No valid indices to plot")
        return

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    if num_plots == 1:
        axes = [axes]

    # Global title format matching media/open_loop_eval_so100.png
    modalities = ", ".join(action_keys)
    fig.suptitle(
        f"Trajectory {traj_id} - Modalities: {modalities}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # Plot state joints only if dimensions match action
        if state_joints.shape == gt_action.shape:
            ax.plot(state_joints[:, action_idx], label="state joints")
        ax.plot(gt_action[:, action_idx], label="gt action joints")
        ax.plot(pred_action[:, action_idx], label="pred action joints")

        # Put a dot every ACTION_HORIZON (inference points)
        for j in range(0, actual_steps, action_horizon):
            if j == 0:
                ax.plot(j, gt_action[j, action_idx], "ro", label="inference point")
            else:
                ax.plot(j, gt_action[j, action_idx], "ro")

        ax.set_title(f"Joint {action_idx}")
        ax.legend()

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def evaluate_trajectory(
    policy,
    loader,
    traj_id: int,
    embodiment_tag,
    action_horizon: int = 16,
    max_steps: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str], list[str]]:
    """Evaluate a single trajectory and return predictions and ground truth."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(max_steps, traj_length)

    logger.info(f"  Trajectory {traj_id}: {actual_steps} steps (max: {traj_length})")

    pred_actions = []

    state_keys = loader.modality_configs["state"].modality_keys
    action_keys = loader.modality_configs["action"].modality_keys

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    # Run inference at action_horizon intervals
    for step_count in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)

        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for lang_key in loader.modality_configs["language"].modality_keys:
            obs[lang_key] = data_point.text

        parsed_obs = parse_observation(obs, loader.modality_configs)
        action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action(action_chunk)

        # Collect actions for each step in the horizon
        for j in range(action_horizon):
            concat_action = np.concatenate([
                np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                for key in action_keys
            ], axis=0)
            pred_actions.append(concat_action)

    # Extract ground truth (slice all to actual_steps for consistent shapes)
    state_joints = extract_state_joints(traj, [f"state.{key}" for key in state_keys])[:actual_steps]
    gt_actions = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[:actual_steps]
    pred_actions = np.array(pred_actions)[:actual_steps]

    assert gt_actions.shape == pred_actions.shape, (
        f"Shape mismatch: gt={gt_actions.shape}, pred={pred_actions.shape}"
    )

    # Compute metrics
    mse = np.mean((gt_actions - pred_actions) ** 2)
    mae = np.mean(np.abs(gt_actions - pred_actions))

    # Per-joint metrics
    per_joint_mse = np.mean((gt_actions - pred_actions) ** 2, axis=0)
    per_joint_mae = np.mean(np.abs(gt_actions - pred_actions), axis=0)

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "per_joint_mse": per_joint_mse.tolist(),
        "per_joint_mae": per_joint_mae.tolist(),
        "num_steps": actual_steps,
    }

    return state_joints, gt_actions, pred_actions, metrics, state_keys, action_keys


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop evaluation for GR00T 1.6"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Path to dataset (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--num-trajectories", "-n",
        type=int,
        default=NUM_TRAJECTORIES,
        help=f"Number of trajectories to evaluate (default: {NUM_TRAJECTORIES}, -1 for all)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for plots and metrics (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable trajectory plots"
    )
    parser.add_argument(
        "--traj-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific trajectory IDs to evaluate (overrides --num-trajectories)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GR00T 1.6 Open-Loop Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {args.output_dir}")
    print("=" * 70)

    # Validate paths
    # Allow HuggingFace model IDs (e.g., "nvidia/GR00T-N1.6-3B") which won't exist as local paths
    is_hf_model = "/" in args.checkpoint and not Path(args.checkpoint).exists()
    if not is_hf_model and not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if not Path(args.dataset).exists():
        logger.error(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import modality config
    import_modality_config(MODALITY_CONFIG_PATH)

    # Load policy
    logger.info("\nLoading model...")
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    # Check if this is a HuggingFace model ID (base model) or local checkpoint
    if is_hf_model:
        # For base model, we need custom loading with modality config override
        logger.info("  Loading base model with custom modality config...")
        policy = load_base_model_policy(args.checkpoint, DEVICE)
    else:
        # For finetuned checkpoints, use standard Gr00tPolicy
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=args.checkpoint,
            device=DEVICE,
        )
    logger.info(f"  Model loaded on {DEVICE}")

    # Load dataset
    modality_config = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset,
        modality_configs=modality_config,
        video_backend="torchcodec",
    )
    logger.info(f"  Dataset loaded: {len(loader)} episodes")

    # Determine which trajectories to evaluate
    if args.traj_ids:
        traj_ids = args.traj_ids
    elif args.num_trajectories == -1:
        traj_ids = list(range(len(loader)))
    else:
        traj_ids = list(range(min(args.num_trajectories, len(loader))))

    logger.info(f"  Evaluating {len(traj_ids)} trajectories: {traj_ids[:10]}{'...' if len(traj_ids) > 10 else ''}")

    # Run evaluation
    logger.info("\nRunning evaluation...")
    all_metrics = []
    final_action_keys = None  # Store for summary

    for traj_id in traj_ids:
        if traj_id >= len(loader):
            logger.warning(f"  Skipping trajectory {traj_id} (out of range)")
            continue

        state_joints, gt_actions, pred_actions, metrics, state_keys, action_keys = evaluate_trajectory(
            policy, loader, traj_id, EmbodimentTag.NEW_EMBODIMENT,
            action_horizon=ACTION_HORIZON, max_steps=STEPS_PER_TRAJ
        )

        metrics["traj_id"] = traj_id
        all_metrics.append(metrics)
        final_action_keys = action_keys  # Keep track for summary

        logger.info(f"  Traj {traj_id}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}")

        # Generate plot (matches official gr00t/eval/open_loop_eval.py format)
        if not args.no_plots:
            plot_path = output_dir / f"traj_{traj_id:04d}.png"
            plot_trajectory_results(
                state_joints, gt_actions, pred_actions, traj_id,
                ACTION_HORIZON, str(plot_path), state_keys, action_keys
            )

    # Aggregate metrics
    if all_metrics:
        avg_mse = np.mean([m["mse"] for m in all_metrics])
        avg_mae = np.mean([m["mae"] for m in all_metrics])
        std_mse = np.std([m["mse"] for m in all_metrics])
        std_mae = np.std([m["mae"] for m in all_metrics])

        # Per-joint aggregation
        all_joint_mse = np.array([m["per_joint_mse"] for m in all_metrics])
        all_joint_mae = np.array([m["per_joint_mae"] for m in all_metrics])
        avg_joint_mse = np.mean(all_joint_mse, axis=0)
        avg_joint_mae = np.mean(all_joint_mae, axis=0)

        summary = {
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "num_trajectories": len(all_metrics),
            "avg_mse": float(avg_mse),
            "std_mse": float(std_mse),
            "avg_mae": float(avg_mae),
            "std_mae": float(std_mae),
            "per_joint_avg_mse": avg_joint_mse.tolist(),
            "per_joint_avg_mae": avg_joint_mae.tolist(),
            "action_keys": final_action_keys,
            "trajectories": all_metrics,
        }

        # Save metrics
        metrics_path = output_dir / "evaluation_results.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nMetrics saved to: {metrics_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"  Trajectories evaluated: {len(all_metrics)}")
        print(f"  Overall MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        print(f"  Overall MAE: {avg_mae:.6f} ± {std_mae:.6f}")
        print()
        print(f"  Per-Action-Key MSE ({', '.join(final_action_keys)}):")
        for i in range(len(avg_joint_mse)):
            print(f"    Action {i}: {avg_joint_mse[i]:.6f}")
        print()

        # Interpretation
        if avg_mse < 0.01:
            print("  RESULT: EXCELLENT - Model should work well on robot")
        elif avg_mse < 0.05:
            print("  RESULT: GOOD - Model should work, may need some tuning")
        else:
            print("  RESULT: POOR - Investigate training or data issues")

        print("=" * 70)

        if not args.no_plots:
            print(f"\nPlots saved to: {output_dir}")

    else:
        logger.error("No trajectories were evaluated!")
        sys.exit(1)


if __name__ == "__main__":
    main()
