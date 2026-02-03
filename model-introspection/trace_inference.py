#!/usr/bin/env python3
"""
Enhanced Inference Tracer for SmolVLA Hallucination Investigation.

This script EXTENDS infer_smolvla_bimanual.py with additional tracing capabilities.
It imports all core functionality from the main inference script to ensure consistency.

Added features over main inference script:
- Per-step JSONL trace file (states, actions, timing, chunk boundaries)
- Configurable image capture at intervals or events
- Dynamic task modification support for language ablation experiments

Usage:
    # Basic tracing (outputs to logs/investigation/)
    python trace_inference.py \
        --checkpoint outputs/smolvla_bimanual/checkpoints/040000/pretrained_model \
        --task-key left_yogurt_bin

    # With image capture and dynamic task
    python trace_inference.py \
        --checkpoint outputs/smolvla_bimanual \
        --task-key left_yogurt_bin \
        --capture-images \
        --capture-interval 30 \
        --dynamic-task \
        --completion-phrase "Task complete. Hold position." \
        --completion-step 180

Output (in logs/case_<run_id>/):
    ├── metadata.json       # Run configuration and summary
    ├── trace.jsonl         # Per-step trace data
    ├── trace.log           # Execution log
    └── images/             # Captured images (if enabled)
"""

import argparse
import json
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# ============================================================================
# IMPORT FROM MAIN INFERENCE SCRIPT (ensures consistency)
# ============================================================================

# Add project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]  # tools -> investigation -> scripts -> jdocs -> project
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Add bimanual script dir to import from main inference script
BIMANUAL_SCRIPT_DIR = PROJECT_ROOT / "jdocs" / "scripts" / "bimanual"
sys.path.insert(0, str(BIMANUAL_SCRIPT_DIR))

# Import core components from main inference script
from infer_smolvla_bimanual import (
    # Constants
    DEFAULT_CHECKPOINT,
    DEVICE,
    DATASET_PATH,
    TASK_EXAMPLES,
    ACTION_INTERVAL,
    TOTAL_DIM,
    HARDWARE_CONFIG_CENTRAL,
    # Classes
    CameraManager,
    BimanualRobotController,
    # Functions
    load_hardware_config,
    format_observation,
    setup_logging,
    # Dynamic task (added by us)
    detect_task_completion,
)

# Global state
running = True

def _signal_handler(sig, frame):
    global running
    print("\nShutdown requested...")
    running = False

# ============================================================================
# OUTPUT PATHS (use logs/ directory directly)
# ============================================================================

LOG_BASE_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# TRACE DATA STRUCTURES
# ============================================================================

@dataclass
class TraceEntry:
    """Single inference step trace data."""
    step: int
    timestamp: float
    elapsed_s: float

    # Robot state
    state_raw: list  # 12D degrees
    state_normalized: list  # 12D normalized (what policy sees)

    # Actions
    action_raw: list  # 12D normalized (policy output)
    action_final: list  # 12D degrees (sent to robot)

    # Task info
    task: str
    task_modified: bool

    # Chunk tracking
    chunk_index: int
    chunk_step: int  # 0-49 within chunk
    is_new_chunk: bool

    # Timing
    inference_time_ms: float

    # Metrics
    action_delta_max: float = 0.0
    gripper_left: float = 0.0
    gripper_right: float = 0.0


@dataclass
class RunMetadata:
    """Metadata for a tracing run."""
    run_id: str
    timestamp: str
    checkpoint: str
    task_key: str
    task_original: str
    duration_s: float
    total_steps: int
    device: str
    dynamic_task_enabled: bool
    completion_phrase: Optional[str]
    completion_step: Optional[int]
    capture_images: bool
    capture_interval: int
    notes: str


# ============================================================================
# TRACER CLASS
# ============================================================================

class InferenceTracer:
    """Manages tracing during SmolVLA inference."""

    def __init__(
        self,
        output_dir: Path,
        capture_images: bool = False,
        capture_interval: int = 50,
        logger=None,
    ):
        self.output_dir = output_dir
        self.capture_images = capture_images
        self.capture_interval = capture_interval
        self.logger = logger

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if capture_images:
            (self.output_dir / "images").mkdir(exist_ok=True)

        self.trace_file = open(self.output_dir / "trace.jsonl", "w")
        self.entries: list[TraceEntry] = []

        self.chunk_index = 0
        self.chunk_step = 0
        self.last_action_queue_len = 0

    def record_step(
        self,
        step: int,
        start_time: float,
        state_raw: np.ndarray,
        state_normalized: np.ndarray,
        action_raw: np.ndarray,
        action_final: np.ndarray,
        task: str,
        task_modified: bool,
        inference_time: float,
        policy_action_queue_len: int,
        images: Optional[dict] = None,
    ) -> TraceEntry:
        """Record a single inference step."""
        now = time.time()

        # Detect new chunk (action queue refilled)
        is_new_chunk = policy_action_queue_len > self.last_action_queue_len
        if is_new_chunk:
            self.chunk_index += 1 if step > 0 else 0
            self.chunk_step = 0
        else:
            self.chunk_step += 1

        self.last_action_queue_len = policy_action_queue_len

        # Compute metrics
        action_delta = np.abs(action_final - state_raw)

        entry = TraceEntry(
            step=step,
            timestamp=now,
            elapsed_s=now - start_time,
            state_raw=state_raw.tolist(),
            state_normalized=state_normalized.tolist() if state_normalized is not None else [],
            action_raw=action_raw.tolist() if action_raw is not None else [],
            action_final=action_final.tolist(),
            task=task,
            task_modified=task_modified,
            chunk_index=self.chunk_index,
            chunk_step=self.chunk_step,
            is_new_chunk=is_new_chunk,
            inference_time_ms=inference_time * 1000,
            action_delta_max=float(np.max(action_delta)),
            gripper_left=float(action_final[5]),
            gripper_right=float(action_final[11]),
        )

        self.entries.append(entry)
        self.trace_file.write(json.dumps(asdict(entry)) + "\n")
        self.trace_file.flush()

        # Capture images if enabled
        if self.capture_images and images is not None:
            should_capture = (
                step % self.capture_interval == 0 or
                is_new_chunk or
                task_modified
            )
            if should_capture:
                self._save_images(step, images)

        return entry

    def _save_images(self, step: int, images: dict):
        """Save images to disk."""
        img_dir = self.output_dir / "images"
        for name, frame in images.items():
            # Convert RGB to BGR for cv2
            if len(frame.shape) == 3 and frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            path = img_dir / f"step_{step:04d}_{name}.jpg"
            cv2.imwrite(str(path), frame_bgr)

    def save_metadata(self, metadata: RunMetadata):
        """Save run metadata."""
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def close(self):
        """Close trace file."""
        self.trace_file.close()

    def get_summary(self) -> dict:
        """Get summary statistics."""
        if not self.entries:
            return {}

        return {
            "total_steps": len(self.entries),
            "total_chunks": self.chunk_index + 1,
            "duration_s": self.entries[-1].elapsed_s,
            "avg_inference_ms": np.mean([e.inference_time_ms for e in self.entries]),
            "max_action_delta": max(e.action_delta_max for e in self.entries),
            "task_modifications": sum(1 for e in self.entries if e.task_modified),
            "new_chunks": sum(1 for e in self.entries if e.is_new_chunk),
        }


# ============================================================================
# TRACED INFERENCE LOOP
# ============================================================================

def run_traced_inference(
    policy,
    preprocessor,
    postprocessor,
    cameras: CameraManager,
    robot: BimanualRobotController,
    tracer: InferenceTracer,
    original_task: str,
    max_duration: float,
    dynamic_task_config: dict,
    device: str,
    dry_run: bool,
    logger,
):
    """
    Run inference with comprehensive tracing.

    This mirrors infer_smolvla_bimanual.py's run_inference_loop() but adds tracing.
    """
    global running

    logger.info(f"Starting traced inference (max {max_duration}s)")
    logger.info(f"Task: {original_task}")
    logger.info(f"Dynamic task: {dynamic_task_config.get('dynamic_task_enabled', False)}")

    start_time = time.time()
    step = 0
    action_history = []
    completion_triggered = False
    current_task = original_task

    policy.reset()

    while running and (time.time() - start_time) < max_duration:
        loop_start = time.time()

        # Capture images and state (same as main inference)
        images = cameras.capture()
        state_raw = robot.get_state()

        # Dynamic task handling
        task_modified_now = False
        if dynamic_task_config.get("dynamic_task_enabled", False) and not completion_triggered:
            if detect_task_completion(
                step=step,
                state=state_raw,
                action_history=action_history,
                trigger_type=dynamic_task_config.get("completion_trigger", "step_count"),
                trigger_step=dynamic_task_config.get("completion_step", 180),
            ):
                completion_triggered = True
                task_modified_now = True
                current_task = dynamic_task_config.get("completion_phrase", "Hold position.")
                logger.info(f"Step {step}: Task modified to: {current_task}")
                policy.reset()  # Force KV cache recompute

        # Format observation (using imported function)
        observation = format_observation(images, state_raw, current_task, device)
        preprocessed = preprocessor(observation)

        # Get normalized state for tracing
        state_normalized = None
        if "observation.state" in preprocessed:
            state_tensor = preprocessed["observation.state"]
            if isinstance(state_tensor, torch.Tensor):
                state_normalized = state_tensor.squeeze().cpu().numpy()

        # Inference
        inf_start = time.time()
        with torch.inference_mode():
            raw_action = policy.select_action(preprocessed)
        inf_time = time.time() - inf_start

        # Get action queue length for chunk tracking
        action_queue_len = len(policy._queues.get("action", [])) if hasattr(policy, '_queues') else 0

        # Get raw action for tracing
        action_raw = None
        if isinstance(raw_action, torch.Tensor):
            action_raw = raw_action.squeeze().cpu().numpy()
        elif isinstance(raw_action, dict) and "action" in raw_action:
            act = raw_action["action"]
            if isinstance(act, torch.Tensor):
                action_raw = act.squeeze().cpu().numpy()

        # Postprocess (same as main inference)
        action_final = postprocessor(raw_action)
        if isinstance(action_final, torch.Tensor):
            action_final = action_final.squeeze().cpu().numpy()
        elif isinstance(action_final, dict):
            action_final = action_final.get("action", action_final)
            if isinstance(action_final, torch.Tensor):
                action_final = action_final.squeeze().cpu().numpy()

        # Trim to 12D if needed
        if len(action_final) > TOTAL_DIM:
            action_final = action_final[:TOTAL_DIM]

        action_history.append(action_final.copy())

        # Record trace
        entry = tracer.record_step(
            step=step,
            start_time=start_time,
            state_raw=state_raw,
            state_normalized=state_normalized,
            action_raw=action_raw,
            action_final=action_final,
            task=current_task,
            task_modified=task_modified_now,
            inference_time=inf_time,
            policy_action_queue_len=action_queue_len,
            images=images if tracer.capture_images else None,
        )

        # Log periodically
        if step % 30 == 0:
            logger.info(
                f"Step {step}: chunk={entry.chunk_index}, "
                f"chunk_step={entry.chunk_step}, "
                f"inf={entry.inference_time_ms:.1f}ms, "
                f"delta_max={entry.action_delta_max:.1f}"
            )

        # Execute action (same as main inference)
        if not dry_run:
            robot.send_action(action_final)

        step += 1

        # Rate control
        elapsed = time.time() - loop_start
        sleep_time = ACTION_INTERVAL - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    summary = tracer.get_summary()
    logger.info("\n" + "=" * 50)
    logger.info("Trace Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Inference Tracer for SmolVLA Hallucination Investigation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic trace collection (images captured by default)
  python trace_inference.py --task-key left_yogurt_bin

  # With custom task string
  python trace_inference.py --task "Use left arm to pick up the orange"

  # With language ablation (inject completion phrase at step 180)
  python trace_inference.py -k left_yogurt_bin --dynamic-task --completion-step 180

  # Dry run (no robot commands)
  python trace_inference.py -k left_yogurt_bin --dry-run

Output: logs/case_<timestamp>/
        ├── metadata.json   # Run config and notes
        ├── trace.jsonl     # Per-step data
        └── images/         # Camera frames
"""
    )

    # ---- Task Selection ----
    task_group = parser.add_argument_group("Task Selection")
    task_group.add_argument(
        "--task-key", "-k",
        choices=list(TASK_EXAMPLES.keys()),
        default="left_yogurt_bin",
        help="Predefined task key (default: left_yogurt_bin). See TASK_EXAMPLES in script."
    )
    task_group.add_argument(
        "--task", "-t",
        help="Custom task string. Overrides --task-key if provided."
    )

    # ---- Image Capture ----
    capture_group = parser.add_argument_group("Image Capture")
    capture_group.add_argument(
        "--no-capture-images",
        action="store_true",
        help="Disable image capture. By default, images ARE captured."
    )
    capture_group.add_argument(
        "--capture-interval",
        type=int,
        default=50,
        help="Save image every N steps (default: 50 = ~1.7s at 30Hz)."
    )

    # ---- Language Ablation ----
    ablation_group = parser.add_argument_group(
        "Language Ablation",
        "Inject completion phrase mid-inference to test if language affects hallucination."
    )
    ablation_group.add_argument(
        "--dynamic-task",
        action="store_true",
        help="Enable language ablation. At --completion-step, inject --completion-phrase."
    )
    ablation_group.add_argument(
        "--completion-phrase",
        default="Task complete. Hold position.",
        help="Phrase to inject (default: 'Task complete. Hold position.')."
    )
    ablation_group.add_argument(
        "--completion-step",
        type=int,
        default=180,
        help="Step number to inject phrase (default: 180). Observe when task completes and set accordingly."
    )

    # ---- Output Organization ----
    output_group = parser.add_argument_group(
        "Output Organization",
        "Trace files saved to: logs/case_<name>/"
    )
    output_group.add_argument(
        "--case-name",
        help="Case folder name. Default: auto-generated timestamp (YYYYMMDD_HHMMSS)."
    )
    output_group.add_argument(
        "--notes",
        default="",
        help="Free-text notes saved to metadata.json. Describe scene setup, distractor positions, etc."
    )

    # ---- Run Options ----
    run_group = parser.add_argument_group("Run Options")
    run_group.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Max inference time in seconds (default: 60)."
    )
    run_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending actions to robot. For testing."
    )
    run_group.add_argument(
        "--checkpoint", "-c",
        default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint path (default: {DEFAULT_CHECKPOINT})."
    )
    run_group.add_argument("--hw-config", help="Hardware config YAML path.")
    run_group.add_argument("--dataset", default=DATASET_PATH, help="Dataset path for normalization stats.")
    run_group.add_argument("--device", default=DEVICE, help="Device: cuda or cpu.")

    args = parser.parse_args()

    # Determine task
    if args.task:
        task = args.task
        task_key = "custom"
    else:
        task = TASK_EXAMPLES[args.task_key]
        task_key = args.task_key

    # Setup output directory (in logs/)
    run_id = args.case_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = LOG_BASE_DIR / f"case_{run_id}"

    # Setup signal handler
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Setup logging (using imported function)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "trace.log"
    logger = setup_logging(log_file)

    # Determine capture mode (images captured by default)
    capture_images = not args.no_capture_images

    logger.info("=" * 60)
    logger.info("SmolVLA Enhanced Inference Tracer")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Task: {task}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Capture images: {capture_images}")
    logger.info(f"Dynamic task: {args.dynamic_task}")
    if args.notes:
        logger.info(f"Notes: {args.notes}")
    tracer = InferenceTracer(
        output_dir=output_dir,
        capture_images=capture_images,
        capture_interval=args.capture_interval,
        logger=logger,
    )

    # Dynamic task config (step_count only - other triggers removed as unreliable)
    dynamic_task_config = {
        "dynamic_task_enabled": args.dynamic_task,
        "completion_phrase": args.completion_phrase,
        "completion_trigger": "step_count",  # Only reliable method
        "completion_step": args.completion_step,
    }

    cameras = None
    robot = None
    summary = {}
    run_error = None

    try:
        # Load hardware config (using imported function)
        hw_config = load_hardware_config(args.hw_config)

        # Load dataset metadata
        logger.info("Loading dataset metadata...")
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        dataset_name = Path(args.dataset).name
        dataset_metadata = LeRobotDatasetMetadata(repo_id=dataset_name, root=args.dataset)

        # Load policy
        logger.info("Loading SmolVLA policy...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        policy = SmolVLAPolicy.from_pretrained(str(checkpoint_path))
        policy.eval()
        policy.to(args.device)
        logger.info(f"Policy loaded: chunk_size={policy.config.chunk_size}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint_path),
            dataset_stats=dataset_metadata.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )

        # Initialize hardware (using imported classes)
        logger.info("Initializing hardware...")
        cameras = CameraManager(hw_config)
        robot = BimanualRobotController(hw_config)

        if args.dry_run:
            logger.info("DRY RUN mode - no robot commands")
            robot.robot = None

        # Run traced inference
        summary = run_traced_inference(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            cameras=cameras,
            robot=robot,
            tracer=tracer,
            original_task=task,
            max_duration=args.duration,
            dynamic_task_config=dynamic_task_config,
            device=args.device,
            dry_run=args.dry_run,
            logger=logger,
        )

        logger.info(f"\nResults saved to: {output_dir}")
        run_error = None

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        summary = {"duration_s": 0, "total_steps": 0}
        run_error = "KeyboardInterrupt"

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        summary = {"duration_s": 0, "total_steps": 0}
        run_error = str(e)

    finally:
        # Always save metadata (even on error)
        try:
            metadata = RunMetadata(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                checkpoint=str(args.checkpoint),
                task_key=task_key,
                task_original=task,
                duration_s=summary.get("duration_s", 0),
                total_steps=summary.get("total_steps", 0),
                device=args.device,
                dynamic_task_enabled=args.dynamic_task,
                completion_phrase=args.completion_phrase if args.dynamic_task else None,
                completion_step=args.completion_step if args.dynamic_task else None,
                capture_images=capture_images,
                capture_interval=args.capture_interval,
                notes=args.notes + (f" [ERROR: {run_error}]" if run_error else ""),
            )
            tracer.save_metadata(metadata)
        except Exception as meta_err:
            logger.error(f"Failed to save metadata: {meta_err}")

        tracer.close()
        if cameras:
            cameras.release()
        if robot:
            robot.disconnect()
        logger.info("Done")


if __name__ == "__main__":
    main()
