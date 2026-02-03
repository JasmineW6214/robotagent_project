#!/usr/bin/env python3
"""
Robot Dataset Visualizer - LeRobot v2/v3 Format Support

A Gradio-based web application for visualizing robot arm datasets
with synchronized camera views and time-series plots.

Features:
- Support for LeRobot v2 and v3 dataset formats
- Sidebar layout: dataset browser on left, display on right
- Compact view: cameras + charts visible without scrolling
- Synchronized video playback with joint data plots
- Continuous playback with play/pause controls

Usage:
    python visualize_dataset_v2.py [OPTIONS]

Options:
    --dataset PATH      Path to dataset (default: current directory)
    --port INT          Gradio server port (default: 7860)
    --share             Create a public link
    --format v2|v3      Dataset format version (default: v2)
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("visualizer")
logger.setLevel(logging.INFO)

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetMetadata:
    """Container for dataset-level metadata."""
    dataset_path: str
    format_version: str  # "v2" or "v3"
    fps: int
    total_episodes: int
    robot_type: str
    joint_names: List[str]
    video_keys: List[str]
    video_path_template: str
    data_path_template: str
    episodes: List[Dict]
    tasks: List[str]
    # v3-specific: detailed episode metadata with file indices and video timestamps
    episodes_v3_meta: Dict[int, Dict] = None  # episode_index -> metadata dict


@dataclass
class EpisodeData:
    """Container for a single episode's data."""
    episode_index: int
    task: str
    length: int
    timestamps: np.ndarray
    actions: np.ndarray
    states: np.ndarray


def video_key_to_label(video_key: str) -> str:
    """Convert a video key to a friendly display label.

    Examples:
        "observation.images.head" -> "Head Camera"
        "observation.images.left_wrist" -> "Left Wrist Camera"
        "head" -> "Head Camera"
        "wrist" -> "Wrist Camera"
    """
    # Remove common prefixes
    key = video_key
    if key.startswith("observation.images."):
        key = key[len("observation.images."):]

    # Convert underscores to spaces and title case
    label = key.replace("_", " ").title()

    # Add "Camera" suffix if not already present
    if not label.lower().endswith("camera"):
        label = f"{label} Camera"

    return label


# =============================================================================
# Dataset Loading - v2 and v3 format support
# =============================================================================

def load_v3_episodes_metadata(dataset_path: str) -> Dict[int, Dict]:
    """Load v3 episode metadata from parquet files in meta/episodes/.

    Returns a dict mapping episode_index to metadata including:
    - data/chunk_index, data/file_index: which parquet file contains the data
    - videos/{key}/chunk_index, file_index, from_timestamp, to_timestamp: video info
    - length: number of frames in episode
    - tasks: task description
    """
    path = Path(dataset_path)
    episodes_dir = path / "meta" / "episodes"

    if not episodes_dir.exists():
        logger.warning(f"v3 episodes metadata dir not found: {episodes_dir}")
        return {}

    episodes_meta = {}

    # Iterate through chunk directories
    for chunk_dir in sorted(episodes_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        # Load each parquet file in the chunk
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            try:
                df = pq.read_table(parquet_file).to_pandas()

                for _, row in df.iterrows():
                    ep_idx = int(row["episode_index"])
                    episodes_meta[ep_idx] = row.to_dict()

            except Exception as e:
                logger.warning(f"Failed to load v3 episode metadata from {parquet_file}: {e}")

    logger.info(f"Loaded v3 metadata for {len(episodes_meta)} episodes")
    return episodes_meta


def detect_format_version(dataset_path: str) -> str:
    """Auto-detect dataset format version."""
    path = Path(dataset_path)
    info_path = path / "meta" / "info.json"

    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    version = info.get("codebase_version", "v2.0")
    if version.startswith("v2"):
        return "v2"
    elif version.startswith("v3"):
        return "v3"
    else:
        # Default to v2 for older formats
        return "v2"


def load_dataset_metadata(dataset_path: str, format_version: str = "auto") -> DatasetMetadata:
    """Load dataset metadata supporting both v2 and v3 formats."""
    path = Path(dataset_path)
    meta_path = path / "meta"

    # Auto-detect format if needed
    if format_version == "auto":
        format_version = detect_format_version(dataset_path)

    logger.info(f"Loading dataset from {dataset_path} (format: {format_version})")

    # Load info.json
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)

    # Load tasks.jsonl
    tasks = []
    tasks_file = meta_path / "tasks.jsonl"
    if tasks_file.exists():
        with open(tasks_file, "r") as f:
            for line in f:
                if line.strip():
                    task_data = json.loads(line.strip())
                    tasks.append(task_data.get("task", ""))

    # Load episodes - different for v2 vs v3
    episodes = []
    episodes_v3_meta = None

    if format_version == "v3":
        # v3: load detailed metadata from parquet files
        episodes_v3_meta = load_v3_episodes_metadata(dataset_path)

        # Build episodes list from v3 metadata
        for ep_idx in sorted(episodes_v3_meta.keys()):
            ep_meta = episodes_v3_meta[ep_idx]
            task_list = ep_meta.get("tasks", [])
            task_str = task_list[0] if isinstance(task_list, list) and len(task_list) > 0 else str(task_list)
            episodes.append({
                "episode_index": ep_idx,
                "task_index": 0,  # v3 stores task directly
                "task": task_str,
                "length": ep_meta.get("length", 0),
            })

        # Update tasks list from v3 metadata if needed
        if not tasks:
            seen_tasks = set()
            for ep in episodes:
                task = ep.get("task", "")
                if task and task not in seen_tasks:
                    tasks.append(task)
                    seen_tasks.add(task)
    else:
        # v2: load from episodes.jsonl
        episodes_file = meta_path / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file, "r") as f:
                for line in f:
                    if line.strip():
                        ep_data = json.loads(line.strip())
                        episodes.append(ep_data)

    # Extract joint names
    joint_names = []
    if "features" in info and "action" in info["features"]:
        joint_names = info["features"]["action"].get("names", [])
    if not joint_names:
        joint_names = [
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"
        ]

    # Extract video keys from features
    video_keys = []
    for key in info.get("features", {}):
        if key.startswith("observation.images."):
            video_keys.append(key)

    # Get path templates (stored for reference, but v3 uses episode metadata for actual paths)
    video_path_template = info.get("video_path", "")
    data_path_template = info.get("data_path", "")

    return DatasetMetadata(
        dataset_path=dataset_path,
        format_version=format_version,
        fps=info.get("fps", 30),
        total_episodes=info.get("total_episodes", len(episodes)),
        robot_type=info.get("robot_type", "unknown"),
        joint_names=joint_names,
        video_keys=video_keys,
        video_path_template=video_path_template,
        data_path_template=data_path_template,
        episodes=episodes,
        tasks=tasks,
        episodes_v3_meta=episodes_v3_meta,
    )


def get_video_path_v2(metadata: DatasetMetadata, episode_index: int, video_key: str) -> Path:
    """Get the video file path for an episode and camera (v2 format)."""
    path = Path(metadata.dataset_path)
    chunk_id = episode_index // 1000

    # Format the template with v2 variable names
    video_path = metadata.video_path_template.format(
        episode_chunk=chunk_id,
        video_key=video_key,
        episode_index=episode_index,
    )

    return path / video_path


def get_video_path_v3(metadata: DatasetMetadata, episode_index: int, video_key: str) -> Tuple[Path, float, float]:
    """Get the video file path and timestamp range for an episode (v3 format).

    Returns:
        Tuple of (video_path, from_timestamp, to_timestamp)
    """
    path = Path(metadata.dataset_path)

    if metadata.episodes_v3_meta is None or episode_index not in metadata.episodes_v3_meta:
        raise ValueError(f"No v3 metadata for episode {episode_index}")

    ep_meta = metadata.episodes_v3_meta[episode_index]

    # Get video-specific metadata
    chunk_key = f"videos/{video_key}/chunk_index"
    file_key = f"videos/{video_key}/file_index"
    from_ts_key = f"videos/{video_key}/from_timestamp"
    to_ts_key = f"videos/{video_key}/to_timestamp"

    chunk_index = int(ep_meta.get(chunk_key, 0))
    file_index = int(ep_meta.get(file_key, 0))
    from_timestamp = float(ep_meta.get(from_ts_key, 0.0))
    to_timestamp = float(ep_meta.get(to_ts_key, 0.0))

    # Build path: videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
    video_path = path / "videos" / video_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"

    return video_path, from_timestamp, to_timestamp


def load_episode_data(metadata: DatasetMetadata, episode_index: int) -> EpisodeData:
    """Load action and state data for a specific episode."""
    path = Path(metadata.dataset_path)

    if pq is None:
        raise ImportError("pyarrow is required for loading parquet files")

    if metadata.format_version == "v3":
        # v3: get file location from episode metadata
        if metadata.episodes_v3_meta is None or episode_index not in metadata.episodes_v3_meta:
            raise ValueError(f"No v3 metadata for episode {episode_index}")

        ep_meta = metadata.episodes_v3_meta[episode_index]
        chunk_index = int(ep_meta.get("data/chunk_index", 0))
        file_index = int(ep_meta.get("data/file_index", 0))

        parquet_path = path / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
    else:
        # v2: calculate path from episode index
        chunk_id = episode_index // 1000
        parquet_path = path / metadata.data_path_template.format(
            episode_chunk=chunk_id,
            episode_index=episode_index,
        )

    # Read parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Filter by episode
    if "episode_index" in df.columns:
        ep_col = df["episode_index"]
        if hasattr(ep_col.iloc[0], '__len__'):
            df = df[[e[0] == episode_index if len(e) > 0 else False for e in ep_col]]
        else:
            df = df[ep_col == episode_index]
        df = df.reset_index(drop=True)

    # Extract data
    actions = np.array(df["action"].tolist())
    states = np.array(df["observation.state"].tolist())

    # Get timestamps
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].values
        if isinstance(timestamps[0], (list, np.ndarray)):
            timestamps = np.array([t[0] if len(t) > 0 else 0.0 for t in timestamps])
    else:
        timestamps = np.arange(len(actions)) / metadata.fps

    # Get task
    task = ""
    for ep in metadata.episodes:
        if ep.get("episode_index") == episode_index:
            # v3 stores task directly, v2 uses task_index
            if "task" in ep:
                task = ep["task"]
            else:
                task_idx = ep.get("task_index", 0)
                if task_idx < len(metadata.tasks):
                    task = metadata.tasks[task_idx]
            break

    return EpisodeData(
        episode_index=episode_index,
        task=task,
        length=len(actions),
        timestamps=timestamps.astype(np.float32),
        actions=actions.astype(np.float32),
        states=states.astype(np.float32),
    )


def resize_frame(img_array: np.ndarray, max_width: int = 320) -> np.ndarray:
    """Resize frame to reduce bandwidth while maintaining aspect ratio."""
    if img_array is None:
        logger.warning("resize_frame: received None image")
        return None

    h, w = img_array.shape[:2]
    logger.debug(f"resize_frame: input shape={img_array.shape}, dtype={img_array.dtype}, max_width={max_width}")

    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        img = Image.fromarray(img_array.astype(np.uint8))
        img = img.resize((max_width, new_h), Image.Resampling.BILINEAR)
        result = np.array(img)
        logger.debug(f"resize_frame: resized to {result.shape}")
        return result
    return img_array


def extract_video_frame(video_path: Path, frame_index: int) -> Optional[np.ndarray]:
    """Extract a single frame from video."""
    if not video_path.exists():
        return None

    # Try av library first (handles AV1 codec)
    if HAS_AV:
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]

            for i, frame in enumerate(container.decode(video=0)):
                if i == frame_index:
                    result = frame.to_ndarray(format="rgb24")
                    container.close()
                    return result
                if i > frame_index:
                    break

            container.close()
        except Exception as e:
            print(f"av failed for {video_path}: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"cv2 failed for {video_path}: {e}")

    return None


def extract_all_frames(video_path: Path) -> Optional[np.ndarray]:
    """Extract all frames from a video file."""
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None

    # Try av library first (handles AV1 codec)
    if HAS_AV:
        try:
            container = av.open(str(video_path))
            frames = []

            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

            container.close()

            if frames:
                return np.array(frames)
        except Exception as e:
            print(f"av failed for {video_path}: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(video_path))
            frames = []

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

            if frames:
                return np.array(frames)
        except Exception as e:
            print(f"cv2 failed for {video_path}: {e}")

    return None


def extract_frames_by_timestamp(video_path: Path, from_timestamp: float, to_timestamp: float, fps: int = 30) -> Optional[np.ndarray]:
    """Extract frames from a video file within a specific timestamp range (for v3 concatenated videos).

    Args:
        video_path: Path to the video file
        from_timestamp: Start timestamp in seconds
        to_timestamp: End timestamp in seconds
        fps: Frames per second of the video

    Returns:
        numpy array of frames or None if extraction fails
    """
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return None

    start_frame = int(from_timestamp * fps)
    end_frame = int(to_timestamp * fps)

    logger.debug(f"Extracting frames {start_frame}-{end_frame} from {video_path} (timestamps {from_timestamp:.2f}-{to_timestamp:.2f}s)")

    # Try av library first (handles AV1 codec)
    if HAS_AV:
        try:
            container = av.open(str(video_path))
            frames = []
            frame_count = 0

            for frame in container.decode(video=0):
                if frame_count >= start_frame and frame_count < end_frame:
                    frames.append(frame.to_ndarray(format="rgb24"))
                elif frame_count >= end_frame:
                    break
                frame_count += 1

            container.close()

            if frames:
                logger.debug(f"Extracted {len(frames)} frames using av")
                return np.array(frames)
        except Exception as e:
            print(f"av failed for {video_path}: {e}")

    # Fallback to OpenCV
    if HAS_CV2:
        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []

            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

            if frames:
                logger.debug(f"Extracted {len(frames)} frames using cv2")
                return np.array(frames)
        except Exception as e:
            print(f"cv2 failed for {video_path}: {e}")

    return None


# =============================================================================
# Plotting - Combined stacked layout with shared x-axis
# =============================================================================

JOINT_NAMES_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_NAME_GRIPPER = "gripper"

COLORS = {
    "shoulder_pan": "#ef4444",    # Red
    "shoulder_lift": "#22c55e",   # Green
    "elbow_flex": "#3b82f6",      # Blue
    "wrist_flex": "#f59e0b",      # Amber
    "wrist_roll": "#8b5cf6",      # Purple
    "gripper": "#ef4444",         # Red
    "state": "solid",
    "action": "dash",
    "marker": "#f97316",          # Orange
}


def generate_combined_plot(
    episode: EpisodeData,
    current_frame: int = 0,
    height: int = 350,
) -> go.Figure:
    """Generate combined arm joints + gripper plot with shared x-axis."""

    # Create subplots: 2 rows, shared x-axis
    # Row 1: Arm joints (taller), Row 2: Gripper (shorter)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=("Arm Joints", "Gripper"),
    )

    x_axis = np.arange(episode.length)
    num_joints = min(5, episode.states.shape[1])

    # Row 1: Arm joints
    for i, joint_name in enumerate(JOINT_NAMES_ARM[:num_joints]):
        color = COLORS.get(joint_name, "#3b82f6")

        # State line (solid)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.states[:, i],
                mode="lines",
                name=f"{joint_name}",
                line=dict(color=color, width=1.5),
                legendgroup=joint_name,
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Action line (dashed)
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=episode.actions[:, i],
                mode="lines",
                name=f"{joint_name} (action)",
                line=dict(color=color, width=1.5, dash="dash"),
                legendgroup=joint_name,
                showlegend=False,
            ),
            row=1, col=1,
        )

    # Row 2: Gripper
    gripper_idx = 5  # Gripper is typically the 6th joint (index 5)
    if episode.states.shape[1] <= gripper_idx:
        gripper_idx = episode.states.shape[1] - 1

    gripper_color = COLORS["gripper"]

    # State line (solid)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=episode.states[:, gripper_idx],
            mode="lines",
            name="gripper",
            line=dict(color=gripper_color, width=2),
            showlegend=True,
        ),
        row=2, col=1,
    )

    # Action line (dashed)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=episode.actions[:, gripper_idx],
            mode="lines",
            name="gripper (action)",
            line=dict(color=gripper_color, width=2, dash="dash"),
            showlegend=False,
        ),
        row=2, col=1,
    )

    # Add vertical marker on both subplots
    fig.add_vline(
        x=current_frame,
        line_dash="solid",
        line_color=COLORS["marker"],
        line_width=2,
        row=1, col=1,
    )
    fig.add_vline(
        x=current_frame,
        line_dash="solid",
        line_color=COLORS["marker"],
        line_width=2,
        row=2, col=1,
    )

    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        template="plotly_dark",
        paper_bgcolor="#1f2937",
        plot_bgcolor="#1f2937",
    )

    # Update axes
    fig.update_yaxes(title_text="Degrees", row=1, col=1)
    fig.update_yaxes(title_text="Gripper", row=2, col=1)
    fig.update_xaxes(title_text="Frame", row=2, col=1)

    # Make subplot titles smaller
    for annotation in fig.layout.annotations:
        annotation.font.size = 10

    return fig


# =============================================================================
# Gradio Application
# =============================================================================

class VisualizerState:
    """Application state."""
    def __init__(self):
        self.metadata: Optional[DatasetMetadata] = None
        self.episode_data: Optional[EpisodeData] = None
        self.video_frames: Dict[str, np.ndarray] = {}
        self.current_frame: int = 0
        self.max_frame: int = 0  # Track max frame to avoid slider bounds issues
        self.is_playing: bool = False


def create_app(default_dataset: str = ".", default_format: str = "v2") -> gr.Blocks:
    """Create the Gradio application with sidebar layout."""

    state = VisualizerState()

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def load_dataset(dataset_path: str, format_version: str) -> Tuple:
        """Load dataset and return episode list."""
        try:
            fmt = format_version if format_version != "auto" else "auto"
            state.metadata = load_dataset_metadata(dataset_path, fmt)

            # Build episode choices
            choices = []
            for ep in state.metadata.episodes:
                idx = ep.get("episode_index", len(choices))
                # v3 stores task directly, v2 uses task_index
                if "task" in ep:
                    task = ep["task"]
                else:
                    task_idx = ep.get("task_index", 0)
                    task = state.metadata.tasks[task_idx] if task_idx < len(state.metadata.tasks) else ""
                label = f"Ep {idx}: {task[:30]}..." if len(task) > 30 else f"Ep {idx}: {task}" if task else f"Episode {idx}"
                choices.append((label, idx))

            status = f"Loaded {state.metadata.total_episodes} episodes ({state.metadata.format_version})"
            info = f"Robot: {state.metadata.robot_type} | FPS: {state.metadata.fps} | Cameras: {len(state.metadata.video_keys)}"

            return (
                gr.Dropdown(choices=choices, value=choices[0][1] if choices else None),
                status,
                info,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                gr.Dropdown(choices=[]),
                f"Error: {str(e)[:100]}",
                "",
            )

    def load_episode(episode_index: int) -> Tuple:
        """Load episode data and first frame."""
        if state.metadata is None or episode_index is None:
            state.max_frame = 0
            return (None, None, None, None, 0, gr.Slider(maximum=1), "No episode", "",
                    gr.update(label="Camera 1"), gr.update(label="Camera 2"), gr.update(label="Camera 3"))

        try:
            # Load data
            state.episode_data = load_episode_data(state.metadata, episode_index)
            state.current_frame = 0
            state.max_frame = state.episode_data.length - 1

            # Load video frames for each camera
            state.video_frames = {}
            for video_key in state.metadata.video_keys:
                if state.metadata.format_version == "v3":
                    # v3: get video path and timestamps, extract frames from time range
                    video_path, from_ts, to_ts = get_video_path_v3(state.metadata, episode_index, video_key)
                    frames = extract_frames_by_timestamp(video_path, from_ts, to_ts, state.metadata.fps)
                else:
                    # v2: simple per-episode video file
                    video_path = get_video_path_v2(state.metadata, episode_index, video_key)
                    frames = extract_all_frames(video_path)

                if frames is not None:
                    state.video_frames[video_key] = frames
                    logger.info(f"Loaded {len(frames)} frames for {video_key}")

            # Get first frames
            cam1_frame, cam2_frame, cam3_frame = get_frames_for_position(0)

            # Generate combined plot
            combined_plot = generate_combined_plot(state.episode_data, current_frame=0)

            frame_info = f"Frame 0/{state.max_frame}"
            task_info = state.episode_data.task

            # Generate dynamic camera labels based on actual video keys
            video_keys = list(state.video_frames.keys())
            cam1_label = video_key_to_label(video_keys[0]) if len(video_keys) >= 1 else "Camera 1"
            cam2_label = video_key_to_label(video_keys[1]) if len(video_keys) >= 2 else "Camera 2"
            cam3_label = video_key_to_label(video_keys[2]) if len(video_keys) >= 3 else "Camera 3"

            return (
                cam1_frame,
                cam2_frame,
                cam3_frame,
                combined_plot,
                0,
                gr.Slider(value=0, maximum=state.max_frame),
                frame_info,
                task_info,
                gr.update(label=cam1_label),
                gr.update(label=cam2_label),
                gr.update(label=cam3_label),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            state.max_frame = 0
            return (None, None, None, None, 0, gr.Slider(maximum=1), f"Error: {e}", "",
                    gr.update(label="Camera 1"), gr.update(label="Camera 2"), gr.update(label="Camera 3"))

    def update_frame(frame_idx: int) -> Tuple:
        """Update display for current frame."""
        if state.episode_data is None:
            return (None, None, None, None, "No data")

        frame_idx = int(frame_idx)
        state.current_frame = frame_idx

        # Get frames as numpy arrays
        cam1_frame, cam2_frame, cam3_frame = get_frames_for_position(frame_idx)

        # Generate combined plot
        combined_plot = generate_combined_plot(state.episode_data, current_frame=frame_idx)

        frame_info = f"Frame {frame_idx}/{state.episode_data.length-1}"

        return (cam1_frame, cam2_frame, cam3_frame, combined_plot, frame_info)

    def step_frame(current: int, delta: int) -> int:
        """Step frame by delta."""
        if state.episode_data is None:
            return 0
        new_frame = max(0, min(int(current) + delta, state.episode_data.length - 1))
        return new_frame

    def toggle_play(is_playing: bool):
        """Toggle play/pause state and control timer."""
        new_state = not is_playing
        btn_text = "⏸ Pause" if new_state else "▶ Play"
        logger.info(f"toggle_play: {is_playing} -> {new_state}")
        return new_state, gr.update(value=btn_text), gr.Timer(active=new_state)

    def get_frames_for_position(frame_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera frames for a given position as numpy arrays."""
        cam1_frame = None
        cam2_frame = None
        cam3_frame = None
        video_keys = list(state.video_frames.keys())

        logger.debug(f"get_frames_for_position: frame_idx={frame_idx}, video_keys={video_keys}")

        if len(video_keys) >= 1:
            frames = state.video_frames[video_keys[0]]
            if frame_idx < len(frames):
                cam1_frame = resize_frame(frames[frame_idx], max_width=400)
                logger.debug(f"  cam1_frame: shape={cam1_frame.shape if cam1_frame is not None else None}, dtype={cam1_frame.dtype if cam1_frame is not None else None}")

        if len(video_keys) >= 2:
            frames = state.video_frames[video_keys[1]]
            if frame_idx < len(frames):
                cam2_frame = resize_frame(frames[frame_idx], max_width=400)
                logger.debug(f"  cam2_frame: shape={cam2_frame.shape if cam2_frame is not None else None}, dtype={cam2_frame.dtype if cam2_frame is not None else None}")

        if len(video_keys) >= 3:
            frames = state.video_frames[video_keys[2]]
            if frame_idx < len(frames):
                cam3_frame = resize_frame(frames[frame_idx], max_width=400)
                logger.debug(f"  cam3_frame: shape={cam3_frame.shape if cam3_frame is not None else None}, dtype={cam3_frame.dtype if cam3_frame is not None else None}")

        return cam1_frame, cam2_frame, cam3_frame

    def advance_frame(current_frame, speed) -> Tuple:
        """Advance frame during playback - only called when timer is active."""
        logger.debug(f"advance_frame CALLED: current_frame={current_frame}, speed={speed}")

        # Handle None inputs
        if current_frame is None:
            logger.warning("advance_frame: current_frame is None!")
            current_frame = state.current_frame if state.current_frame else 0
        if speed is None:
            logger.warning("advance_frame: speed is None!")
            speed = 1

        current_frame = int(current_frame)
        speed = int(speed)

        if state.episode_data is None or state.max_frame == 0:
            logger.warning("advance_frame: No episode data, stopping timer")
            return (gr.update(), gr.update(), gr.update(), gr.update(), 0, gr.Timer(active=False))

        max_frame = state.max_frame
        current_frame = min(max(0, current_frame), max_frame)
        new_frame = current_frame + speed

        # End of episode - stop timer
        if new_frame >= max_frame:
            new_frame = max_frame
            state.current_frame = new_frame
            cam1_frame, cam2_frame, cam3_frame = get_frames_for_position(new_frame)
            logger.info(f"Playback ended at frame {new_frame}")
            return (
                cam1_frame,
                cam2_frame,
                cam3_frame,
                f"Frame {new_frame}/{max_frame}",
                int(new_frame),
                gr.Timer(active=False),  # Stop timer
            )

        # Playing - update frames
        state.current_frame = new_frame
        cam1_frame, cam2_frame, cam3_frame = get_frames_for_position(new_frame)

        return (
            cam1_frame,
            cam2_frame,
            cam3_frame,
            f"Frame {new_frame}/{max_frame}",
            int(new_frame),
            gr.update(),  # Keep timer running
        )

    def on_browse_select(selected_path: str) -> str:
        """Handle browse selection."""
        if selected_path:
            return selected_path
        return gr.update()

    # -------------------------------------------------------------------------
    # Build UI
    # -------------------------------------------------------------------------

    with gr.Blocks(
        title="Robot Dataset Visualizer",
        fill_height=True,
    ) as app:

        gr.Markdown("## Robot Dataset Visualizer")

        with gr.Row():
            # -----------------------------------------------------------------
            # Left Sidebar: Dataset Browser
            # -----------------------------------------------------------------
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Dataset")

                dataset_path = gr.Textbox(
                    label="Path",
                    value=default_dataset,
                    placeholder="/path/to/dataset",
                )

                # Browse button with file explorer
                with gr.Row():
                    browse_btn = gr.Button("Browse", scale=1)

                format_dropdown = gr.Dropdown(
                    label="Format",
                    choices=[("LeRobot v2", "v2"), ("LeRobot v3", "v3"), ("Auto-detect", "auto")],
                    value=default_format,
                )

                load_btn = gr.Button("Load Dataset", variant="primary")

                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                )

                info_text = gr.Textbox(
                    label="Info",
                    interactive=False,
                    lines=1,
                )

                gr.Markdown("### Episodes")

                episode_dropdown = gr.Dropdown(
                    label="Select Episode",
                    choices=[],
                    interactive=True,
                )

                task_display = gr.Textbox(
                    label="Task",
                    interactive=False,
                    lines=2,
                )

            # -----------------------------------------------------------------
            # Right Main Display: Cameras + Charts (stacked vertically)
            # -----------------------------------------------------------------
            with gr.Column(scale=3):

                # Camera views side by side - 3 cameras
                # Labels are updated dynamically based on actual video keys when episode loads
                with gr.Row():
                    cam1_img = gr.Image(
                        label="Camera 1",
                        show_label=True,
                        height=240,
                        width=320,
                    )
                    cam2_img = gr.Image(
                        label="Camera 2",
                        show_label=True,
                        height=240,
                        width=320,
                    )
                    cam3_img = gr.Image(
                        label="Camera 3",
                        show_label=True,
                        height=240,
                        width=320,
                    )

                # Combined arm joints + gripper plot (stacked with shared x-axis)
                combined_plot = gr.Plot(
                    label="Joint States (Arm + Gripper)",
                )

                # Timeline slider - set high default max to avoid bounds errors during async updates
                timeline_slider = gr.Slider(
                    minimum=0,
                    maximum=100000,
                    step=1,
                    value=0,
                    label="Timeline",
                    interactive=True,
                )

                # Playback controls at the bottom
                with gr.Row():
                    prev_10_btn = gr.Button("⏪ -10", scale=1)
                    prev_btn = gr.Button("◀ -1", scale=1)
                    play_btn = gr.Button("▶ Play", variant="primary", scale=2)
                    next_btn = gr.Button("+1 ▶", scale=1)
                    next_10_btn = gr.Button("+10 ⏩", scale=1)
                    speed_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=3,
                        label="Speed",
                        scale=1,
                    )
                    frame_info = gr.Textbox(
                        value="Frame 0/0",
                        interactive=False,
                        scale=2,
                        show_label=False,
                    )

        # Hidden state for playback
        is_playing_state = gr.State(False)

        # ---------------------------------------------------------------------
        # File Browser Modal
        # ---------------------------------------------------------------------
        with gr.Row(visible=False) as browse_modal:
            file_browser = gr.FileExplorer(
                label="Select Dataset Directory",
                file_count="single",
                root_dir=os.path.expanduser("~"),
            )

        browse_visible = gr.State(False)

        def toggle_browse(visible):
            return not visible, gr.Row(visible=not visible)

        browse_btn.click(
            fn=toggle_browse,
            inputs=[browse_visible],
            outputs=[browse_visible, browse_modal],
        )

        def select_from_browser(selected, visible):
            if selected:
                # Get directory path
                path = str(selected)
                if os.path.isfile(path):
                    path = os.path.dirname(path)
                return path, False, gr.Row(visible=False)
            return gr.update(), visible, gr.update()

        file_browser.change(
            fn=select_from_browser,
            inputs=[file_browser, browse_visible],
            outputs=[dataset_path, browse_visible, browse_modal],
        )

        # ---------------------------------------------------------------------
        # Event Bindings
        # ---------------------------------------------------------------------

        load_btn.click(
            fn=load_dataset,
            inputs=[dataset_path, format_dropdown],
            outputs=[episode_dropdown, status_text, info_text],
        )

        episode_dropdown.change(
            fn=load_episode,
            inputs=[episode_dropdown],
            outputs=[
                cam1_img, cam2_img, cam3_img,
                combined_plot,
                timeline_slider, timeline_slider,
                frame_info, task_display,
                cam1_img, cam2_img, cam3_img,  # For label updates
            ],
        )

        timeline_slider.change(
            fn=update_frame,
            inputs=[timeline_slider],
            outputs=[cam1_img, cam2_img, cam3_img, combined_plot, frame_info],
            show_progress="hidden",
        )

        # Navigation buttons
        prev_10_btn.click(
            fn=lambda x: step_frame(x, -10),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        prev_btn.click(
            fn=lambda x: step_frame(x, -1),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        next_btn.click(
            fn=lambda x: step_frame(x, 1),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )
        next_10_btn.click(
            fn=lambda x: step_frame(x, 10),
            inputs=[timeline_slider],
            outputs=[timeline_slider],
        )

        # Timer for playback - inactive by default, activated by play button
        timer = gr.Timer(value=0.1, active=False)

        # Play/Pause button - controls timer
        play_btn.click(
            fn=toggle_play,
            inputs=[is_playing_state],
            outputs=[is_playing_state, play_btn, timer],
        )

        # Timer tick - only runs when playing
        timer.tick(
            fn=advance_frame,
            inputs=[timeline_slider, speed_slider],
            outputs=[
                cam1_img, cam2_img, cam3_img,
                frame_info, timeline_slider,
                timer,
            ],
            show_progress="hidden",
        )

        # Update button text and plot when play state changes (e.g., when reaching end)
        def on_play_state_change(is_playing, current_frame):
            btn_text = "⏸ Pause" if is_playing else "▶ Play"
            logger.info(f"on_play_state_change: is_playing={is_playing}")

            # When stopping, update plot to show current position
            if not is_playing and state.episode_data is not None:
                current_frame = int(current_frame) if current_frame is not None else 0
                plot = generate_combined_plot(state.episode_data, current_frame=current_frame)
                return gr.update(value=btn_text), plot, gr.Timer(active=False)

            return gr.update(value=btn_text), gr.update(), gr.Timer(active=is_playing)

        is_playing_state.change(
            fn=on_play_state_change,
            inputs=[is_playing_state, timeline_slider],
            outputs=[play_btn, combined_plot, timer],
        )

    return app


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Robot Dataset Visualizer")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=".",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=7860,
        help="Server port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["v2", "v3", "auto"],
        default="v2",
        help="Dataset format version (default: v2)",
    )
    args = parser.parse_args()

    app = create_app(default_dataset=args.dataset, default_format=args.format)
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
