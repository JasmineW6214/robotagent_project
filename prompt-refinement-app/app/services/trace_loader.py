"""
Trace Loader Service.

Parses VLM trace directories from logs/images/ to extract:
- Images (before/after for monitoring, camera images for planning)
- System and user prompts
- VLM outputs
- Context information
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class VLMTrace:
    """Base class for VLM traces."""
    trace_id: str
    call_type: str  # "planning" or "monitoring"
    call_number: int
    timestamp: str
    system_prompt: str
    user_prompt: str
    response: str  # Raw VLM response
    parsed_output: dict

    # Source info
    source_dir: Path = field(default=None)
    detail_file: Path = field(default=None)


@dataclass
class PlanningTrace(VLMTrace):
    """A planning VLM call trace."""
    user_command: str = ""
    context: str = ""
    task_selected: Optional[str] = None
    confidence: float = 0.0
    images: dict[str, Path] = field(default_factory=dict)  # camera_name -> path


@dataclass
class MonitoringTrace(VLMTrace):
    """A monitoring VLM call trace."""
    task_name: str = ""
    success: bool = False
    control_action: str = ""
    failure_reason: Optional[str] = None
    before_image: Optional[Path] = None
    after_image: Optional[Path] = None


class TraceLoader:
    """Load VLM traces from logs/images directories."""

    def __init__(self, logs_base: Path = None):
        """Initialize trace loader.

        Args:
            logs_base: Base path to logs directory. Defaults to project logs/images.
        """
        if logs_base is None:
            # Default to project logs
            project_root = Path(__file__).parent.parent.parent.parent
            logs_base = project_root / "logs" / "images"

        self.logs_base = Path(logs_base)

    def list_trace_dirs(self) -> list[dict]:
        """List available trace directories.

        Returns:
            List of dicts with trace directory info:
            {
                "id": "20260111_212709",
                "path": Path(...),
                "timestamp": datetime,
                "num_planning": 1,
                "num_monitoring": 8
            }
        """
        dirs = []

        if not self.logs_base.exists():
            logger.warning(f"Logs base directory not found: {self.logs_base}")
            return dirs

        for trace_dir in sorted(self.logs_base.iterdir(), reverse=True):
            if not trace_dir.is_dir():
                continue

            # Parse timestamp from directory name
            # Supports two formats:
            # - Old: YYYYMMDD_HHMMSS (e.g., 20260111_212709)
            # - New: YYYYMMDD_HHMMSS_<command> (e.g., 20260113_155547_put_food_on_plate)
            dir_name = trace_dir.name
            try:
                # Try old format first
                ts = datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
            except ValueError:
                # Try new format: extract first 15 chars (YYYYMMDD_HHMMSS)
                if len(dir_name) >= 15 and dir_name[8] == "_":
                    try:
                        ts = datetime.strptime(dir_name[:15], "%Y%m%d_%H%M%S")
                    except ValueError:
                        continue
                else:
                    continue

            # Count planning and monitoring files
            planning_files = list(trace_dir.glob("plan_*_detail.json"))
            monitoring_files = list(trace_dir.glob("monitor_*_detail.json"))

            if not planning_files and not monitoring_files:
                continue  # Skip empty directories

            dirs.append({
                "id": trace_dir.name,
                "path": trace_dir,
                "timestamp": ts,
                "num_planning": len(planning_files),
                "num_monitoring": len(monitoring_files),
            })

        return dirs

    def load_trace_dir(self, trace_id: str) -> dict:
        """Load all traces from a directory.

        Args:
            trace_id: Directory name (e.g., "20260111_212709")

        Returns:
            Dict with planning and monitoring traces
        """
        trace_dir = self.logs_base / trace_id

        if not trace_dir.exists():
            raise ValueError(f"Trace directory not found: {trace_dir}")

        planning_traces = []
        monitoring_traces = []

        # Load planning traces
        for detail_file in sorted(trace_dir.glob("plan_*_detail.json")):
            try:
                trace = self._load_planning_trace(detail_file)
                if trace:
                    planning_traces.append(trace)
            except Exception as e:
                logger.error(f"Error loading {detail_file}: {e}")

        # Load monitoring traces
        for detail_file in sorted(trace_dir.glob("monitor_*_detail.json")):
            try:
                trace = self._load_monitoring_trace(detail_file)
                if trace:
                    monitoring_traces.append(trace)
            except Exception as e:
                logger.error(f"Error loading {detail_file}: {e}")

        return {
            "id": trace_id,
            "path": str(trace_dir),
            "planning": planning_traces,
            "monitoring": monitoring_traces,
        }

    def load_single_trace(self, trace_id: str, call_type: str, call_number: int) -> VLMTrace:
        """Load a single trace by ID and call info.

        Args:
            trace_id: Directory name
            call_type: "planning" or "monitoring"
            call_number: Call number (1-indexed)

        Returns:
            VLMTrace (PlanningTrace or MonitoringTrace)
        """
        trace_dir = self.logs_base / trace_id

        if call_type == "planning":
            pattern = f"plan_{call_number:04d}_detail.json"
            files = list(trace_dir.glob(pattern))
            if files:
                return self._load_planning_trace(files[0])
        else:
            pattern = f"monitor_{call_number:04d}_detail.json"
            files = list(trace_dir.glob(pattern))
            if files:
                return self._load_monitoring_trace(files[0])

        raise ValueError(f"Trace not found: {trace_id}/{call_type}/{call_number}")

    def _load_planning_trace(self, detail_file: Path) -> PlanningTrace:
        """Load a planning trace from detail JSON."""
        with open(detail_file) as f:
            data = json.load(f)

        trace_dir = detail_file.parent

        # Find associated images
        images = {}
        call_num = data.get("call_number", 1)
        prefix = f"plan_{call_num:04d}"

        for img_file in trace_dir.glob(f"{prefix}_*.jpg"):
            # Extract camera name from filename
            # e.g., plan_0001_central_0001.jpg -> central
            parts = img_file.stem.split("_")
            if len(parts) >= 3:
                camera_name = parts[2]
                images[camera_name] = img_file

        # Parse response JSON
        response = data.get("response", "{}")
        try:
            parsed = json.loads(response) if isinstance(response, str) else response
        except json.JSONDecodeError:
            parsed = {"raw": response}

        return PlanningTrace(
            trace_id=trace_dir.name,
            call_type="planning",
            call_number=data.get("call_number", 1),
            timestamp=data.get("timestamp", ""),
            system_prompt=data.get("system_prompt", ""),
            user_prompt=data.get("user_prompt", ""),
            response=response,
            parsed_output=parsed,
            source_dir=trace_dir,
            detail_file=detail_file,
            user_command=data.get("user_command", ""),
            context=data.get("context", ""),
            task_selected=data.get("task_selected"),
            confidence=data.get("confidence", 0.0),
            images=images,
        )

    def _load_monitoring_trace(self, detail_file: Path) -> MonitoringTrace:
        """Load a monitoring trace from detail JSON."""
        with open(detail_file) as f:
            data = json.load(f)

        trace_dir = detail_file.parent
        call_num = data.get("call_number", 1)

        # Find before/after images
        before_paths = data.get("before_image_paths", {})
        after_paths = data.get("after_image_paths", {})

        before_image = None
        after_image = None

        # Get first before image
        for key, path in before_paths.items():
            p = Path(path) if Path(path).is_absolute() else trace_dir / Path(path).name
            if p.exists():
                before_image = p
                break

        # Get first after image
        for key, path in after_paths.items():
            p = Path(path) if Path(path).is_absolute() else trace_dir / Path(path).name
            if p.exists():
                after_image = p
                break

        # Parse response JSON
        response = data.get("response", "{}")
        try:
            parsed = json.loads(response) if isinstance(response, str) else response
        except json.JSONDecodeError:
            parsed = {"raw": response}

        return MonitoringTrace(
            trace_id=trace_dir.name,
            call_type="monitoring",
            call_number=call_num,
            timestamp=data.get("timestamp", ""),
            system_prompt=data.get("system_prompt", ""),
            user_prompt=data.get("user_prompt", ""),
            response=response,
            parsed_output=parsed,
            source_dir=trace_dir,
            detail_file=detail_file,
            task_name=data.get("task_name", ""),
            success=data.get("success", False),
            control_action=data.get("control_action", ""),
            failure_reason=data.get("failure_reason"),
            before_image=before_image,
            after_image=after_image,
        )

    def trace_to_dict(self, trace: VLMTrace) -> dict:
        """Convert a trace to a JSON-serializable dict."""
        base = {
            "trace_id": trace.trace_id,
            "call_type": trace.call_type,
            "call_number": trace.call_number,
            "timestamp": trace.timestamp,
            "system_prompt": trace.system_prompt,
            "user_prompt": trace.user_prompt,
            "response": trace.response,
            "parsed_output": trace.parsed_output,
        }

        if isinstance(trace, PlanningTrace):
            base.update({
                "user_command": trace.user_command,
                "context": trace.context,
                "task_selected": trace.task_selected,
                "confidence": trace.confidence,
                "images": {k: str(v) for k, v in trace.images.items()},
            })
        elif isinstance(trace, MonitoringTrace):
            base.update({
                "task_name": trace.task_name,
                "success": trace.success,
                "control_action": trace.control_action,
                "failure_reason": trace.failure_reason,
                "before_image": str(trace.before_image) if trace.before_image else None,
                "after_image": str(trace.after_image) if trace.after_image else None,
            })

        return base
