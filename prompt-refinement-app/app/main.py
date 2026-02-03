"""
Prompt Refinement Web App - FastAPI Backend.

A web-based tool for experimenting with VLM prompts.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .services.trace_loader import TraceLoader
from .services.vlm_runner import VLMRunner, MockVLMRunner
from .services.vllm_runner import VLLMRunner
from .services.tensorrt_runner import TensorRTRunner
from .services.claude_runner import ClaudeRunner, MockClaudeRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
trace_loader: TraceLoader = None
vlm_runner = None  # Transformers-based runner
vllm_runner = None  # vLLM server-based runner
tensorrt_runner = None  # TensorRT-LLM server-based runner
claude_runner = None
USE_MOCK = False  # Set to True to use mock models (no GPU required)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global trace_loader, vlm_runner, vllm_runner, tensorrt_runner, claude_runner, USE_MOCK

    # Initialize trace loader
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / "logs" / "images"
    trace_loader = TraceLoader(logs_dir)

    # Initialize runners (mock by default for safety)
    if USE_MOCK:
        logger.info("Using MOCK runners (no GPU required)")
        vlm_runner = MockVLMRunner()
        claude_runner = MockClaudeRunner()
    else:
        logger.info("Using REAL runners (GPU and API key required)")
        vlm_runner = VLMRunner()
        claude_runner = ClaudeRunner()

        # Initialize vLLM runner (connects to external server)
        vllm_runner = VLLMRunner(base_url="http://localhost:8000")
        if vllm_runner.is_available():
            logger.info("vLLM server detected at localhost:8000")

            # Warm up KV cache for planning and monitoring prompts
            # This makes the first real request ~2x faster
            prompts_dir = project_root / "src" / "robotagent" / "vlm" / "prompts"
            system_prompts = []
            for prompt_file in ["planning_system_v2.md", "monitoring_system.md"]:
                prompt_path = prompts_dir / prompt_file
                if prompt_path.exists():
                    system_prompts.append(prompt_path.read_text())

            if system_prompts:
                vllm_runner.warm_cache(system_prompts)
        else:
            logger.info("vLLM server not running (use scripts/start_vllm_server.sh)")

        # Initialize TensorRT-LLM runner (connects to Docker server)
        tensorrt_runner = TensorRTRunner(base_url="http://localhost:8001")
        if tensorrt_runner.is_available():
            logger.info("TensorRT-LLM server detected at localhost:8001 (~56 tok/s)")
        else:
            logger.info("TensorRT-LLM server not running (use scripts/start_tensorrt_server.sh)")

    logger.info("Prompt Refinement App initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Prompt Refinement Factory",
    description="Web tool for experimenting with VLM prompts",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ============= API Models =============

class RunRequest(BaseModel):
    """Request to run VLM inference."""
    model: str  # "4B-Instruct", "4B-Thinking", etc.
    mode: str  # "planning" or "monitoring"
    system_prompt: str
    user_prompt: str
    images: dict[str, str]  # {name: path_or_base64}
    context: Optional[dict] = None
    max_tokens: int = 256  # Max output tokens (256 for planning, can reduce for monitoring)
    disable_thinking: bool = False  # If True, disable thinking mode via /no_think
    use_sampling: bool = False  # If True, use sampling; default False (greedy, faster)
    image_resolution: str = "small"  # "micro", "tiny", "small" (recommended), "default"
    backend: str = "transformers"  # "transformers", "vllm", or "tensorrt"


class CompareRequest(BaseModel):
    """Request to compare multiple models."""
    models: list[str]  # ["4B-Instruct", "4B-Thinking"]
    mode: str
    system_prompt: str
    user_prompt: str
    images: dict[str, str]
    context: Optional[dict] = None
    max_tokens: int = 256  # Max output tokens (256 for planning, can reduce for monitoring)
    disable_thinking: bool = False  # If True, disable thinking mode via /no_think
    use_sampling: bool = False  # If True, use sampling; default False (greedy, faster)
    image_resolution: str = "small"  # "micro", "tiny", "small" (recommended), "default"
    backend: str = "transformers"  # "transformers", "vllm", or "tensorrt"


class PromptVersion(BaseModel):
    """A saved prompt version."""
    version_id: str
    name: str
    mode: str  # "planning" or "monitoring"
    system_prompt: str
    user_prompt: str
    notes: Optional[str] = None


# ============= Routes =============

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Prompt Refinement Factory API", "docs": "/docs"}


@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "mock_mode": USE_MOCK,
        "claude_available": claude_runner.is_available() if claude_runner else False,
        "vllm_available": vllm_runner.is_available() if vllm_runner else False,
        "tensorrt_available": tensorrt_runner.is_available() if tensorrt_runner else False,
    }


# ============= Trace Routes =============

@app.get("/api/traces")
async def list_traces():
    """List available trace directories."""
    dirs = trace_loader.list_trace_dirs()
    return [{
        "id": d["id"],
        "timestamp": d["timestamp"].isoformat(),
        "num_planning": d["num_planning"],
        "num_monitoring": d["num_monitoring"],
    } for d in dirs]


@app.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    """Load all traces from a directory."""
    try:
        data = trace_loader.load_trace_dir(trace_id)
        return {
            "id": data["id"],
            "path": data["path"],
            "planning": [trace_loader.trace_to_dict(t) for t in data["planning"]],
            "monitoring": [trace_loader.trace_to_dict(t) for t in data["monitoring"]],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/traces/{trace_id}/{call_type}/{call_number}")
async def get_single_trace(trace_id: str, call_type: str, call_number: int):
    """Load a single trace."""
    try:
        trace = trace_loader.load_single_trace(trace_id, call_type, call_number)
        return trace_loader.trace_to_dict(trace)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/images/{trace_id}/{filename}")
async def get_image(trace_id: str, filename: str):
    """Serve an image from a trace directory."""
    project_root = Path(__file__).parent.parent.parent
    image_path = project_root / "logs" / "images" / trace_id / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpeg")


# ============= Model Routes =============

@app.get("/api/models")
async def list_models():
    """List available models."""
    qwen_models = vlm_runner.list_available_models() if vlm_runner else []
    vllm_models = vllm_runner.list_available_models() if vllm_runner else []
    tensorrt_models = tensorrt_runner.list_available_models() if tensorrt_runner else []

    return {
        "qwen": qwen_models,
        "vllm": vllm_models,
        "tensorrt": tensorrt_models,
        "vllm_available": vllm_runner.is_available() if vllm_runner else False,
        "tensorrt_available": tensorrt_runner.is_available() if tensorrt_runner else False,
        "current_loaded": vlm_runner._current_model if vlm_runner else None,
    }


@app.get("/api/vllm/status")
async def vllm_status():
    """Check vLLM server status and available models."""
    if not vllm_runner:
        return {"available": False, "models": [], "error": "vLLM runner not initialized"}

    available = vllm_runner.is_available()
    models = vllm_runner.get_models() if available else []

    return {
        "available": available,
        "models": models,
        "base_url": vllm_runner._base_url if vllm_runner else None,
    }


@app.get("/api/tensorrt/status")
async def tensorrt_status():
    """Check TensorRT-LLM server status and available models."""
    if not tensorrt_runner:
        return {"available": False, "models": [], "error": "TensorRT-LLM runner not initialized"}

    available = tensorrt_runner.is_available()
    models = tensorrt_runner.get_models() if available else []

    return {
        "available": available,
        "models": models,
        "base_url": tensorrt_runner._base_url if tensorrt_runner else None,
        "performance": "~56 tok/s (2x faster than vLLM)",
    }


@app.post("/api/unload")
async def unload_model():
    """Unload current model and free GPU memory."""
    if not vlm_runner:
        raise HTTPException(status_code=500, detail="VLM runner not initialized")

    result = vlm_runner.unload()
    return result


@app.post("/api/run")
async def run_model(request: RunRequest):
    """Run VLM inference."""
    if request.model == "claude":
        if not claude_runner:
            raise HTTPException(status_code=500, detail="Claude runner not initialized")
        result = claude_runner.run(
            mode=request.mode,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            images=request.images,
            context=request.context,
        )
    elif request.backend == "vllm":
        # Use vLLM server backend
        if not vllm_runner:
            raise HTTPException(status_code=500, detail="vLLM runner not initialized")
        if not vllm_runner.is_available():
            raise HTTPException(
                status_code=503,
                detail="vLLM server not running. Start with: ./scripts/start_vllm_server.sh"
            )
        result = vllm_runner.run(
            model=request.model,
            mode=request.mode,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            images=request.images,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=0.7 if request.use_sampling else 0.0,
            image_resolution=request.image_resolution,
        )
    elif request.backend == "tensorrt":
        # Use TensorRT-LLM server backend (~56 tok/s, 2x faster than vLLM)
        if not tensorrt_runner:
            raise HTTPException(status_code=500, detail="TensorRT-LLM runner not initialized")
        if not tensorrt_runner.is_available():
            raise HTTPException(
                status_code=503,
                detail="TensorRT-LLM server not running. Start with: ./scripts/start_tensorrt_server.sh"
            )
        result = tensorrt_runner.run(
            model=request.model,
            mode=request.mode,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            images=request.images,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=0.7 if request.use_sampling else 0.0,
            image_resolution=request.image_resolution,
        )
    else:
        # Use Transformers backend (default)
        if not vlm_runner:
            raise HTTPException(status_code=500, detail="VLM runner not initialized")
        result = vlm_runner.run(
            model=request.model,
            mode=request.mode,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            images=request.images,
            context=request.context,
            max_tokens=request.max_tokens,
            disable_thinking=request.disable_thinking,
            use_sampling=request.use_sampling,
            image_resolution=request.image_resolution,
        )

    response = {
        "model": result.model,
        "backend": request.backend,
        "success": result.success,
        "raw_response": result.raw_response,
        "parsed_output": result.parsed_output,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }

    # Include detailed metrics if available (Transformers backend)
    if hasattr(result, 'metrics') and result.metrics:
        response["metrics"] = {
            "total_ms": result.metrics.total_ms,
            "preprocessing_ms": result.metrics.preprocessing_ms,
            "generation_ms": result.metrics.generation_ms,
            "postprocessing_ms": result.metrics.postprocessing_ms,
            "input_tokens": result.metrics.input_tokens,
            "output_tokens": result.metrics.output_tokens,
            "tokens_per_second": result.metrics.tokens_per_second,
        }
    # vLLM metrics
    elif hasattr(result, 'tokens_per_second'):
        response["metrics"] = {
            "total_ms": result.latency_ms,
            "generation_ms": result.latency_ms,  # For vLLM, total latency is generation time
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.completion_tokens,
            "tokens_per_second": result.tokens_per_second,
        }

    return response


@app.post("/api/compare")
async def compare_models(request: CompareRequest):
    """Run multiple models and compare outputs."""
    results = {}

    for model in request.models:
        if model == "claude":
            if claude_runner:
                result = claude_runner.run(
                    mode=request.mode,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    images=request.images,
                    context=request.context,
                )
                results[model] = {
                    "success": result.success,
                    "raw_response": result.raw_response,
                    "parsed_output": result.parsed_output,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
        elif request.backend == "vllm":
            # Use vLLM server backend
            if vllm_runner and vllm_runner.is_available():
                result = vllm_runner.run(
                    model=model,
                    mode=request.mode,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    images=request.images,
                    context=request.context,
                    max_tokens=request.max_tokens,
                    temperature=0.7 if request.use_sampling else 0.0,
                    image_resolution=request.image_resolution,
                )
                result_data = {
                    "success": result.success,
                    "raw_response": result.raw_response,
                    "parsed_output": result.parsed_output,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
                if hasattr(result, 'tokens_per_second'):
                    result_data["metrics"] = {
                        "total_ms": result.latency_ms,
                        "generation_ms": result.latency_ms,
                        "prompt_tokens": result.prompt_tokens,
                        "output_tokens": result.completion_tokens,
                        "tokens_per_second": result.tokens_per_second,
                    }
                results[model] = result_data
        elif request.backend == "tensorrt":
            # Use TensorRT-LLM server backend
            if tensorrt_runner and tensorrt_runner.is_available():
                result = tensorrt_runner.run(
                    model=model,
                    mode=request.mode,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    images=request.images,
                    context=request.context,
                    max_tokens=request.max_tokens,
                    temperature=0.7 if request.use_sampling else 0.0,
                    image_resolution=request.image_resolution,
                )
                result_data = {
                    "success": result.success,
                    "raw_response": result.raw_response,
                    "parsed_output": result.parsed_output,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
                if hasattr(result, 'tokens_per_second'):
                    result_data["metrics"] = {
                        "total_ms": result.latency_ms,
                        "generation_ms": result.latency_ms,
                        "prompt_tokens": result.prompt_tokens,
                        "output_tokens": result.completion_tokens,
                        "tokens_per_second": result.tokens_per_second,
                    }
                results[model] = result_data
        else:
            # Use Transformers backend (default)
            if vlm_runner:
                result = vlm_runner.run(
                    model=model,
                    mode=request.mode,
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    images=request.images,
                    context=request.context,
                    max_tokens=request.max_tokens,
                    disable_thinking=request.disable_thinking,
                    use_sampling=request.use_sampling,
                    image_resolution=request.image_resolution,
                )
                result_data = {
                    "success": result.success,
                    "raw_response": result.raw_response,
                    "parsed_output": result.parsed_output,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
                if result.metrics:
                    result_data["metrics"] = {
                        "total_ms": result.metrics.total_ms,
                        "generation_ms": result.metrics.generation_ms,
                        "output_tokens": result.metrics.output_tokens,
                        "tokens_per_second": result.metrics.tokens_per_second,
                    }
                results[model] = result_data

    # Compare outputs
    comparison = _compare_outputs(results)

    return {
        "results": results,
        "comparison": comparison,
    }


def _compare_outputs(results: dict) -> dict:
    """Compare outputs from multiple models."""
    if len(results) < 2:
        return {"discrepancies": []}

    discrepancies = []
    models = list(results.keys())

    # Get reference (prefer Claude)
    ref_model = "claude" if "claude" in models else models[0]
    ref_output = results[ref_model].get("parsed_output", {})

    for model in models:
        if model == ref_model:
            continue

        output = results[model].get("parsed_output", {})

        # Compare key fields
        for field in ["success", "current_task", "next_task", "control_action"]:
            if field in ref_output or field in output:
                ref_val = ref_output.get(field)
                other_val = output.get(field)
                if ref_val != other_val:
                    discrepancies.append({
                        "field": field,
                        "reference": {"model": ref_model, "value": ref_val},
                        "compared": {"model": model, "value": other_val},
                    })

    return {
        "reference_model": ref_model,
        "discrepancies": discrepancies,
        "match": len(discrepancies) == 0,
    }


# ============= Prompt Version Routes =============

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "versions"


@app.get("/api/prompts")
async def list_prompt_versions():
    """List saved prompt versions."""
    versions = []

    if PROMPTS_DIR.exists():
        for version_dir in sorted(PROMPTS_DIR.iterdir()):
            if version_dir.is_dir():
                meta_file = version_dir / "meta.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    versions.append(meta)

    return versions


@app.get("/api/prompts/{version_id}")
async def get_prompt_version(version_id: str):
    """Get a specific prompt version."""
    version_dir = PROMPTS_DIR / version_id

    if not version_dir.exists():
        raise HTTPException(status_code=404, detail="Version not found")

    meta_file = version_dir / "meta.json"
    system_file = version_dir / "system.md"
    user_file = version_dir / "user.md"

    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    system_prompt = ""
    if system_file.exists():
        with open(system_file) as f:
            system_prompt = f.read()

    user_prompt = ""
    if user_file.exists():
        with open(user_file) as f:
            user_prompt = f.read()

    return {
        **meta,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


@app.post("/api/prompts")
async def save_prompt_version(version: PromptVersion):
    """Save a new prompt version."""
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    version_dir = PROMPTS_DIR / version.version_id
    version_dir.mkdir(exist_ok=True)

    # Save metadata
    meta = {
        "version_id": version.version_id,
        "name": version.name,
        "mode": version.mode,
        "notes": version.notes,
    }
    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save prompts
    with open(version_dir / "system.md", "w") as f:
        f.write(version.system_prompt)

    with open(version_dir / "user.md", "w") as f:
        f.write(version.user_prompt)

    return {"status": "saved", "version_id": version.version_id}


# ============= Export for Claude Code =============

EXPORTS_DIR = Path(__file__).parent.parent / "exports"


@app.post("/api/export-for-claude")
async def export_for_claude(request: RunRequest):
    """Export inputs in a format for Claude Code comparison.

    Creates a directory with:
    - images (copied)
    - prompt.md (combined prompt ready for Claude Code)
    """
    import shutil
    from datetime import datetime

    # Create export directory
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = EXPORTS_DIR / f"export_{timestamp}"
    export_dir.mkdir()

    # Copy images
    image_refs = []
    for key, value in request.images.items():
        if isinstance(value, str):
            # Handle file path or URL
            if value.startswith("/api/images/"):
                # It's an API URL, extract the actual path
                parts = value.replace("/api/images/", "").split("/")
                if len(parts) == 2:
                    trace_id, filename = parts
                    project_root = Path(__file__).parent.parent.parent
                    src_path = project_root / "logs" / "images" / trace_id / filename
                    if src_path.exists():
                        dst_path = export_dir / f"{key}.jpg"
                        shutil.copy(src_path, dst_path)
                        image_refs.append((key, dst_path.name))
            elif Path(value).exists():
                # Direct file path
                dst_path = export_dir / f"{key}.jpg"
                shutil.copy(value, dst_path)
                image_refs.append((key, dst_path.name))

    # Create the prompt file for Claude Code
    prompt_content = f"""# VLM Comparison Request

## Mode: {request.mode}

## Images

"""
    for key, filename in image_refs:
        prompt_content += f"### {key.upper()} Image\n"
        prompt_content += f"![{key}]({filename})\n\n"

    prompt_content += f"""## System Prompt

{request.system_prompt}

## User Prompt

{request.user_prompt}

## Instructions for Claude Code

Please analyze the images above and respond according to the system and user prompts.
Output your response as JSON matching this format:

"""

    if request.mode == "monitoring":
        prompt_content += """```json
{
  "success": true/false,
  "failure_type": "string if failed",
  "failure_reason": "string if failed",
  "reasoning": "your detailed analysis",
  "control_action": "continue/retry/replan/skip/pause",
  "confidence": 0.0-1.0
}
```
"""
    else:
        prompt_content += """```json
{
  "thought": "your reasoning",
  "next_task": "task_name or null",
  "confidence": 0.0-1.0,
  "needs_clarification": true/false
}
```
"""

    # Write prompt file
    prompt_file = export_dir / "prompt.md"
    with open(prompt_file, "w") as f:
        f.write(prompt_content)

    # Also write raw prompts separately for reference
    with open(export_dir / "system_prompt.txt", "w") as f:
        f.write(request.system_prompt)

    with open(export_dir / "user_prompt.txt", "w") as f:
        f.write(request.user_prompt)

    return {
        "export_dir": str(export_dir),
        "files": {
            "prompt": str(prompt_file),
            "images": [str(export_dir / img[1]) for img in image_refs],
        },
        "usage": f"Run: claude 'Read {prompt_file} and analyze the images in {export_dir}'",
    }


# ============= Current Prompts Routes =============

@app.get("/api/current-prompts/{mode}")
async def get_current_prompts(mode: str, version: str = "v1"):
    """Get current production prompts from robotagent.

    Args:
        mode: "planning" or "monitoring"
        version: "v1" (original coded task names) or "v2" (string-based task descriptions)
    """
    project_root = Path(__file__).parent.parent.parent
    prompts_dir = project_root / "src" / "robotagent" / "vlm" / "prompts"

    # Determine file suffix based on version
    suffix = "" if version == "v1" else "_v2"

    if mode == "planning":
        system_file = prompts_dir / f"planning_system{suffix}.md"
        user_file = prompts_dir / f"planning_user{suffix}.md"
    elif mode == "monitoring":
        system_file = prompts_dir / f"monitoring_system{suffix}.md"
        user_file = prompts_dir / f"monitoring_user{suffix}.md"
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

    system_prompt = ""
    user_prompt = ""

    if system_file.exists():
        with open(system_file) as f:
            system_prompt = f.read()
    else:
        # Fall back to v1 if v2 doesn't exist
        if version == "v2":
            return await get_current_prompts(mode, "v1")

    if user_file.exists():
        with open(user_file) as f:
            user_prompt = f.read()

    return {
        "mode": mode,
        "version": version,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


@app.get("/api/prompt-versions")
async def list_available_prompt_versions():
    """List available prompt versions (v1, v2, etc.)."""
    project_root = Path(__file__).parent.parent.parent
    prompts_dir = project_root / "src" / "robotagent" / "vlm" / "prompts"

    versions = {"planning": [], "monitoring": []}

    # Check for planning prompts
    if (prompts_dir / "planning_system.md").exists():
        versions["planning"].append({
            "id": "v1",
            "name": "V1 - Coded Task Names",
            "description": "Original format: 'left_orange_plate'"
        })
    if (prompts_dir / "planning_system_v2.md").exists():
        versions["planning"].append({
            "id": "v2",
            "name": "V2 - String Task Descriptions",
            "description": "New format: 'Use left arm to pick up the orange...'"
        })

    # Check for monitoring prompts
    if (prompts_dir / "monitoring_system.md").exists():
        versions["monitoring"].append({
            "id": "v1",
            "name": "V1 - Original",
            "description": "Original monitoring prompt"
        })
    if (prompts_dir / "monitoring_system_v2.md").exists():
        versions["monitoring"].append({
            "id": "v2",
            "name": "V2 - Updated",
            "description": "Updated monitoring prompt"
        })

    return versions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
