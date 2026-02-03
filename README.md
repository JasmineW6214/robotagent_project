# Tools for robot agent project

- **`prompt-refinement-app/`** — Web app for iterating on VLM prompts against real robot execution traces. FastAPI backend with a simple HTML/JS frontend. Supports multiple inference backends: Qwen3-VL (Transformers), vLLM, TensorRT-LLM, and Claude.
- **`model-introspection/`** — Analysis scripts for understanding VLM internals: attention patterns, denoising processes, hallucination detection, and feature comparison.
- **`tools/`** — Evaluation and dataset visualization utilities.
