#!/usr/bin/env python3
"""
Run the Prompt Refinement Web App.

Usage:
    python prompt_refinement/run_app.py 

Options:
              Default is real mode (requires GPU or API key).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def get_local_ip():
    """Get the local IP address for network access."""
    import socket
    try:
        # Connect to an external address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main():
    parser = argparse.ArgumentParser(description="Prompt Refinement Web App")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    import prompt_refinement.app.main as app_module

    local_ip = get_local_ip()

    print(f"\n{'=' * 60}")
    print("  PROMPT REFINEMENT FACTORY")
    print(f"{'=' * 60}")
    print(f"  Local:    http://localhost:{args.port}")
    print(f"  Network:  http://{local_ip}:{args.port}")
    print(f"  API Docs: http://{local_ip}:{args.port}/docs")
    print(f"{'=' * 60}\n")

    import uvicorn

    uvicorn.run(
        "prompt_refinement.app.main:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
