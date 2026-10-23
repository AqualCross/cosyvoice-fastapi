"""Launcher for the CosyVoice FastAPI server.

Examples:
    uv run run_api.py --gpu 0 --port 8001
    uv run run_api.py --default-model cosyvoice3 --fp16
    uv run run_api.py --api-token secret --repo ../CosyVoice
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CosyVoice API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to (default: 8001)")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="Physical GPU index to use (sets CUDA_VISIBLE_DEVICES before torch starts)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device hint: 'cpu' hides all GPUs; anything else keeps the CUDA_VISIBLE_DEVICES choice",
    )
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Path to the cloned CosyVoice repository (default: ../CosyVoice)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model registry, e.g. cosyvoice2=<dir>,cosyvoice3=<dir>",
    )
    parser.add_argument("--default-model", type=str, default="cosyvoice2", help="Model id loaded at startup")
    parser.add_argument("--voices-dir", type=str, default="voices", help="Directory with reference voices")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=False, help="Use fp16 inference")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Use fp32 inference (default)")
    parser.add_argument(
        "--warmup", dest="warmup", action="store_true", default=True,
        help="Run one warmup inference after loading (default: on)",
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", help="Skip the warmup inference")
    parser.add_argument("--max-loaded", type=int, default=1, help="How many models stay resident (default: 1)")
    parser.add_argument("--api-token", type=str, default=None, help="Require this Bearer token on /v1/*")
    parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level (default: info)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device and args.device.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.repo:
        os.environ["COSYVOICE_REPO"] = args.repo
    if args.models:
        os.environ["COSYVOICE_MODELS"] = args.models
    os.environ["COSYVOICE_DEFAULT_MODEL"] = args.default_model
    os.environ["COSYVOICE_VOICES_DIR"] = args.voices_dir
    os.environ["COSYVOICE_FP16"] = "1" if args.fp16 else "0"
    os.environ["COSYVOICE_WARMUP"] = "1" if args.warmup else "0"
    os.environ["COSYVOICE_MAX_LOADED"] = str(args.max_loaded)
    if args.api_token:
        os.environ["COSYVOICE_API_TOKEN"] = args.api_token

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Install the requirements first.", file=sys.stderr)
        return 1

    uvicorn.run("api.main:app", host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
