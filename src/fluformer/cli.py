from __future__ import annotations

import argparse
import json

import torch

from . import __version__
from .training import resolve_device


def _doctor() -> int:
    payload = {
        "fluformer": __version__,
        "torch": torch.__version__,
        "default_device": str(resolve_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fluformer", description="Reusable Fluformer model components")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="report PyTorch and accelerator availability")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.print_help()
    return 0
