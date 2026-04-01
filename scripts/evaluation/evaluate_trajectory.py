#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import runpy
import sys

SCRIPT = Path(__file__).resolve().with_name("run_full_evaluation.py")


def main() -> None:
    sys.argv = [str(SCRIPT), *sys.argv[1:], "--trajectory-only"]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
