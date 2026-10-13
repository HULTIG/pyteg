"""Run the whole reproduction pipeline, step by step.

Usage:  python pipeline/run_all.py [--from-step N]

This is the same as running step1 ... step7 one after the other. Each step
can also be run on its own; see REPRODUCING.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    "step1_download.py",
    "step2_prepare_corpora.py",
    "step3_extract_mwus.py",
    "step4_build_indexes.py",
    "step5_prepare_benchmarks.py",
    "step6_run_experiments.py",
    "step7_write_report.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-step", type=int, default=1, choices=range(1, len(STEPS) + 1))
    args = parser.parse_args()
    for number, script in enumerate(STEPS, start=1):
        if number < args.from_step:
            continue
        print(f"\n=== step {number}: {script} ===", flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)])
        if result.returncode != 0:
            print(f"step {number} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
