"""
This script calls the Aider CLI to convert a Python file to Taichi code.
It uses the Aider model to generate the code and then reflects on the changes made.
"""

import argparse
import subprocess
from pathlib import Path


def _call_aider(extra_args: list[str]) -> None:
    """Run aider with the provided extra CLI flags."""
    base_cmd = [
        "aider",
        "--model", "o3",
        "--architect",
        "--reasoning-effort", "high",
        "--editor-model", "gpt-4.1",
        "--no-detect-urls",
        "--yes-always",
        "--read",
    ] + extra_args
    subprocess.run(base_cmd, check=True)


def convert(file_path: Path, prompt: str) -> None:
    """
    Run the two-step aider conversion on *file_path* using *prompt*.

    The first run is done with --no-auto-commit; the second (reflection)
    run commits automatically.
    """
    file_str = str(file_path.resolve())

    # first pass
    _call_aider([
        file_str,
        "--message",
        f"CREATE a taichi version of the python file provided follow the guide in {prompt}",
    ])

    # reflection / verification pass
    _call_aider([
        file_str,
        "--message",
        f"Double check the taichi conversion of the python file has been corretly implemented, following the guide: {prompt}",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Python file to Taichi using aider."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the Python file that should be converted.",
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Prompt describing the desired conversion.",
    )
    args = parser.parse_args()

    convert(args.file, args.prompt)
