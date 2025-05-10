"""
This script calls the Aider CLI to convert a Python file to Taichi code.
It uses the Aider model to generate the code and then reflects on the changes made.
"""

import argparse
import subprocess
from pathlib import Path

GUIDE_PATH = (Path(__file__).resolve().parent
              / "taichi_conversion_guide.md")    # <-- hard path to the guide


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
        # "--read",                # DELETE this line
    ] + extra_args
    subprocess.run(base_cmd, check=True)


def convert(file_path: Path, prompt: str | None = None) -> None:
    """
    Run the two-step aider conversion on *file_path*.

    If *prompt* is None, a default prompt is generated that asks aider to
    convert the given Python file to a Taichi version following the guide.
    """
    if prompt is None:
        prompt = (
            f"Take the following python file {file_path.name} "
            "and convert it to a taichi version following the "
            "taichi_conversion_guide.md."
        )
    file_str = str(file_path.resolve())

    # first pass
    _call_aider([
        "--read", str(GUIDE_PATH),          # add the guide as read-only
        file_str,
        "--message", prompt,          # <- use generated or user-supplied prompt
    ])

    # reflection / verification pass
    _call_aider([
        "--read", str(GUIDE_PATH),          # add the guide again for the reflection pass
        file_str,
        "--message",
        "Double-check that the Taichi conversion has been correctly implemented.",
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
        default=None,                # <- no longer required
        help="Custom prompt for the conversion; if omitted, a default prompt is used.",
    )
    args = parser.parse_args()

    convert(args.file, args.prompt)
