"""
This script calls the Aider CLI to convert a Python file to Taichi code.
It uses the Aider model to generate the code and then reflects on the changes made.
"""

import argparse
import subprocess
from pathlib import Path

GUIDE_PATH = Path(
    r"C:\GitHub\particula\particula\backend\taichi_class_development_guide.md"
)


def _call_aider(extra_args: list[str]) -> None:
    """Run aider with the provided extra CLI flags."""
    base_cmd = [
        "aider",
        "--model",
        "o3",
        "--architect",
        "--reasoning-effort",
        "high",
        "--editor-model",
        "gpt-4.1",
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
            "taichi_class_development_guide.md. DO NOT modify"
            f" the original file {file_path.name} or the guide."
            " If the file is already in taichi, review it and "
            "make sure it follows the guide and has tests."
        )
    file_str = str(file_path.resolve())

    # first pass
    _call_aider(
        [
            "--read",
            str(GUIDE_PATH),  # add the guide as read-only
            file_str,
            "--message",
            prompt,  # <- use generated or user-supplied prompt
        ]
    )

    # reflection / verification pass
    _call_aider(
        [
            "--read",
            str(GUIDE_PATH),  # add the guide again for the reflection pass
            file_str,
            "--message",
            "Double-check that the Taichi conversion has been correctly implemented.",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Python file to Taichi using aider."
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a Python file OR to a directory that contains Python files. "
            "If a directory is supplied, every *.py file in that directory "
            "(non-recursive) is converted one-by-one."
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,  # <- no longer required
        help="Custom prompt for the conversion; if omitted, a default prompt is used.",
    )
    args = parser.parse_args()

    target = args.path.resolve()
    if target.is_dir():
        for py_file in target.glob("*.py"):  # non-recursive
            convert(py_file, args.prompt)
    else:
        convert(target, args.prompt)
