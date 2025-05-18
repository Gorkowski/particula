"""
This script calls the Aider CLI to convert a Python file to Taichi code.
It uses the Aider model to generate the code and then reflects on the changes made.
"""

import argparse
import subprocess
from pathlib import Path

GUIDE_PATH = Path(
    r"C:\GitHub\particula\particula\backend\code_naming_guide.md"
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
            "and revise it to follow the "
            "code_naming_guide.md. Modify only the names"
            f"and not the code logic in {file_path.name}."
            " If the file already looks good, review it and "
            "make sure it follows the guide."
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
            "Double-check that the guide has been correctly followed.",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a benchmark with aider."
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to a Python file OR to a directory that contains Python files. "
            "If a directory is supplied, every *.py file in that directory "
            "(recursively) is converted one-by-one, skipping any __init__.py files."
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
        for py_file in target.rglob("*.py"):      # recursive
            if py_file.name == "__init__.py":     # skip package initializers
                continue
            convert(py_file, args.prompt)
    else:
        convert(target, args.prompt)
