"""YAML loading helper with lazy dependency import.

The loader avoids importing :mod:`yaml` at module import time to prevent
an optional dependency from becoming mandatory for users who do not need
YAML support.
"""

from pathlib import Path
from typing import Any, TextIO, Union


def _read_stream(stream: TextIO) -> str:
    """Read YAML content from an open text stream."""

    return stream.read()


def load_yaml(
    path_or_str: Union[str, Path, TextIO], *, from_str: bool = False
) -> Any:
    """Load YAML content from a file path, string, or text stream.

    Args:
        path_or_str: File path, YAML string content, or text IO stream.
        from_str: If True, treat ``path_or_str`` as YAML text content. If
            False, ``path_or_str`` must be a file path that will be read
            as UTF-8 text.

    Returns:
        Parsed YAML content produced by ``yaml.safe_load``.

    Raises:
        ImportError: If :mod:`yaml` is not installed.
        FileNotFoundError: If the provided path does not exist.
        OSError: If the file cannot be read.
        ValueError: If YAML parsing fails.
        TypeError: If an unsupported input type is provided.
    """

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install pyyaml to use load_yaml") from exc

    if from_str:
        if hasattr(path_or_str, "read"):
            content = _read_stream(path_or_str)  # type: ignore[arg-type]
        else:
            content = str(path_or_str)
    else:
        if isinstance(path_or_str, (str, Path)):
            path = Path(path_or_str)
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                message = f"Failed to read YAML file {path}: {exc}"
                raise type(exc)(message).with_traceback(
                    exc.__traceback__
                ) from exc
        else:
            raise TypeError(
                "path_or_str must be a path-like object when from_str is False"
            )

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML: {exc}") from exc
