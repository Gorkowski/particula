# /// script
# dependencies = [
#   "aider-chat",
#   "openai"
# ]
# ///

"""director.py
Self‑directed AI coding assistant that iteratively writes, executes, and
self‑evaluates code until it satisfies a user‑defined specification.

The *Director* class orchestrates three roles:

1. **Coder**   – Generates or edits source files with an LLM.
2. **Executor** – Runs an arbitrary shell command (e.g. tests).
3. **Evaluator** – Uses a second LLM to judge success and give feedback.

Configuration is provided via a YAML file (see `DirectorConfig`).
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    """Structured result returned by the *Evaluator* LLM."""

    success: bool
    feedback: Optional[str] = None


class DirectorConfig(BaseModel):
    """Runtime configuration loaded from a YAML file.

    Attributes:
        prompt: Either a Markdown file path or the literal prompt text.
        coder_model: Name of the LLM used for code generation.
        evaluator_model: Name of the LLM used to evaluate execution output.
        max_iterations: Maximum *coder → execute → evaluate* cycles.
        execution_command: Shell command executed each cycle.
        context_editable: List of paths the *Coder* may modify.
        context_read_only: List of paths provided as read‑only context.
        evaluator: Currently only "default" is implemented.
    """

    prompt: str
    coder_model: str = Field(..., alias="coder_model")
    evaluator_model: Literal["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]
    max_iterations: int = Field(gt=0)
    execution_command: str
    context_editable: List[str]
    context_read_only: List[str] = []
    evaluator: Literal["default"] = "default"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_ALLOWED_EVALUATOR_MODELS: set[str] = {
    "gpt-4.1",
    "o3",
}


def _read_text(path: Path) -> str:
    with path.open() as fp:
        return fp.read()


# ---------------------------------------------------------------------------
# Director implementation
# ---------------------------------------------------------------------------


class Director:
    """Coordinator that drives the *Coder → Executor → Evaluator* cycle."""

    log_file = Path("director.log")

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(self, config_path: Path | str) -> None:
        self.config = self._validate_config(Path(config_path))
        self.llm_client = OpenAI()
        self._setup_logging()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def direct(self) -> None:
        """Run up to *max_iterations* improvement cycles."""

        evaluation = EvaluationResult(success=False)
        execution_output = ""

        for iteration in range(self.config.max_iterations):
            logging.info(
                "Iteration %d/%d", iteration + 1, self.config.max_iterations
            )

            prompt = self._build_coder_prompt(
                iteration, self.config.prompt, execution_output, evaluation
            )
            self._invoke_coder(prompt)

            execution_output = self._execute_command()
            evaluation = self._evaluate_run(execution_output)

            if evaluation.success:
                logging.info("Success after %d iterations", iteration + 1)
                return

            logging.info(
                "Continuing – %d attempts remain",
                self.config.max_iterations - iteration - 1,
            )

        logging.warning(
            "Failed to satisfy requirements within the iteration limit",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_file, mode="a"),
            ],
        )

    @staticmethod
    def _validate_config(path: Path) -> DirectorConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")


        # If the prompt is a Markdown file, replace with its contents.
        prompt_candidate = Path(data["prompt"])
        if prompt_candidate.suffix == ".md" and prompt_candidate.exists():
            data["prompt"] = _read_text(prompt_candidate)

        try:
            cfg = DirectorConfig(**data)
        except ValidationError as err:
            raise ValueError(f"Invalid configuration: {err}") from err

        # Basic path validations.
        for p in cfg.context_editable + cfg.context_read_only:
            if not Path(p).exists():
                raise FileNotFoundError(f"Context file not found: {p}")

        return cfg

    # ---------------------------- Coder --------------------------------

    def _invoke_coder(self, prompt: str) -> None:
        model = Model(self.config.coder_model)
        coder = Coder.create(
            main_model=model,
            io=InputOutput(yes=True),
            fnames=self.config.context_editable,
            read_only_fnames=self.config.context_read_only,
            auto_commits=False,
            suggest_shell_commands=False,
        )
        coder.run(prompt)

    # --------------------------- Executor ------------------------------

    def _execute_command(self) -> str:
        """Run *execution_command* and return combined stdout/stderr."""

        logging.info("Executing: %s", self.config.execution_command)
        result = subprocess.run(
            self.config.execution_command.split(),
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        logging.debug("Execution output:\n%s", output)
        return output

    # --------------------------- Evaluator -----------------------------

    def _evaluate_run(self, execution_output: str) -> EvaluationResult:
        if self.config.evaluator != "default":
            raise ValueError("Only the default evaluator is implemented.")

        prompt = self._build_evaluation_prompt(execution_output)
        logging.debug("Evaluation prompt:\n%s", prompt)

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.config.evaluator_model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_response = completion.choices[0].message.content
            logging.debug("Raw evaluation response: %s", raw_response)

            parsed_json = self._extract_json(raw_response)
            return EvaluationResult.model_validate_json(parsed_json)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Primary evaluation failed – %s", exc)
            raise

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_coder_prompt(
        self,
        iteration: int,
        base_prompt: str,
        execution_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        if iteration == 0:
            return base_prompt

        remaining = self.config.max_iterations - iteration
        return (
            "# Generate the next iteration of code to achieve the user's desired "
            "result based on their original instructions and the feedback from "
            "the previous attempt.\n"
            "> Generate a new prompt in the same style as the original instructions "
            "for the next iteration of code.\n\n"
            f"## This is your {iteration}th attempt to generate the code.\n"
            f"> You have {remaining} attempts remaining.\n\n"
            "## User's original instructions:\n"
            + base_prompt
            + "\n\n"
            + "## Previous execution output:\n"
            + execution_output
            + "\n\n"
            + "## Feedback on previous attempt:\n"
            + (evaluation.feedback or "<none>")
        )

    def _build_evaluation_prompt(self, execution_output: str) -> str:
        editable = {
            Path(p).name: _read_text(Path(p))
            for p in self.config.context_editable
        }
        readonly = {
            Path(p).name: _read_text(Path(p))
            for p in self.config.context_read_only
        }

        spec = {"success": "bool", "feedback": "str | None"}

        return (
            "Evaluate the execution output and decide whether the user's desired "
            "result has been achieved. Use the following inputs:\n\n"
            "### User's Desired Result:\n"
            + self.config.prompt
            + "\n\n"
            + "### Editable Files:\n"
            + json.dumps(editable)
            + "\n\n"
            + "### Read‑Only Files:\n"
            + json.dumps(readonly)
            + "\n\n"
            + "### Execution Command:\n"
            + self.config.execution_command
            + "\n\n"
            + "### Execution Output:\n"
            + execution_output
            + "\n\n"
            + "### Checklist:\n"
            "- Does the execution output indicate success?\n"
            "- Are all tasks in the user's desired result satisfied?\n"
            "- Ignore warnings.\n\n" + "### Response Format:\n"
            "> Return **exactly** JSON compatible with `json.loads`, no extra text.\n\n"
            + "Schema: "
            + json.dumps(spec)
        )

    # ------------------------------------------------------------------
    # JSON extraction utility
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Strip Markdown fences and return a JSON string."""
        if "```" in raw:
            raw = raw.split("```", 1)[-1].split("```", 1)[0]
        return raw.strip()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:  # noqa: D401
    """Parse command‑line arguments."""
    parser
