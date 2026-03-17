"""Load and validate pipeline configuration from config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from config.exceptions import PipelineError


@dataclass
class SourceConfig:
    schema: str
    table: str
    primary_key: str


@dataclass
class TargetConfig:
    column: str


@dataclass
class LabelConfig:
    name: str
    description: str = ""


@dataclass
class PromptConfig:
    system_message: str
    classification_context: str
    instructions_template: str
    response_key: str


@dataclass
class PipelineConfig:
    source: SourceConfig
    target: TargetConfig
    context_columns: List[str]
    display_column: str
    labels: List[LabelConfig]
    fallback_label: str
    prompt: PromptConfig
    batch_size: int
    max_rpm: int
    test_row_limit: int

    # --- Derived helpers ---

    @property
    def label_names(self) -> List[str]:
        return [l.name for l in self.labels]

    @property
    def label_descriptions(self) -> Dict[str, str]:
        return {l.name: l.description for l in self.labels}


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config from YAML file and validate.

    Resolution order for config path:
      1. Explicit ``path`` argument
      2. ``CONFIG_PATH`` environment variable
      3. ``config.yaml`` in the project root (two levels up from this file)
    """
    if path is None:
        path = os.getenv("CONFIG_PATH")
    if path is None:
        # Default: project_root/config.yaml
        path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    path = Path(path)

    if not path.exists():
        raise PipelineError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise PipelineError("Config file must be a YAML mapping")

    return _parse_and_validate(raw, path)


def _parse_and_validate(raw: dict, path: Path) -> PipelineConfig:
    """Parse raw YAML dict into a validated PipelineConfig."""

    def _require(section: dict, key: str, context: str) -> object:
        if key not in section:
            raise PipelineError(f"Missing required config key '{key}' in {context} ({path})")
        return section[key]

    # --- source ---
    source_raw = _require(raw, "source", "root")
    source = SourceConfig(
        schema=str(_require(source_raw, "schema", "source")),
        table=str(_require(source_raw, "table", "source")),
        primary_key=str(_require(source_raw, "primary_key", "source")),
    )

    # --- target ---
    target_raw = _require(raw, "target", "root")
    target = TargetConfig(column=str(_require(target_raw, "column", "target")))

    # --- context_columns ---
    context_columns = list(_require(raw, "context_columns", "root"))
    if not context_columns:
        raise PipelineError(f"context_columns must be non-empty ({path})")

    # --- display_column ---
    display_column = str(raw.get("display_column", context_columns[0]))

    # --- labels ---
    labels_raw = _require(raw, "labels", "root")
    if not labels_raw:
        raise PipelineError(f"labels must be non-empty ({path})")
    labels = []
    for i, entry in enumerate(labels_raw):
        if isinstance(entry, str):
            labels.append(LabelConfig(name=entry))
        elif isinstance(entry, dict):
            name = entry.get("name")
            if not name:
                raise PipelineError(f"Label at index {i} missing 'name' ({path})")
            labels.append(LabelConfig(name=str(name), description=str(entry.get("description", ""))))
        else:
            raise PipelineError(f"Invalid label at index {i}: {entry} ({path})")

    # --- fallback_label ---
    fallback_label = str(_require(raw, "fallback_label", "root"))
    label_name_set = {l.name for l in labels}
    if fallback_label not in label_name_set:
        raise PipelineError(
            f"fallback_label '{fallback_label}' not found in labels: {sorted(label_name_set)} ({path})"
        )

    # --- prompt ---
    prompt_raw = _require(raw, "prompt", "root")
    response_key = str(_require(prompt_raw, "response_key", "prompt"))
    if not response_key:
        raise PipelineError(f"prompt.response_key must be non-empty ({path})")

    instructions_template = str(_require(prompt_raw, "instructions_template", "prompt"))
    # Validate placeholders
    for placeholder in ("{columns}", "{fallback_label}", "{response_key}"):
        if placeholder not in instructions_template:
            raise PipelineError(
                f"prompt.instructions_template must contain '{placeholder}' ({path})"
            )

    prompt = PromptConfig(
        system_message=str(_require(prompt_raw, "system_message", "prompt")),
        classification_context=str(_require(prompt_raw, "classification_context", "prompt")),
        instructions_template=instructions_template,
        response_key=response_key,
    )

    # --- settings ---
    settings_raw = raw.get("settings", {})
    batch_size = int(settings_raw.get("batch_size", 10))
    max_rpm = int(settings_raw.get("max_rpm", 30))
    test_row_limit = int(settings_raw.get("test_row_limit", 100))

    return PipelineConfig(
        source=source,
        target=target,
        context_columns=context_columns,
        display_column=display_column,
        labels=labels,
        fallback_label=fallback_label,
        prompt=prompt,
        batch_size=batch_size,
        max_rpm=max_rpm,
        test_row_limit=test_row_limit,
    )


__all__ = [
    "PipelineConfig",
    "SourceConfig",
    "TargetConfig",
    "LabelConfig",
    "PromptConfig",
    "load_config",
]
