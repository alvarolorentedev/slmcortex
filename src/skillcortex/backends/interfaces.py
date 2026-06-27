from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TrainerBackend(Protocol):
    def training_config(self) -> dict[str, Any]: ...

    def training_metadata(
        self,
        skill_id: str,
        examples: list[Any],
        *,
        rank: int,
        elapsed: float,
        seed: int | None,
        iterations: int,
    ) -> dict[str, Any]: ...

    def saved_parameter_count(self, adapter_directory: Path) -> int: ...

    def build_skill_command(
        self,
        skill: str,
        dataset_directory: Path,
        adapter_directory: Path,
        *,
        seed: int | None,
    ) -> list[str]: ...

    def build_generic_training_command(
        self,
        dataset_directory: Path,
        adapter_directory: Path,
        *,
        rank: int,
        seed: int | None,
    ) -> list[str]: ...


class EvaluatorBackend(Protocol):
    def aggregate_results(self, rows: list[dict[str, Any]]) -> dict[str, Any]: ...

    def extract_code(self, generation: str) -> str: ...

    def fuzzy_match(self, left: str, right: str) -> float: ...

    def python_syntax_valid(self, code: str) -> bool: ...

    def run_fixture(self, fixture: Any, code: str) -> tuple[bool | None, str]: ...


class ModelBackend(Protocol):
    def load_model(
        self,
        adapter: Path | None = None,
        *,
        model_name: str | None = None,
    ) -> tuple[object, object]: ...

    def generate_text(
        self,
        model: object,
        tokenizer: object,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, int | None, int | None]: ...

    def route_text(self, text: str) -> list[str]: ...


class AdapterCompositionBackend(Protocol):
    def validate_adapter_configs(self, configs: list[dict[str, Any]]) -> None: ...

    def validate_adapter_metadata(self, metadata: list[dict[str, Any]]) -> None: ...

    def temporary_composed_adapter(self, adapter_paths: list[Path]): ...
