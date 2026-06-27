from __future__ import annotations

from pathlib import Path
from typing import Any

from skill_lattice_coder import __version__ as legacy_version
from skill_lattice_coder.compose import (
    temporary_composed_adapter,
    validate_adapter_configs,
    validate_adapter_metadata,
)
from skill_lattice_coder.config import ARTIFACT_DIR, CONFIG_DIR, ROOT, base_config, training_config
from skill_lattice_coder.data import dataset_hash, load_jsonl, select_for_skill, write_mlx_dataset
from skill_lattice_coder.inference import infer
from skill_lattice_coder.metrics import aggregate_results, extract_code, fuzzy_match, python_syntax_valid
from skill_lattice_coder.model_loader import generate_text, load_model
from skill_lattice_coder.router import RuleRouter
from skill_lattice_coder.schemas import EvaluationResult, ExecutionFixture, SKILLS
from skill_lattice_coder.train_skill import _metadata as research_metadata
from skill_lattice_coder.train_skill import _saved_parameter_count, _training_command, build_skill_command
from skill_lattice_coder.utils import run_fixture

from .interfaces import AdapterCompositionBackend, EvaluatorBackend, ModelBackend, TrainerBackend


saved_parameter_count = _saved_parameter_count


class LegacyTrainerBackend(TrainerBackend):
    def training_config(self) -> dict[str, Any]:
        return training_config()

    def training_metadata(
        self,
        skill_id: str,
        examples: list[Any],
        *,
        rank: int,
        elapsed: float,
        seed: int | None,
        iterations: int,
    ) -> dict[str, Any]:
        return research_metadata(
            skill_id,
            examples,
            rank=rank,
            elapsed=elapsed,
            seed=seed,
            iterations=iterations,
        )

    def saved_parameter_count(self, adapter_directory: Path) -> int:
        return _saved_parameter_count(adapter_directory)

    def build_skill_command(
        self,
        skill: str,
        dataset_directory: Path,
        adapter_directory: Path,
        *,
        seed: int | None,
    ) -> list[str]:
        return build_skill_command(skill, dataset_directory, adapter_directory, seed=seed)

    def build_generic_training_command(
        self,
        dataset_directory: Path,
        adapter_directory: Path,
        *,
        rank: int,
        seed: int | None,
    ) -> list[str]:
        return _training_command(dataset_directory, adapter_directory, rank=rank, seed=seed)


class LegacyEvaluatorBackend(EvaluatorBackend):
    def aggregate_results(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregate_results(rows)

    def extract_code(self, generation: str) -> str:
        return extract_code(generation)

    def fuzzy_match(self, left: str, right: str) -> float:
        return fuzzy_match(left, right)

    def python_syntax_valid(self, code: str) -> bool:
        return python_syntax_valid(code)

    def run_fixture(self, fixture: Any, code: str) -> tuple[bool | None, str]:
        return run_fixture(fixture, code)


class LegacyModelBackend(ModelBackend):
    def load_model(
        self,
        adapter: Path | None = None,
        *,
        model_name: str | None = None,
    ) -> tuple[object, object]:
        if model_name is None:
            return load_model(adapter)
        return load_model(adapter=adapter, model_name=model_name)

    def generate_text(
        self,
        model: object,
        tokenizer: object,
        prompt: str | None = None,
        *,
        messages: list[dict[str, str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[str, int | None, int | None]:
        return generate_text(
            model,
            tokenizer,
            prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def route_text(self, text: str) -> list[str]:
        return list(RuleRouter().route(text).selected_skills)


class LegacyAdapterCompositionBackend(AdapterCompositionBackend):
    def validate_adapter_configs(self, configs: list[dict[str, Any]]) -> None:
        validate_adapter_configs(configs)

    def validate_adapter_metadata(self, metadata: list[dict[str, Any]]) -> None:
        validate_adapter_metadata(metadata)

    def temporary_composed_adapter(self, adapter_paths: list[Path]):
        return temporary_composed_adapter(adapter_paths)


trainer_backend = LegacyTrainerBackend()
evaluator_backend = LegacyEvaluatorBackend()
model_backend = LegacyModelBackend()
adapter_composition_backend = LegacyAdapterCompositionBackend()
