import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .backends.legacy import ExecutionFixture, evaluator_backend, model_backend, trainer_backend
from .contracts import TASK_TYPES


aggregate_results = evaluator_backend.aggregate_results
extract_code = evaluator_backend.extract_code
fuzzy_match = evaluator_backend.fuzzy_match
generate_text = model_backend.generate_text
load_model = model_backend.load_model
python_syntax_valid = evaluator_backend.python_syntax_valid
research_metadata = trainer_backend.training_metadata
run_fixture = evaluator_backend.run_fixture
saved_parameter_count = trainer_backend.saved_parameter_count
training_config = trainer_backend.training_config
training_command = trainer_backend.build_generic_training_command


@dataclass(slots=True)
class ProductTrainingExample:
    id: str
    task_type: str
    prompt: str
    target: str
    execution: ExecutionFixture | None = None
    group: str | None = None
    metadata: dict | None = None
    semantic_family: str | None = None
    skills: list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be non-empty")
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {self.task_type}")
        if isinstance(self.execution, dict):
            self.execution = ExecutionFixture.from_dict(self.execution)
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a mapping")
        if self.semantic_family is not None and not isinstance(self.semantic_family, str):
            raise ValueError("semantic_family must be a string")
        if self.skills is not None:
            if not isinstance(self.skills, list) or any(
                not isinstance(item, str) or not item.strip() for item in self.skills
            ):
                raise ValueError("skills must be a list of non-empty strings")

    @classmethod
    def from_dict(cls, value: dict) -> "ProductTrainingExample":
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


def load_product_jsonl(path: str | Path) -> list[ProductTrainingExample]:
    examples: list[ProductTrainingExample] = []
    seen: set[str] = set()
    candidate = Path(path)
    with candidate.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                example = ProductTrainingExample.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{candidate}:{line_number}: {error}") from error
            if example.id in seen:
                raise ValueError(f"{candidate}:{line_number}: duplicate id {example.id}")
            seen.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"{candidate} is empty")
    return examples


def write_product_mlx_dataset(
    examples: list[ProductTrainingExample], directory: str | Path
) -> Path:
    candidate = Path(directory)
    candidate.mkdir(parents=True, exist_ok=True)
    valid = examples[::10] or examples[:1]
    train = [example for index, example in enumerate(examples) if index % 10] or list(examples)
    for name, rows in (("train.jsonl", train), ("valid.jsonl", valid)):
        (candidate / name).write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": example.prompt},
                            {"role": "assistant", "content": example.target},
                        ]
                    }
                )
                + "\n"
                for example in rows
            )
        )
    return candidate


def train_product_skill_to_run_directory(
    *,
    skill_id: str,
    train_dataset: Path,
    run_directory: Path,
    seed: int | None,
    force: bool,
) -> tuple[Path, dict]:
    run_directory.mkdir(parents=True, exist_ok=True)
    examples = load_product_jsonl(train_dataset)
    training_directory = run_directory / "training-data"
    adapter_directory = run_directory / "adapters" / skill_id
    if adapter_directory.exists() and any(adapter_directory.iterdir()) and not force:
        raise FileExistsError(f"{adapter_directory} exists; pass --force to replace it")
    if adapter_directory.exists():
        import shutil

        shutil.rmtree(adapter_directory)
    dataset_directory = write_product_mlx_dataset(examples, training_directory)
    command = training_command(dataset_directory, adapter_directory, rank=8, seed=seed)
    start = time.perf_counter()
    import subprocess

    subprocess.run(command, check=True)
    metadata = research_metadata(
        skill_id,
        examples,
        rank=8,
        elapsed=time.perf_counter() - start,
        seed=seed,
        iterations=training_config()["iterations"],
    )
    metadata["trainable_parameters"] = saved_parameter_count(adapter_directory)
    metadata["training_command"] = command
    (adapter_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return adapter_directory, metadata


def evaluate_product_skill_adapter(
    *,
    skill_id: str,
    dataset: Path,
    output: Path,
    adapter_dir: Path,
) -> Path:
    examples = load_product_jsonl(dataset)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    model_cache: dict = {}
    raw_path = output / "results.jsonl"
    with raw_path.open("w") as handle:
        for example in examples:
            for mode, resolved_adapter in (("base", None), ("single-skill", adapter_dir)):
                try:
                    generation = _generate_for_example(
                        example.prompt,
                        adapter_dir=resolved_adapter,
                        model_cache=model_cache,
                    )
                    text = extract_code(generation["generation"])
                    syntax = (
                        python_syntax_valid(text)
                        if example.task_type != "test_generation"
                        else None
                    )
                    execution = None
                    if example.execution:
                        execution, _ = run_fixture(example.execution, text)
                    row = {
                        "example_id": example.id,
                        "task_type": example.task_type,
                        "mode": mode,
                        "generation": text,
                        "exact_match": text.strip() == example.target.strip(),
                        "fuzzy_score": fuzzy_match(text, example.target),
                        "syntax_valid": syntax,
                        "execution_passed": execution,
                        "latency_seconds": generation["latency_seconds"],
                        "selected_skills": [skill_id] if resolved_adapter else [],
                        "active_adapter_count": 1 if resolved_adapter else 0,
                        "active_adapter_parameters": generation["active_adapter_parameters"],
                        "prompt_tokens": generation["prompt_tokens"],
                        "generated_tokens": generation["generated_tokens"],
                        "peak_memory_bytes": generation["peak_memory_bytes"],
                        "benchmark_group": example.group,
                    }
                except Exception as error:
                    row = {
                        "example_id": example.id,
                        "task_type": example.task_type,
                        "mode": mode,
                        "generation": "",
                        "exact_match": False,
                        "fuzzy_score": 0,
                        "syntax_valid": None,
                        "execution_passed": None,
                        "latency_seconds": 0,
                        "selected_skills": [skill_id] if resolved_adapter else [],
                        "active_adapter_count": 1 if resolved_adapter else 0,
                        "active_adapter_parameters": 0,
                        "prompt_tokens": None,
                        "generated_tokens": None,
                        "peak_memory_bytes": None,
                        "error": str(error),
                        "benchmark_group": example.group,
                    }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
    summary = aggregate_results(rows)
    tasks = {
        task: aggregate_results([row for row in rows if row["task_type"] == task])
        for task in sorted({row["task_type"] for row in rows})
    }
    payload = {
        "hypothesis": None,
        "modes": summary,
        "tasks": tasks,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    (output / "report.md").write_text(_evaluation_report(skill_id, summary, tasks))
    return output / "summary.json"


def _generate_for_example(
    prompt: str,
    *,
    adapter_dir: Path | None,
    model_cache: dict,
) -> dict:
    cache_key = str(adapter_dir.resolve()) if adapter_dir is not None else "__base__"
    cached = model_cache.get(cache_key)
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except ImportError:
        mx = None
    start = time.perf_counter()
    if cached is None:
        model, tokenizer = load_model(adapter_dir)
        model_cache[cache_key] = (model, tokenizer)
    else:
        model, tokenizer = cached
    generation, prompt_tokens, generated_tokens = generate_text(model, tokenizer, prompt)
    return {
        "generation": generation,
        "latency_seconds": time.perf_counter() - start,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "peak_memory_bytes": int(mx.get_peak_memory()) if mx else None,
        "active_adapter_parameters": (
            saved_parameter_count(adapter_dir) if adapter_dir is not None else 0
        ),
    }


def _evaluation_report(skill_id: str, summary: dict, tasks: dict) -> str:
    lines = [
        f"# SkillCortex Single Skill Evaluation: {skill_id}",
        "",
        "| Mode | Count | Fuzzy | Exact | Syntax | Execution | Active params |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("base", "single-skill"):
        if mode not in summary:
            continue
        value = summary[mode]
        lines.append(
            f"| {mode} | {value['count']} | {value['fuzzy_score']:.3f} | "
            f"{value['exact_match_rate']:.3f} | {_format(value['syntax_valid_rate'])} | "
            f"{_format(value['execution_pass_rate'])} | {value['active_adapter_parameters']:.0f} |"
        )
    lines.extend(["", "## By task", ""])
    for task, modes in tasks.items():
        scores = ", ".join(
            f"{mode}={values['fuzzy_score']:.3f}" for mode, values in sorted(modes.items())
        )
        lines.append(f"- `{task}`: {scores}")
    return "\n".join(lines) + "\n"


def _format(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"