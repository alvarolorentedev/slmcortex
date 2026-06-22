from contextlib import contextmanager
from pathlib import Path

import pytest

import skill_lattice_coder.inference as inference


def test_all_inference_modes_use_expected_adapters(monkeypatch):
    loaded = []

    monkeypatch.setattr(
        inference, "require_adapter", lambda name, root=None: Path(name)
    )
    monkeypatch.setattr(
        inference,
        "adapter_metadata",
        lambda name, root=None: {"trainable_parameters": 10},
    )

    @contextmanager
    def composed(paths):
        assert [path.name for path in paths] == ["debugging_skill", "python_skill"]
        yield Path("composed")

    monkeypatch.setattr(inference, "temporary_composed_adapter", composed)
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda adapter: loaded.append(adapter) or ("model", "tokenizer"),
    )
    monkeypatch.setattr(inference, "generate_text", lambda *args: ("generated", 4, 2))

    assert inference.infer("base", "Write code").active_adapter_count == 0
    assert inference.infer("generic", "Write code").active_adapter_count == 1
    assert inference.infer(
        "single-skill", "Fix code", skill="debugging_skill"
    ).selected_skills == ["debugging_skill"]
    lattice = inference.infer(
        "lattice",
        "Fix this Python traceback",
        task_type="debugging",
        router_policy="legacy_rule_router",
    )
    assert lattice.selected_skills == ["debugging_skill", "python_skill"]
    assert [None, Path("generic"), Path("debugging_skill"), Path("composed")] == loaded


def test_single_skill_requires_skill_name():
    with pytest.raises(ValueError, match="required"):
        inference.infer("single-skill", "Fix code", dry_run=True)


def test_oracle_lattice_uses_supplied_skills(monkeypatch):
    monkeypatch.setattr(
        inference,
        "adapter_metadata",
        lambda name, root=None: {"trainable_parameters": 10},
    )
    result = inference.infer(
        "oracle-lattice",
        "ambiguous prompt",
        skills=["python_skill", "debugging_skill"],
        dry_run=True,
    )
    assert result.selected_skills == ["python_skill", "debugging_skill"]


def test_inference_reuses_loaded_model(monkeypatch):
    loads = []
    cache = {}
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda adapter: loads.append(adapter) or ("model", "tokenizer"),
    )
    monkeypatch.setattr(inference, "generate_text", lambda *args: ("generated", 1, 1))

    inference.infer("base", "first", model_cache=cache)
    inference.infer("base", "second", model_cache=cache)

    assert loads == [None]


def test_policy_base_fallback_loads_true_base_without_adapter_state(monkeypatch):
    loaded = []
    cache = {("python_skill",): ("adapter-model", "adapter-tokenizer")}
    monkeypatch.setattr(
        inference,
        "require_adapter",
        lambda *args: (_ for _ in ()).throw(AssertionError("adapter loaded")),
    )
    monkeypatch.setattr(
        inference,
        "load_model",
        lambda adapter: loaded.append(adapter) or ("base-model", "base-tokenizer"),
    )
    monkeypatch.setattr(inference, "generate_text", lambda *args: ("generated", 1, 1))

    result = inference.infer(
        "lattice",
        "Write a function",
        task_type="python_generation",
        router_policy="python_only_for_test_generation",
        model_cache=cache,
    )

    assert loaded == [None]
    assert result.selected_skills == []
    assert result.route.route_type == "base_fallback"
    assert result.active_adapter_count == 0
    assert result.active_adapter_parameters == 0
    assert cache[()] == ("base-model", "base-tokenizer")


def test_policy_inference_routes_debugging_and_tests():
    debugging = inference.infer(
        "lattice",
        "ignored",
        task_type="debugging",
        router_policy="python_only_for_test_generation",
        dry_run=True,
    )
    tests = inference.infer(
        "lattice",
        "ignored",
        task_type="test_generation",
        router_policy="python_only_for_test_generation",
        dry_run=True,
    )
    assert debugging.selected_skills == ["debugging_skill", "python_skill"]
    assert tests.selected_skills == ["python_skill", "test_generation_skill"]


def test_lattice_defaults_to_protected_router_and_legacy_is_explicit():
    protected = inference.infer(
        "lattice",
        "Write a Python function",
        task_type="python_generation",
        dry_run=True,
    )
    legacy = inference.infer(
        "lattice",
        "Write a Python function",
        task_type="python_generation",
        router_policy="legacy_rule_router",
        dry_run=True,
    )
    alias = inference.infer(
        "lattice",
        "Write a Python function",
        task_type="python_generation",
        router_policy="protected_skill_router",
        dry_run=True,
    )
    assert protected.selected_skills == []
    assert alias.selected_skills == []
    assert legacy.selected_skills == ["python_skill"]


def test_weighted_composition_reuses_existing_adapter_composer(monkeypatch):
    seen = []

    @contextmanager
    def composed(paths, weights=None):
        seen.append(([path.name for path in paths], weights))
        yield Path("weighted")

    monkeypatch.setattr(inference, "temporary_composed_adapter", composed)
    monkeypatch.setattr(
        inference, "require_adapter", lambda name, root=None: Path(name)
    )
    monkeypatch.setattr(
        inference,
        "adapter_metadata",
        lambda name, root=None: {"trainable_parameters": 10},
    )
    monkeypatch.setattr(inference, "load_model", lambda adapter: ("model", "tokenizer"))
    monkeypatch.setattr(inference, "generate_text", lambda *args: ("generated", 1, 1))

    inference.infer(
        "lattice",
        "ignored",
        task_type="debugging",
        composition_weights=[0.75, 0.25],
    )

    assert seen == [(["debugging_skill", "python_skill"], [0.75, 0.25])]
