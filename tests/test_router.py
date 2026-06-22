from skill_lattice_coder.schemas import RouteDecision
from skill_lattice_coder.router import (
    ProtectedSkillRouterWithoutFailureBorn,
    ProtectedSkillRouter,
    PythonOnlyForTestGenerationRouter,
    RuleRouter,
    SkillCortexRouterV1,
)


def test_router_handles_single_overlapping_and_ambiguous_prompts():
    router = RuleRouter()
    assert router.route("Write a Python function").selected_skills == ["python_skill"]
    assert router.route(
        "Fix this Python traceback and failing test"
    ).selected_skills == [
        "debugging_skill",
        "python_skill",
    ]
    assert router.route(
        "Generate pytest tests for this Python function"
    ).selected_skills == [
        "test_generation_skill",
        "python_skill",
    ]
    assert router.route("Explain this algorithm").selected_skills == ["python_skill"]


def test_explicit_dataset_tags_override_prompt_rules():
    decision = RuleRouter().route(
        "Write Python tests",
        tags=["debugging_skill", "python_skill", "test_generation_skill"],
    )
    assert decision.selected_skills == ["debugging_skill", "python_skill"]
    assert "tags" in decision.reason.lower()


def test_base_fallback_route_is_valid():
    decision = RouteDecision(
        [],
        1.0,
        "Base is stronger.",
        route_type="base_fallback",
    )
    assert decision.selected_skills == []
    assert decision.route_type == "base_fallback"


def test_python_only_for_test_generation_routes_exactly_by_task():
    router = PythonOnlyForTestGenerationRouter()
    python = router.route("python_generation")
    debugging = router.route("debugging")
    tests = router.route("test_generation")

    assert python.selected_skills == []
    assert python.route_type == "base_fallback"
    assert debugging.selected_skills == ["debugging_skill", "python_skill"]
    assert debugging.route_type == "adapter"
    assert tests.selected_skills == ["python_skill", "test_generation_skill"]
    assert tests.route_type == "adapter"


def test_protected_skill_router_is_the_architectural_alias():
    assert ProtectedSkillRouter is PythonOnlyForTestGenerationRouter


def test_skillcortex_router_v1_gates_promoted_alternating_skill():
    router = SkillCortexRouterV1()

    assert router.route("debugging", "alternating").selected_skills == [
        "debugging_skill",
        "python_skill",
        "alternating_skill",
    ]
    assert router.route("test_generation", "alternating").selected_skills == [
        "python_skill",
        "test_generation_skill",
        "alternating_skill",
    ]
    assert router.route("python_generation", "alternating").selected_skills == []
    assert router.route("debugging", "overlay") == ProtectedSkillRouterWithoutFailureBorn().route("debugging")
    assert router.route("test_generation", None) == ProtectedSkillRouterWithoutFailureBorn().route("test_generation")


def test_protected_router_without_failure_born_remains_the_validated_baseline():
    assert ProtectedSkillRouterWithoutFailureBorn is PythonOnlyForTestGenerationRouter
