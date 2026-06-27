import json
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from wsgiref.simple_server import make_server

from .backends.legacy import adapter_composition_backend, model_backend
from .contracts import TASK_TYPES
from .composer import _package_fingerprint
from .packaging import _read_json, _read_yaml, validate_skill_package


temporary_composed_adapter = adapter_composition_backend.temporary_composed_adapter
generate_text = model_backend.generate_text
load_model = model_backend.load_model


REQUIRED_RUNTIME_FILES = (
    "composition.yaml",
    "router_config.json",
    "active_skills.json",
    "compatibility_report.json",
    "budget_report.json",
    "checksums.json",
)


@dataclass(slots=True)
class RuntimeSkill:
    skill_id: str
    name: str
    version: str
    package_path: Path
    adapter_path: Path
    fingerprint: str
    allowed_task_types: list[str]
    activation: dict[str, Any]
    trainable_parameters: int


@dataclass(slots=True)
class RuntimeBundle:
    path: Path
    name: str
    runtime_model: str
    source_model: str
    quantization: str
    strategy: str
    routes: list[dict[str, Any]]
    skills: dict[str, RuntimeSkill]
    compatibility_report: dict[str, Any]
    budget_report: dict[str, Any]
    checksums: dict[str, Any]


@dataclass(slots=True)
class RuntimeRouteDecision:
    selected_skills: list[str]
    confidence: float
    reason: str
    route_type: str = "adapter"

    def __post_init__(self) -> None:
        if any(not isinstance(skill_id, str) or not skill_id.strip() for skill_id in self.selected_skills):
            raise ValueError("selected_skills must contain non-empty skill ids")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not isinstance(self.route_type, str) or not self.route_type.strip():
            raise ValueError("route_type must be non-empty")


class SkillRuntime:
    def __init__(self, bundle: RuntimeBundle):
        self.bundle = bundle
        self._cache: dict[tuple[str, ...], tuple[object, object]] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, runtime_path: Path) -> "SkillRuntime":
        return cls(load_runtime_bundle(runtime_path))

    def validate(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "runtime": self.bundle.name,
            "path": str(self.bundle.path),
            "skills": sorted(self.bundle.skills),
            "runtime_model": self.bundle.runtime_model,
            "strategy": self.bundle.strategy,
        }

    def infer(
        self,
        *,
        prompt: str | None = None,
        system: str | None = None,
        messages: list[dict[str, str]] | None = None,
        task_type: str | None = None,
        semantic_family: str | None = None,
        skill_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        resolved_messages = _normalize_messages(prompt=prompt, system=system, messages=messages)
        decision = self.route(
            messages=resolved_messages,
            task_type=task_type,
            semantic_family=semantic_family,
            skill_override=skill_override,
        )
        active_parameters = sum(
            self.bundle.skills[skill_id].trainable_parameters
            for skill_id in decision.selected_skills
        )
        if dry_run:
            return {
                "status": "dry-run",
                "runtime": self.bundle.name,
                "task_type": decision.reason.split("task_type=", 1)[1].split(" ", 1)[0],
                "semantic_family": semantic_family,
                "route_type": decision.route_type,
                "selected_skills": decision.selected_skills,
                "reason": decision.reason,
                "active_adapter_count": len(decision.selected_skills),
                "active_adapter_parameters": active_parameters,
            }

        start = time.perf_counter()
        model, tokenizer = self._get_model(tuple(decision.selected_skills))
        generation, prompt_tokens, generated_tokens = generate_text(
            model,
            tokenizer,
            messages=resolved_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "status": "complete",
            "runtime": self.bundle.name,
            "task_type": decision.reason.split("task_type=", 1)[1].split(" ", 1)[0],
            "semantic_family": semantic_family,
            "route_type": decision.route_type,
            "selected_skills": decision.selected_skills,
            "reason": decision.reason,
            "generation": generation,
            "latency_seconds": time.perf_counter() - start,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "active_adapter_count": len(decision.selected_skills),
            "active_adapter_parameters": active_parameters,
        }

    def route(
        self,
        *,
        messages: list[dict[str, str]],
        task_type: str | None = None,
        semantic_family: str | None = None,
        skill_override: str | None = None,
    ) -> RuntimeRouteDecision:
        if skill_override is not None:
            if skill_override not in self.bundle.skills:
                raise ValueError(f"unknown runtime skill override: {skill_override}")
            return RuntimeRouteDecision(
                [skill_override],
                1.0,
                f"explicit skill override selected task_type={task_type or 'python_generation'}",
            )

        resolved_task_type = task_type or _infer_task_type(messages)
        if resolved_task_type not in TASK_TYPES:
            raise ValueError(f"unknown task_type: {resolved_task_type}")
        route = _select_route(self.bundle.routes, resolved_task_type, semantic_family)
        return RuntimeRouteDecision(
            list(route["selected_skills"]),
            1.0,
            f"runtime bundle route {route['route_id']} selected task_type={resolved_task_type}",
            route_type=route["route_type"],
        )

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_chat_request(payload, runtime_name=self.bundle.name)
        result = self.infer(
            messages=normalized["messages"],
            task_type=normalized["task_type"],
            semantic_family=normalized["semantic_family"],
            skill_override=normalized["skill_override"],
            max_tokens=normalized["max_tokens"],
            temperature=normalized["temperature"],
            dry_run=False,
        )
        prompt_tokens = result.get("prompt_tokens") or 0
        generated_tokens = result.get("generated_tokens") or 0
        return {
            "id": f"chatcmpl-skillcortex-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.bundle.name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["generation"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generated_tokens,
                "total_tokens": prompt_tokens + generated_tokens,
            },
        }

    def _get_model(self, selected_skills: tuple[str, ...]) -> tuple[object, object]:
        cached = self._cache.get(selected_skills)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._cache.get(selected_skills)
            if cached is not None:
                return cached
            adapter_paths = [self.bundle.skills[skill_id].adapter_path for skill_id in selected_skills]
            if not adapter_paths:
                model, tokenizer = load_model(model_name=self.bundle.runtime_model)
            else:
                adapter_context = (
                    temporary_composed_adapter(adapter_paths)
                    if len(adapter_paths) > 1
                    else nullcontext(adapter_paths[0])
                )
                with adapter_context as adapter_path:
                    model, tokenizer = load_model(
                        adapter=adapter_path,
                        model_name=self.bundle.runtime_model,
                    )
            self._cache[selected_skills] = (model, tokenizer)
            return model, tokenizer


class OpenAICompatApp:
    def __init__(self, runtime: SkillRuntime):
        self.runtime = runtime

    def __call__(self, environ, start_response):
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")
        try:
            if method == "GET" and path == "/healthz":
                return _json_response(start_response, 200, {"status": "ok", "runtime": self.runtime.bundle.name})
            if method == "GET" and path == "/v1/models":
                return _json_response(
                    start_response,
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": self.runtime.bundle.name,
                                "object": "model",
                                "owned_by": "skillcortex",
                            }
                        ],
                    },
                )
            if method == "POST" and path == "/v1/chat/completions":
                body = _read_request_body(environ)
                payload = json.loads(body or "{}")
                return _json_response(start_response, 200, self.runtime.chat_completion(payload))
            return _json_response(start_response, 404, {"error": {"message": "not found"}})
        except ValueError as error:
            return _json_response(start_response, 400, {"error": {"message": str(error)}})
        except Exception as error:  # pragma: no cover - defensive server boundary
            return _json_response(start_response, 500, {"error": {"message": str(error)}})


def load_runtime_bundle(runtime_path: Path) -> RuntimeBundle:
    root = runtime_path.resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"runtime bundle not found: {root}")
    missing = [name for name in REQUIRED_RUNTIME_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"runtime bundle is missing required files: {missing[0]}")

    composition = _read_yaml(root / "composition.yaml")
    router_config = _read_json(root / "router_config.json")
    active_skills = _read_json(root / "active_skills.json")
    compatibility_report = _read_json(root / "compatibility_report.json")
    budget_report = _read_json(root / "budget_report.json")
    checksums = _read_json(root / "checksums.json")

    _validate_bundle_metadata(composition, router_config, active_skills, compatibility_report, checksums)
    _validate_bundle_checksums(root, checksums)
    skills = _load_runtime_skills(composition, checksums)
    _validate_router_routes(router_config.get("routes") or [], skills)
    _validate_active_skills(active_skills, skills, router_config.get("routes") or [])

    runtime = composition.get("runtime") or {}
    return RuntimeBundle(
        path=root,
        name=root.name,
        runtime_model=runtime["runtime_model"],
        source_model=runtime["source_model"],
        quantization=runtime["quantization"],
        strategy=composition["strategy"],
        routes=list(router_config["routes"]),
        skills=skills,
        compatibility_report=compatibility_report,
        budget_report=budget_report,
        checksums=checksums,
    )


def validate_runtime_bundle(runtime_path: Path) -> dict[str, Any]:
    return SkillRuntime.load(runtime_path).validate()


def load_chat_request(path: Path, *, runtime_name: str | None = None) -> dict[str, Any]:
    payload = _read_json(path.resolve())
    return normalize_chat_request(payload, runtime_name=runtime_name)


def serve_runtime(
    *,
    runtime_path: Path,
    host: str,
    port: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    runtime = SkillRuntime.load(runtime_path)
    if dry_run:
        return {
            "status": "dry-run",
            "runtime": runtime.bundle.name,
            "host": host,
            "port": port,
            "model": runtime.bundle.runtime_model,
            "skills": sorted(runtime.bundle.skills),
        }
    app = OpenAICompatApp(runtime)
    with make_server(host, port, app) as server:
        print(f"Serving SkillCortex runtime '{runtime.bundle.name}' on http://{host}:{port}")
        server.serve_forever()
    return {"status": "stopped"}


def _coerce_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    coerced = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError("message role must be one of system, user, assistant")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        coerced.append({"role": role, "content": content})
    return coerced


def normalize_chat_request(payload: dict[str, Any], *, runtime_name: str | None = None) -> dict[str, Any]:
    model_name = payload.get("model")
    if runtime_name and model_name and model_name != runtime_name:
        raise ValueError(f"unknown model: {model_name}")
    if payload.get("stream"):
        raise ValueError("streaming is not supported")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    task_type = payload.get("task_type")
    if task_type is not None and task_type not in TASK_TYPES:
        raise ValueError(f"unknown task_type: {task_type}")
    return {
        "model": model_name,
        "messages": _coerce_messages(messages),
        "task_type": task_type,
        "semantic_family": payload.get("semantic_family"),
        "skill_override": payload.get("skill_override"),
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
    }


def _infer_task_type(messages: list[dict[str, str]]) -> str:
    user_text = "\n".join(message["content"] for message in messages if message["role"] == "user")
    selected = model_backend.route_text(user_text)
    if "debugging_skill" in selected:
        return "debugging"
    if "test_generation_skill" in selected:
        return "test_generation"
    return "python_generation"


def _json_response(start_response, status_code: int, payload: dict[str, Any]):
    body = json.dumps(payload).encode() + b"\n"
    start_response(
        f"{status_code} {_status_text(status_code)}",
        [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def _load_runtime_skills(composition: dict[str, Any], checksums: dict[str, Any]) -> dict[str, RuntimeSkill]:
    skill_entries = composition.get("skills") or []
    fingerprints_by_skill = {
        item["skill_id"]: item["fingerprint"] for item in checksums.get("source_packages") or []
    }
    skills: dict[str, RuntimeSkill] = {}
    for item in skill_entries:
        package_path = Path(item["package_path"]).resolve()
        validate_skill_package(package_path)
        metadata = _read_json(package_path / "metadata.json")
        manifest = _read_yaml(package_path / "skill.yaml")
        fingerprint = _package_fingerprint(manifest, metadata)
        expected = item.get("fingerprint")
        if fingerprint != expected:
            raise ValueError(f"runtime package fingerprint mismatch for {item['skill_id']}")
        if fingerprints_by_skill.get(item["skill_id"]) != fingerprint:
            raise ValueError(f"runtime checksum fingerprint mismatch for {item['skill_id']}")
        adapter_files = (metadata.get("adapter") or {}).get("files") or {}
        adapter_relative = adapter_files.get("weights") or "adapter/adapters.safetensors"
        adapter_path = (package_path / adapter_relative).parent
        skills[item["skill_id"]] = RuntimeSkill(
            skill_id=item["skill_id"],
            name=item["name"],
            version=item["version"],
            package_path=package_path,
            adapter_path=adapter_path,
            fingerprint=fingerprint,
            allowed_task_types=list(item["composition"]["capabilities"]["allowed_task_types"]),
            activation=dict(item["composition"]["activation"]),
            trainable_parameters=int(item["adapter"]["trainable_parameters"]),
        )
    return skills


def _normalize_messages(
    *,
    prompt: str | None,
    system: str | None,
    messages: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if messages is not None:
        return _coerce_messages(messages)
    if prompt is None:
        raise ValueError("prompt or messages is required")
    normalized = []
    if system:
        normalized.append({"role": "system", "content": system})
    normalized.append({"role": "user", "content": prompt})
    return normalized


def _read_request_body(environ) -> str:
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length <= 0:
        return ""
    return environ["wsgi.input"].read(length).decode()


def _select_route(routes: list[dict[str, Any]], task_type: str, semantic_family: str | None) -> dict[str, Any]:
    for route in routes:
        if route["task_type"] == task_type and route.get("semantic_family") == semantic_family:
            return route
    for route in routes:
        if route["task_type"] == task_type and route.get("semantic_family") is None:
            return route
    raise ValueError(f"runtime bundle is missing a default route for {task_type}")


def _status_text(status_code: int) -> str:
    return {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        500: "Internal Server Error",
    }[status_code]


def _validate_active_skills(
    active_skills: dict[str, Any],
    skills: dict[str, RuntimeSkill],
    routes: list[dict[str, Any]],
) -> None:
    route_ids = {route["route_id"] for route in routes}
    active_ids = {item["skill_id"] for item in active_skills.get("skills") or []}
    if active_ids != set(skills):
        raise ValueError("active_skills.json must match composition skills")
    for item in active_skills.get("skills") or []:
        membership = set(item.get("route_membership") or [])
        if not membership.issubset(route_ids):
            raise ValueError(f"unknown route membership for {item['skill_id']}")


def _validate_bundle_checksums(root: Path, checksums: dict[str, Any]) -> None:
    files = checksums.get("files") or {}
    for relative, expected in sorted(files.items()):
        candidate = root / relative
        if not candidate.exists():
            raise ValueError(f"runtime bundle checksum target is missing: {relative}")
        actual = _sha256(candidate)
        if actual != expected:
            raise ValueError(f"runtime bundle checksum mismatch: {relative}")


def _validate_bundle_metadata(
    composition: dict[str, Any],
    router_config: dict[str, Any],
    active_skills: dict[str, Any],
    compatibility_report: dict[str, Any],
    checksums: dict[str, Any],
) -> None:
    if composition.get("schema_version") != "1":
        raise ValueError("composition.yaml schema_version must be '1'")
    if composition.get("composition_type") != "skill_runtime_bundle":
        raise ValueError("composition.yaml composition_type must be 'skill_runtime_bundle'")
    if composition.get("status") != "complete":
        raise ValueError("composition.yaml status must be 'complete'")
    if router_config.get("schema_version") != "1":
        raise ValueError("router_config.json schema_version must be '1'")
    if active_skills.get("schema_version") != "1":
        raise ValueError("active_skills.json schema_version must be '1'")
    if compatibility_report.get("schema_version") != "1":
        raise ValueError("compatibility_report.json schema_version must be '1'")
    if compatibility_report.get("status") != "valid":
        errors = compatibility_report.get("errors") or ["unknown runtime compatibility error"]
        raise ValueError(errors[0])
    if checksums.get("schema_version") != "1":
        raise ValueError("checksums.json schema_version must be '1'")
    if router_config.get("strategy") != composition.get("strategy"):
        raise ValueError("runtime bundle strategy mismatch")
    if list(router_config.get("routes") or []) != list(composition.get("routes") or []):
        raise ValueError("router_config.json routes must match composition.yaml")


def _validate_router_routes(routes: list[dict[str, Any]], skills: dict[str, RuntimeSkill]) -> None:
    if not routes:
        raise ValueError("runtime bundle routes are required")
    default_routes = set()
    for route in routes:
        task_type = route.get("task_type")
        if task_type not in TASK_TYPES:
            raise ValueError(f"unknown runtime task_type: {task_type}")
        selected = route.get("selected_skills") or []
        unknown = [skill_id for skill_id in selected if skill_id not in skills]
        if unknown:
            raise ValueError(f"route references unknown skill: {unknown[0]}")
        if route.get("semantic_family") is None:
            default_routes.add(task_type)
    missing_defaults = [task_type for task_type in TASK_TYPES if task_type not in default_routes]
    if missing_defaults:
        raise ValueError(f"runtime bundle is missing default route: {missing_defaults[0]}")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()