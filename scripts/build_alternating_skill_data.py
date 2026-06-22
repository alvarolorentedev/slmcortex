#!/usr/bin/env python3
"""Build deterministic leakage-safe alternating-skill train and holdout data."""

import json
from pathlib import Path

CASES = (
    ("empty", [], []),
    ("singleton", [9], [9]),
    ("even_length", [3, 8, 5, 2], [3, 5]),
    ("odd_length", [4, 1, 7, 2, 6], [4, 7, 6]),
    ("duplicates", [2, 2, 2, 2, 2], [2, 2, 2]),
    ("value_confusion", [1, 0, 1, 0, 1], [1, 1, 1]),
    ("already_patterned", ["a", "x", "b", "x", "c"], ["a", "b", "c"]),
    ("non_patterned", [10, -4, 7, 3, 0, 8], [10, 7, 0]),
)
MUTANTS = (
    "values[1::2]",
    "values[::1]",
    "values[::2][1:]",
    "[value for value in values if value]",
    "[value for value in values if values.index(value) % 2 == 0]",
)


def _metadata(split, task_type):
    return {
        "semantic_family": "alternating",
        "candidate_skill": "alternating_skill",
        "source": "synthetic_failure_born",
        "split": split,
        "task_type": task_type,
        "evaluation_leakage_safe": True,
    }


def _debug_example(split, index):
    label, values, expected = CASES[index % len(CASES)]
    function = f"select_even_slots_{split}_{index:02d}"
    mutant = MUTANTS[index % len(MUTANTS)]
    target = f"def {function}(items):\n    return items[::2]"
    prompt = (
        f"Repair `{function}` so it keeps elements by zero-based position "
        f"0, 2, 4, and so on. Return code only. Case focus: {label}.\n\n"
        f"def {function}(items):\n    return {mutant.replace('values', 'items')}"
    )
    fixture = (
        f"from solution import {function}\n\n"
        f"def test_{function}():\n"
        f"    assert {function}({values!r}) == {expected!r}\n"
        f"    assert {function}([]) == []\n"
        f"    assert {function}([5]) == [5]\n"
    )
    return {
        "id": f"failure-born-alternating-{split}-debug-{index:02d}",
        "group": "alternating",
        "task_type": "debugging",
        "skills": ["alternating_skill"],
        "prompt": prompt,
        "target": target,
        "execution": {
            "files": {"test_solution.py": fixture},
            "command": ["python", "-m", "pytest", "-q"],
            "timeout_seconds": 10,
        },
        "metadata": _metadata(split, "debugging"),
    }


def _test_example(split, index):
    label, values, expected = CASES[(index + 3) % len(CASES)]
    function = f"take_position_zero_two_{split}_{index:02d}"
    prompt = (
        f"Write pytest tests for `{function}(items)`, which returns elements "
        f"at zero-based positions 0, 2, 4, etc. Import it from solution. "
        f"Return tests only. Include a {label} case."
    )
    target = (
        f"from solution import {function}\n\n"
        f"def test_{function}():\n"
        f"    assert {function}({values!r}) == {expected!r}\n"
        f"    assert {function}([]) == []\n"
        f"    assert {function}([11]) == [11]\n"
    )
    correct = f"def {function}(items):\n    return items[::2]\n"
    mutant = f"def {function}(items):\n    return items[1::2]\n"
    verifier = (
        "import os\nimport pathlib\nimport shutil\nimport subprocess\nimport sys\n\n"
        "root = pathlib.Path(__file__).parent\n"
        "env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}\n"
        "correct = subprocess.run([sys.executable, '-m', 'pytest', '-q', "
        "'test_generated.py'], cwd=root, env=env).returncode\n"
        f"(root / 'solution.py').write_text({mutant!r})\n"
        "shutil.rmtree(root / '__pycache__', ignore_errors=True)\n"
        "mutant = subprocess.run([sys.executable, '-m', 'pytest', '-q', "
        "'test_generated.py'], cwd=root, env=env).returncode\n"
        "raise SystemExit(0 if correct == 0 and mutant != 0 else 1)\n"
    )
    return {
        "id": f"failure-born-alternating-{split}-test-{index:02d}",
        "group": "alternating",
        "task_type": "test_generation",
        "skills": ["alternating_skill"],
        "prompt": prompt,
        "target": target,
        "execution": {
            "files": {"solution.py": correct, "verify_tests.py": verifier},
            "command": ["python", "verify_tests.py"],
            "timeout_seconds": 10,
        },
        "metadata": _metadata(split, "test_generation"),
    }


def build_datasets(root="data/failure_born/alternating_skill"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    datasets = {
        "train": [
            *(_debug_example("train", index) for index in range(25)),
            *(_test_example("train", index) for index in range(15)),
        ],
        "holdout": [
            *(_debug_example("holdout", index) for index in range(15)),
            *(_test_example("holdout", index) for index in range(15)),
        ],
    }
    paths = []
    for split, rows in datasets.items():
        path = root / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)
    return tuple(paths)


if __name__ == "__main__":
    train, holdout = build_datasets()
    print(train)
    print(holdout)
