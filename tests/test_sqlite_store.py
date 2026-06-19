from pathlib import Path

from repo_brain.models import FileRecord
from repo_brain.storage.sqlite_store import SQLiteStore


def test_store_round_trips_files(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "index.sqlite3")
    record = FileRecord("a.py", "python", 3, "abc", 1, False, False, False)
    store.replace_files([record])
    assert store.files() == {"a.py": record}


def test_store_removes_stale_files(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "index.sqlite3")
    store.replace_files(
        [
            FileRecord("a.py", "python", 3, "a", 1, False, False, False),
            FileRecord("b.py", "python", 3, "b", 1, False, False, False),
        ]
    )
    store.replace_files([FileRecord("a.py", "python", 3, "a", 1, False, False, False)])
    assert set(store.files()) == {"a.py"}

