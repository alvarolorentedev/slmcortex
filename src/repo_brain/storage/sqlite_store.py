from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from repo_brain.models import FileRecord


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    digest TEXT NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    is_test INTEGER NOT NULL,
                    is_config INTEGER NOT NULL,
                    is_generated INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO schema_migrations(version) VALUES (1);
                INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', '1');
                """
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def files(self) -> dict[str, FileRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT path, language, size, digest, modified_ns, is_test, is_config, "
                "is_generated FROM files"
            )
            return {
                row[0]: FileRecord(
                    row[0], row[1], row[2], row[3], row[4], bool(row[5]), bool(row[6]), bool(row[7])
                )
                for row in rows
            }

    def replace_files(self, records: Iterable[FileRecord]) -> None:
        rows = list(records)
        with self.connect() as connection:
            connection.execute("DELETE FROM files")
            connection.executemany(
                "INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        record.path,
                        record.language,
                        record.size,
                        record.digest,
                        record.modified_ns,
                        record.is_test,
                        record.is_config,
                        record.is_generated,
                    )
                    for record in rows
                ],
            )

    def set_metadata(self, key: str, value: object) -> None:
        encoded = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", (key, encoded)
            )

    def metadata(self) -> dict[str, str]:
        with self.connect() as connection:
            return dict(connection.execute("SELECT key, value FROM metadata"))

