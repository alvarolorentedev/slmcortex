from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from repo_brain.models import DependencyEdge, FileRecord, SymbolRecord, TestRecord


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
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    qualified_name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    signature TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dependencies (
                    source_file TEXT NOT NULL,
                    target TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    resolved_file TEXT,
                    PRIMARY KEY (source_file, target, kind)
                );
                CREATE TABLE IF NOT EXISTS tests (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    command TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_errors (
                    file_path TEXT NOT NULL,
                    message TEXT NOT NULL
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

    def replace_analysis(
        self,
        symbols: Iterable[SymbolRecord],
        dependencies: Iterable[DependencyEdge],
        tests: Iterable[TestRecord],
        errors: Iterable[tuple[str, str]],
    ) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM symbols")
            connection.execute("DELETE FROM dependencies")
            connection.execute("DELETE FROM tests")
            connection.execute("DELETE FROM analysis_errors")
            connection.executemany(
                "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.id,
                        item.file_path,
                        item.name,
                        item.qualified_name,
                        item.kind,
                        item.line_start,
                        item.line_end,
                        item.signature,
                    )
                    for item in symbols
                ],
            )
            connection.executemany(
                "INSERT INTO dependencies VALUES (?, ?, ?, ?)",
                [
                    (item.source_file, item.target, item.kind, item.resolved_file)
                    for item in dependencies
                ],
            )
            connection.executemany(
                "INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.id,
                        item.file_path,
                        item.name,
                        item.framework,
                        item.line_start,
                        item.command,
                    )
                    for item in tests
                ],
            )
            connection.executemany("INSERT INTO analysis_errors VALUES (?, ?)", list(errors))

    def symbols(self) -> list[SymbolRecord]:
        with self.connect() as connection:
            return [SymbolRecord(*row) for row in connection.execute("SELECT * FROM symbols")]

    def dependencies(self) -> list[DependencyEdge]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM dependencies")
            return [DependencyEdge(*row) for row in rows]

    def tests(self) -> list[TestRecord]:
        with self.connect() as connection:
            return [TestRecord(*row) for row in connection.execute("SELECT * FROM tests")]

    def analysis_errors(self) -> list[dict[str, str]]:
        with self.connect() as connection:
            return [
                {"path": row[0], "message": row[1]}
                for row in connection.execute("SELECT file_path, message FROM analysis_errors")
            ]
