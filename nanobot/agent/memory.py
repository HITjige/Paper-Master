"""Memory system: pure file I/O store, lightweight Consolidator, and Dream processor."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import uuid
import weakref
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from nanobot.agent.context_budget import ContextBudget
from nanobot.agent.runner import AgentRunSpec, AgentRunner
from nanobot.agent.skill_lifecycle import SkillCandidateManager
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.utils.gitstore import GitStore
from nanobot.utils.helpers import (
    ensure_dir,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    strip_think,
)
from nanobot.utils.prompt_templates import render_template

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session, SessionManager


_HISTORY_LOCKS: dict[str, threading.RLock] = {}
_HISTORY_LOCKS_GUARD = threading.Lock()


def _shared_history_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _HISTORY_LOCKS_GUARD:
        return _HISTORY_LOCKS.setdefault(key, threading.RLock())


# ---------------------------------------------------------------------------
# MemoryStore — pure file I/O layer
# ---------------------------------------------------------------------------

class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""

    _DEFAULT_MAX_HISTORY = 1000  # 定义了 history.jsonl 中最多保留 1000 条流水账历史。超过的旧记录会被丢弃。
    # 为了兼容老版本的 HISTORY.md 文件而预先编译的正则。用于匹配诸如 [2023-10-01 12:00] USER: 这样的旧式纯文本日志。
    _LEGACY_ENTRY_START_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*")
    _LEGACY_TIMESTAMP_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]\s*")
    _LEGACY_RAW_MESSAGE_RE = re.compile(
        r"^\[\d{4}-\d{2}-\d{2}[^\]]*\]\s+[A-Z][A-Z0-9_]*(?:\s+\[tools:\s*[^\]]+\])?:"
    )

    def __init__(self, workspace: Path, max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.history_lock_file = self.memory_dir / ".history.lock"
        self.structured_db = self.memory_dir / "memory.sqlite3"
        self.legacy_history_file = self.memory_dir / "HISTORY.md"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        # 游标文件（用于标记处理到哪一条了）
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._history_lock = _shared_history_lock(self.history_file)
        self._structured_lock = _shared_history_lock(self.structured_db)
        self._fts_available = False
        # 把 MEMORY.md 和 USER.md 交给内置的微型 Git 客户端托管，防止这些重要文件被意外改乱。
        self._git = GitStore(workspace, tracked_files=[
            "SOUL.md", "USER.md", "memory/MEMORY.md",
        ])
        self._maybe_migrate_legacy_history()
        self._init_structured_store()

    @property
    def git(self) -> GitStore:
        return self._git

    # -- generic helpers -----------------------------------------------------

    @staticmethod
    def read_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temp_name = output.name
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_name, path)
            try:
                directory_fd = os.open(
                    path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # File fsync + atomic replace is still the best available
                # fallback on platforms that cannot fsync directories.
                pass
        finally:
            if temp_name:
                Path(temp_name).unlink(missing_ok=True)

    @contextmanager
    def _history_write_guard(self):
        """Serialize history/cursor mutations in-process and across POSIX workers."""
        with self._history_lock:
            lock_handle = self.history_lock_file.open("a+", encoding="utf-8")
            try:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
                lock_handle.close()

    def _connect_structured(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.structured_db, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _init_structured_store(self) -> None:
        with self._structured_lock, self._connect_structured() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    source_cursor INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    valid_from TEXT,
                    expires_at TEXT,
                    supersedes TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    content_hash TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_identity
                ON memory_records(scope, kind, content_hash)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_scope_status
                ON memory_records(scope, status)
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        id UNINDEXED,
                        subject,
                        content,
                        tokenize='unicode61'
                    )
                    """
                )
                self._fts_available = True
            except sqlite3.OperationalError:
                logger.warning("SQLite FTS5 unavailable; structured memory uses lexical fallback")

    def _maybe_migrate_legacy_history(self) -> None:
        """One-time upgrade from legacy HISTORY.md to history.jsonl.

        The migration is best-effort and prioritizes preserving as much content
        as possible over perfect parsing.
        """
        if not self.legacy_history_file.exists():
            return
        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            return

        try:
            legacy_text = self.legacy_history_file.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            logger.exception("Failed to read legacy HISTORY.md for migration")
            return

        entries = self._parse_legacy_history(legacy_text)
        try:
            if entries:
                self._write_entries(entries)
                last_cursor = entries[-1]["cursor"]
                self._atomic_write_text(self._cursor_file, str(last_cursor))
                # Default to "already processed" so upgrades do not replay the
                # user's entire historical archive into Dream on first start.
                self._atomic_write_text(self._dream_cursor_file, str(last_cursor))

            backup_path = self._next_legacy_backup_path()
            self.legacy_history_file.replace(backup_path)
            logger.info(
                "Migrated legacy HISTORY.md to history.jsonl ({} entries)",
                len(entries),
            )
        except Exception:
            logger.exception("Failed to migrate legacy HISTORY.md")

    def _parse_legacy_history(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        fallback_timestamp = self._legacy_fallback_timestamp()
        entries: list[dict[str, Any]] = []
        chunks = self._split_legacy_history_chunks(normalized)

        for cursor, chunk in enumerate(chunks, start=1):
            timestamp = fallback_timestamp
            content = chunk
            match = self._LEGACY_TIMESTAMP_RE.match(chunk)
            if match:
                timestamp = match.group(1)
                remainder = chunk[match.end():].lstrip()
                if remainder:
                    content = remainder

            entries.append({
                "cursor": cursor,
                "timestamp": timestamp,
                "content": content,
            })
        return entries

    def _split_legacy_history_chunks(self, text: str) -> list[str]:
        lines = text.split("\n")
        chunks: list[str] = []
        current: list[str] = []
        saw_blank_separator = False

        for line in lines:
            if saw_blank_separator and line.strip() and current:
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            if self._should_start_new_legacy_chunk(line, current):
                chunks.append("\n".join(current).strip())
                current = [line]
                saw_blank_separator = False
                continue
            current.append(line)
            saw_blank_separator = not line.strip()

        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _should_start_new_legacy_chunk(self, line: str, current: list[str]) -> bool:
        if not current:
            return False
        if not self._LEGACY_ENTRY_START_RE.match(line):
            return False
        if self._is_raw_legacy_chunk(current) and self._LEGACY_RAW_MESSAGE_RE.match(line):
            return False
        return True

    def _is_raw_legacy_chunk(self, lines: list[str]) -> bool:
        first_nonempty = next((line for line in lines if line.strip()), "")
        match = self._LEGACY_TIMESTAMP_RE.match(first_nonempty)
        if not match:
            return False
        return first_nonempty[match.end():].lstrip().startswith("[RAW]")

    def _legacy_fallback_timestamp(self) -> str:
        try:
            return datetime.fromtimestamp(
                self.legacy_history_file.stat().st_mtime,
            ).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _next_legacy_backup_path(self) -> Path:
        candidate = self.memory_dir / "HISTORY.md.bak"
        suffix = 2
        while candidate.exists():
            candidate = self.memory_dir / f"HISTORY.md.bak.{suffix}"
            suffix += 1
        return candidate

    # -- MEMORY.md (long-term facts) -----------------------------------------

    def read_memory(self) -> str:
        return self.read_file(self.memory_file)

    def write_memory(self, content: str) -> None:
        self._atomic_write_text(self.memory_file, content)

    # -- SOUL.md -------------------------------------------------------------

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)

    def write_soul(self, content: str) -> None:
        self._atomic_write_text(self.soul_file, content)

    # -- USER.md -------------------------------------------------------------

    def read_user(self) -> str:
        return self.read_file(self.user_file)

    def write_user(self, content: str) -> None:
        self._atomic_write_text(self.user_file, content)

    # -- context injection (used by context.py) ------------------------------

    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # -- structured long-term memory ---------------------------------------

    @staticmethod
    def _memory_tokens(text: str) -> set[str]:
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9][a-z0-9_.:/-]{1,}", lowered))
        for run in re.findall(r"[\u3400-\u9fff]+", lowered):
            if len(run) == 1:
                tokens.add(run)
            else:
                tokens.update(run[index:index + 2] for index in range(len(run) - 1))
        return tokens

    def upsert_memory_record(
        self,
        *,
        scope: str,
        kind: str,
        content: str,
        subject: str = "",
        confidence: float = 0.7,
        evidence: list[dict[str, Any]] | None = None,
        source_cursor: int | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        supersedes: str | None = None,
        status: str = "active",
    ) -> str:
        """Idempotently store a scoped memory record with provenance."""
        with self._structured_lock, self._connect_structured() as connection:
            return self._upsert_memory_record(
                connection,
                scope=scope,
                kind=kind,
                content=content,
                subject=subject,
                confidence=confidence,
                evidence=evidence,
                source_cursor=source_cursor,
                valid_from=valid_from,
                expires_at=expires_at,
                supersedes=supersedes,
                status=status,
            )

    def _upsert_memory_record(
        self,
        connection: sqlite3.Connection,
        *,
        scope: str,
        kind: str,
        content: str,
        subject: str = "",
        confidence: float = 0.7,
        evidence: list[dict[str, Any]] | None = None,
        source_cursor: int | None = None,
        valid_from: str | None = None,
        expires_at: str | None = None,
        supersedes: str | None = None,
        status: str = "active",
    ) -> str:
        """Upsert one record using the caller's transaction."""
        normalized = re.sub(r"\s+", " ", content).strip()
        if not normalized:
            raise ValueError("Structured memory content cannot be empty")
        normalized_scope = scope.strip() or "workspace"
        normalized_kind = kind.strip().lower() or "fact"
        content_hash = hashlib.sha256(normalized.casefold().encode("utf-8")).hexdigest()
        now = datetime.now().astimezone().isoformat()
        record_id = f"mem_{uuid.uuid4().hex}"
        evidence_json = json.dumps(evidence or [], ensure_ascii=False)
        bounded_confidence = min(1.0, max(0.0, float(confidence)))
        existing = connection.execute(
            """
            SELECT id, created_at FROM memory_records
            WHERE scope = ? AND kind = ? AND content_hash = ?
            """,
            (normalized_scope, normalized_kind, content_hash),
        ).fetchone()
        if existing is not None:
            record_id = str(existing["id"])
            created_at = str(existing["created_at"])
        else:
            created_at = now
        normalized_subject = subject.strip()
        if normalized_subject and not supersedes:
            previous_subject = connection.execute(
                """
                SELECT id FROM memory_records
                WHERE scope = ? AND kind = ? AND subject = ?
                  AND status = 'active' AND content_hash != ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (
                    normalized_scope,
                    normalized_kind,
                    normalized_subject,
                    content_hash,
                ),
            ).fetchone()
            if previous_subject is not None:
                supersedes = str(previous_subject["id"])
        connection.execute(
            """
            INSERT INTO memory_records(
                id, scope, kind, subject, content, confidence,
                evidence_json, source_cursor, created_at, updated_at,
                valid_from, expires_at, supersedes, status, content_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, kind, content_hash) DO UPDATE SET
                subject = excluded.subject,
                confidence = MAX(memory_records.confidence, excluded.confidence),
                evidence_json = excluded.evidence_json,
                source_cursor = COALESCE(excluded.source_cursor, memory_records.source_cursor),
                updated_at = excluded.updated_at,
                valid_from = COALESCE(excluded.valid_from, memory_records.valid_from),
                expires_at = excluded.expires_at,
                supersedes = COALESCE(excluded.supersedes, memory_records.supersedes),
                status = excluded.status
            """,
            (
                record_id,
                normalized_scope,
                normalized_kind,
                normalized_subject,
                normalized,
                bounded_confidence,
                evidence_json,
                source_cursor,
                created_at,
                now,
                valid_from,
                expires_at,
                supersedes,
                status,
                content_hash,
            ),
        )
        if self._fts_available:
            connection.execute("DELETE FROM memory_fts WHERE id = ?", (record_id,))
            connection.execute(
                "INSERT INTO memory_fts(id, subject, content) VALUES (?, ?, ?)",
                (record_id, normalized_subject, normalized),
            )
        if supersedes:
            connection.execute(
                "UPDATE memory_records SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, supersedes),
            )
        if normalized_subject:
            connection.execute(
                """
                UPDATE memory_records
                SET status = 'superseded', updated_at = ?
                WHERE scope = ? AND kind = ? AND subject = ?
                  AND id != ? AND status = 'active'
                """,
                (
                    now,
                    normalized_scope,
                    normalized_kind,
                    normalized_subject,
                    record_id,
                ),
            )
        return record_id

    def apply_memory_proposals(
        self,
        proposals: list[dict[str, Any]],
        *,
        scope: str,
        evidence: list[dict[str, Any]] | None = None,
        source_cursor: int | None = None,
    ) -> list[str]:
        """Atomically apply one Dream batch, including removals and corrections."""
        changed_ids: list[str] = []
        normalized_scope = scope.strip() or "workspace"
        with self._structured_lock, self._connect_structured() as connection:
            for proposal in proposals:
                action = str(proposal.get("action", "upsert")).strip().lower()
                kind = str(proposal.get("kind", "fact")).strip().lower() or "fact"
                if action == "remove":
                    clauses = ["scope = ?", "status = 'active'"]
                    parameters: list[Any] = [normalized_scope]
                    subject = str(proposal.get("subject", "")).strip()
                    old_content = re.sub(
                        r"\s+",
                        " ",
                        str(proposal.get("old_content", "") or proposal.get("content", "")),
                    ).strip()
                    if subject:
                        clauses.extend(["kind = ?", "subject = ?"])
                        parameters.extend([kind, subject])
                    elif old_content:
                        content_hash = hashlib.sha256(
                            old_content.casefold().encode("utf-8")
                        ).hexdigest()
                        clauses.append("content_hash = ?")
                        parameters.append(content_hash)
                    else:
                        continue
                    matched = connection.execute(
                        f"SELECT id FROM memory_records WHERE {' AND '.join(clauses)}",
                        parameters,
                    ).fetchall()
                    matched_ids = [str(row["id"]) for row in matched]
                    if not matched_ids:
                        continue
                    now = datetime.now().astimezone().isoformat()
                    placeholders = ",".join("?" for _ in matched_ids)
                    connection.execute(
                        f"UPDATE memory_records SET status = 'invalidated', updated_at = ? "
                        f"WHERE id IN ({placeholders})",
                        [now, *matched_ids],
                    )
                    if self._fts_available:
                        connection.execute(
                            f"DELETE FROM memory_fts WHERE id IN ({placeholders})",
                            matched_ids,
                        )
                    changed_ids.extend(matched_ids)
                    continue
                if action != "upsert":
                    continue
                changed_ids.append(self._upsert_memory_record(
                    connection,
                    scope=normalized_scope,
                    kind=kind,
                    subject=str(proposal.get("subject", "")),
                    content=str(proposal.get("content", "")),
                    confidence=float(proposal.get("confidence", 0.7)),
                    evidence=evidence,
                    source_cursor=source_cursor,
                    valid_from=proposal.get("valid_from"),
                    expires_at=proposal.get("expires_at"),
                ))
        return changed_ids

    def search_memory_records(
        self,
        query: str,
        *,
        scopes: set[str] | None = None,
        top_k: int = 8,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Retrieve scoped memories using FTS/BM25, overlap, confidence and recency."""
        if top_k <= 0:
            return []
        if scopes is not None and not scopes:
            return []
        now = datetime.now().astimezone()
        scope_values = sorted(scopes) if scopes is not None else []
        parameters: list[Any] = []
        scope_clause = ""
        if scope_values:
            placeholders = ",".join("?" for _ in scope_values)
            scope_clause = f"AND scope IN ({placeholders})"
            parameters.extend(scope_values)
        parameters.append(float(min_confidence))
        sql = f"""
            SELECT * FROM memory_records
            WHERE status = 'active'
              {scope_clause}
              AND confidence >= ?
        """
        with self._structured_lock, self._connect_structured() as connection:
            rows = [dict(row) for row in connection.execute(sql, parameters).fetchall()]
            fts_scores: dict[str, float] = {}
            if self._fts_available and query.strip():
                terms = re.findall(r"[\w\u3400-\u9fff]+", query.lower())[:12]
                expression = " OR ".join(
                    f'"{term.replace(chr(34), chr(34) * 2)}"'
                    for term in terms
                    if term
                )
                if expression:
                    try:
                        matches = connection.execute(
                            "SELECT id, bm25(memory_fts) AS rank FROM memory_fts WHERE memory_fts MATCH ?",
                            (expression,),
                        ).fetchall()
                        raw_scores = {
                            str(row["id"]): max(0.0, -float(row["rank"]))
                            for row in matches
                        }
                        max_raw_score = max(raw_scores.values(), default=0.0)
                        if max_raw_score > 0:
                            fts_scores = {
                                record_id: score / max_raw_score
                                for record_id, score in raw_scores.items()
                            }
                    except sqlite3.OperationalError:
                        pass

        query_tokens = self._memory_tokens(query)
        ranked: list[dict[str, Any]] = []
        for row in rows:
            valid_from = row.get("valid_from")
            if valid_from:
                try:
                    if datetime.fromisoformat(str(valid_from)) > now:
                        continue
                except (TypeError, ValueError):
                    pass
            expires_at = row.get("expires_at")
            if expires_at:
                try:
                    if datetime.fromisoformat(str(expires_at)) <= now:
                        continue
                except (TypeError, ValueError):
                    pass
            row_tokens = self._memory_tokens(
                f"{row.get('subject', '')} {row.get('content', '')}"
            )
            overlap = (
                len(query_tokens & row_tokens) / max(1, len(query_tokens))
                if query_tokens
                else 0.0
            )
            try:
                updated = datetime.fromisoformat(str(row.get("updated_at")))
                age_days = max(0.0, (now - updated).total_seconds() / 86400.0)
                recency = 1.0 / (1.0 + age_days / 30.0)
            except (TypeError, ValueError):
                recency = 0.5
            kind_boost = 1.0 if row.get("kind") in {"preference", "constraint"} else 0.0
            score = (
                0.45 * fts_scores.get(str(row["id"]), 0.0)
                + 0.30 * overlap
                + 0.15 * float(row.get("confidence", 0.0))
                + 0.07 * recency
                + 0.03 * kind_boost
            )
            if query_tokens and not (
                fts_scores.get(str(row["id"]), 0.0) > 0.0 or overlap > 0.0
            ):
                continue
            row["score"] = round(score, 6)
            try:
                row["evidence"] = json.loads(row.pop("evidence_json", "[]"))
            except json.JSONDecodeError:
                row["evidence"] = []
            ranked.append(row)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def render_memory_records(records: list[dict[str, Any]]) -> str:
        lines = []
        for record in records:
            kind = str(record.get("kind", "fact")).upper()
            subject = str(record.get("subject", "")).strip()
            prefix = f"{subject}: " if subject else ""
            lines.append(
                f"- [{kind}] {prefix}{record.get('content', '')} "
                f"(confidence={float(record.get('confidence', 0.0)):.2f})"
            )
        return "\n".join(lines)

    # -- history.jsonl — append-only, JSONL format ---------------------------

    def append_history(
        self,
        entry: str,
        *,
        session_key: str | None = None,
        kind: str = "conversation_summary",
        evidence: dict[str, Any] | None = None,
    ) -> int:
        """Append *entry* to history.jsonl and return its auto-incrementing cursor."""
        with self._history_write_guard():
            cursor = self._next_cursor()
            record = {
                "event_id": f"hist_{uuid.uuid4().hex}",
                "cursor": cursor,
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                "timezone": datetime.now().astimezone().strftime("%z"),
                "content": strip_think(entry.rstrip()) or entry.rstrip(),
                "kind": kind,
                "scope": session_key or "workspace",
                "evidence": evidence or {},
            }
            with open(self.history_file, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            self._atomic_write_text(self._cursor_file, str(cursor))
            return cursor

    def _next_cursor(self) -> int:
        """Read the current cursor counter and return next value."""
        if self._cursor_file.exists():
            try:
                return int(self._cursor_file.read_text(encoding="utf-8").strip()) + 1
            except (ValueError, OSError):
                pass
        # Fallback: read last line's cursor from the JSONL file.
        last = self._read_last_entry()
        if last and last.get("cursor"):
            return last["cursor"] + 1
        return 1

    def read_unprocessed_history(
        self,
        since_cursor: int,
        *,
        scopes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return history entries with cursor > *since_cursor*."""
        return [
            entry
            for entry in self._read_entries()
            if entry.get("cursor", 0) > since_cursor
            and (scopes is None or str(entry.get("scope", "workspace")) in scopes)
        ]

    def compact_history(self, processed_cursor: int | None = None) -> None:
        """Bound processed history without ever deleting unprocessed entries."""
        if self.max_history_entries <= 0:
            return
        with self._history_write_guard():
            entries = self._read_entries()
            cursor = (
                self.get_last_dream_cursor()
                if processed_cursor is None
                else max(0, int(processed_cursor))
            )
            processed = [entry for entry in entries if entry.get("cursor", 0) <= cursor]
            unprocessed = [entry for entry in entries if entry.get("cursor", 0) > cursor]
            kept = [*processed[-self.max_history_entries:], *unprocessed]
            if kept != entries:
                self._write_entries(kept)

    # -- JSONL helpers -------------------------------------------------------

    def _read_entries(self) -> list[dict[str, Any]]:
        """Read all entries from history.jsonl."""
        entries: list[dict[str, Any]] = []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def _read_last_entry(self) -> dict[str, Any] | None:
        """Read the last entry from the JSONL file efficiently.

        如果文件行数超过最大设定值（1000条），扔掉最旧的记录"""
        try:
            with open(self.history_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        """Overwrite history.jsonl with the given entries."""
        content = "".join(
            json.dumps(entry, ensure_ascii=False) + "\n"
            for entry in entries
        )
        self._atomic_write_text(self.history_file, content)

    # -- dream cursor --------------------------------------------------------

    def get_last_dream_cursor(self) -> int:
        if self._dream_cursor_file.exists():
            try:
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0

    def set_last_dream_cursor(self, cursor: int) -> None:
        candidate = max(0, int(cursor))
        with self._history_write_guard():
            current = self.get_last_dream_cursor()
            if candidate > current:
                self._atomic_write_text(self._dream_cursor_file, str(candidate))

    # -- message formatting utility ------------------------------------------

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        """格式化消息列表为纯文本，供 LLM 总结用。每条消息一行，格式如下：\n
        [2023-10-01 12:00] USER: Hello\n
        [2023-10-01 12:01] ASSISTANT [tools: web_search]: I found this information...
        """
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    def raw_archive(
        self,
        messages: list[dict],
        *,
        session_key: str | None = None,
    ) -> str:
        """Persist raw evidence as an artifact and return a bounded checkpoint."""
        artifact_dir = ensure_dir(self.memory_dir / "raw_history")
        artifact_id = uuid.uuid4().hex
        artifact_path = artifact_dir / f"{artifact_id}.json"
        payload = json.dumps(messages, ensure_ascii=False, indent=2)
        self._atomic_write_text(artifact_path, payload)
        latest_user = next((
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "user" and message.get("content")
        ), "")
        latest_assistant = next((
            str(message.get("content", ""))
            for message in reversed(messages)
            if message.get("role") == "assistant" and message.get("content")
        ), "")
        checkpoint = (
            "[DEGRADED CHECKPOINT]\n"
            f"- Raw evidence: memory/raw_history/{artifact_path.name}\n"
            f"- Messages: {len(messages)}\n"
            f"- Latest user request: {latest_user[:1200]}\n"
            f"- Latest assistant state: {latest_assistant[:1200]}"
        )
        self.append_history(
            checkpoint,
            session_key=session_key,
            kind="degraded_checkpoint",
            evidence={"artifact": str(artifact_path.relative_to(self.workspace))},
        )
        logger.warning(
            "Memory consolidation degraded: raw-archived {} messages to {}",
            len(messages),
            artifact_path,
        )
        return checkpoint



# ---------------------------------------------------------------------------
# Consolidator — lightweight token-budget triggered consolidation
# ---------------------------------------------------------------------------


class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""

    # 最大连续压缩的轮数。如果一次压不完，最多连压5次，防止陷入死循环。
    _MAX_CONSOLIDATION_ROUNDS = 5
    # 每一次“打包压缩”时，最多提取 60 条消息丢给模型去总结。如果是极度废话连篇的消息，不设置上限会被撑爆
    _MAX_CHUNK_MESSAGES = 60  # hard cap per consolidation round

    # 非常关键的安全冗余量：预授权 1024 Token 空闲给“测算误差”。由于本地的分词器和线上闭源模型的的分词算法不可能完全一致，要留出余地防报错
    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., list[dict[str, Any]]],
        get_tool_definitions: Callable[[], list[dict[str, Any]]],
        max_completion_tokens: int = 4096,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._get_tool_definitions = get_tool_definitions
        # 并发锁 (弱引用字典)。为了防止同一会话 (session_key) 正在压条目的同时，又来了一条新消息触发又一次并发压缩导致乱套
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session.
        
        为指定的那个 Session 获取专属压缩锁。"""
        return self._locks.setdefault(session_key, asyncio.Lock())

    def pick_consolidation_boundary(
        self,
        session: Session,
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        # 上一次压缩的位置
        start = session.last_consolidated
        # 如果起点已经等于全部消息数（全被压过了），或者根本不需要移除Token，直接返回
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx in range(start, len(session.messages)):
            message = session.messages[idx]
            # 注意这个判断：必须要求当前这条是被 `user` (用户) 发送的消息。
            # 这是因为把工具输出或半个思考切给大模型很不讲理。在User说话前切一刀，是一轮对话的天然起点。
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(message)

        return last_boundary

    def _cap_consolidation_boundary(
        self,
        session: Session,
        end_idx: int,
    ) -> int | None:
        """Clamp the chunk size without breaking the user-turn boundary.
        
        保证一次性切割出让大模型总结的条目不超过 _MAX_CHUNK_MESSAGES 条，否则大模型总结会 OOM。"""
        start = session.last_consolidated
        # 如果切出来的范围 <= 60 条，很安全，返回
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx

        # 否则强行把终点收缩到起点 + 60
        capped_end = start + self._MAX_CHUNK_MESSAGES
        # 但是，强行收缩也要从这个60的死线往前倒退，一定要退到一个 "role" == "user" 的安全边界上再切
        for idx in range(capped_end, start, -1):
            if session.messages[idx].get("role") == "user":
                return idx
        return None

    def estimate_session_prompt_tokens(
        self,
        session: Session,
        *,
        session_summary: str | None = None,
    ) -> tuple[int, str]:
        """Estimate current prompt size for the normal session history view.
        
        利用分词器或者 API 测试当前会话到底花了多少 Token"""
        history = session.get_history(max_messages=0)
        channel, chat_id = (session.key.split(":", 1) if ":" in session.key else (None, None))
        probe_messages = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
            session_summary=session_summary,
        )
        # 抛给大模型底层的 Tokenizer 计算最终开销
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._get_tool_definitions(),
        )

    @staticmethod
    def _normalize_working_checkpoint(raw: str) -> str:
        """Render structured summaries while accepting legacy plain text."""
        text = strip_think(raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text or "(nothing)"
        if not isinstance(payload, dict):
            return text or "(nothing)"
        sections = ["# Working Checkpoint"]
        summary = str(payload.get("summary", "")).strip()
        if summary:
            sections.append(f"## Summary\n{summary[:3000]}")
        field_titles = {
            "active_goals": "Active Goals",
            "constraints": "Constraints",
            "decisions": "Decisions",
            "completed": "Completed",
            "open_items": "Open Items",
            "artifacts": "Artifacts",
            "memory_candidates": "Memory Candidates",
        }
        for field, title in field_titles.items():
            values = payload.get(field, [])
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            normalized = [str(value).strip()[:1000] for value in values if str(value).strip()]
            if normalized:
                sections.append(f"## {title}\n" + "\n".join(f"- {value}" for value in normalized[:20]))
        return "\n\n".join(sections) if len(sections) > 1 else "(nothing)"

    async def archive(
        self,
        messages: list[dict],
        *,
        session_key: str | None = None,
    ) -> str | None:
        """Summarize messages via LLM and append to history.jsonl.

        Returns the summary text on success, None if nothing to archive.

        调用大模型把历史消息总结成一段话，然后追加到 history.jsonl 里，作为长期记忆的一部分
        """
        if not messages:
            return None
        try:
            formatted = MemoryStore._format_messages(messages)
            # 使用针对性的 system prompt ("agent/consolidator_archive.md" 模板)，让大模型提炼核心信息
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/consolidator_archive.md",
                            strip=True,
                        ),
                    },
                    {"role": "user", "content": formatted},
                ],
                tools=None,
                tool_choice=None,  # 不允许它在总结记忆时又跑去调工具
            )
            if response.finish_reason == "error":
                raise RuntimeError(f"LLM returned error: {response.content}")
            summary = self._normalize_working_checkpoint(response.content or "")
            self.store.append_history(
                summary,
                session_key=session_key,
                kind="working_checkpoint",
                evidence={"message_count": len(messages)},
            )
            return summary
        except Exception:
            logger.warning("Consolidation LLM call failed, persisting bounded checkpoint")
            return self.store.raw_archive(messages, session_key=session_key)

    async def maybe_consolidate_by_tokens(
        self,
        session: Session,
        *,
        session_summary: str | None = None,
    ) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        if not session.messages or self.context_window_tokens <= 0:
            return

        # 锁住当前会话，防止脏写
        lock = self.get_lock(session.key)
        async with lock:
            # 算出我们真正的 "剩余 Token 预算"
            # 模型能吃的极限 - 模型生成回答所需的预留 - 我们的估算安全垫
            budget = ContextBudget(
                context_window_tokens=self.context_window_tokens,
                output_reserve_tokens=self.max_completion_tokens,
                safety_buffer_tokens=self._SAFETY_BUFFER,
            ).prompt_tokens
            target = budget // 2
            try:
                estimated, source = self.estimate_session_prompt_tokens(
                    session,
                    session_summary=session_summary,
                )
            except Exception:
                logger.exception("Token estimation failed for {}", session.key)
                estimated, source = 0, "error"
            if estimated <= 0:
                return
            if estimated < budget:
                unconsolidated_count = len(session.messages) - session.last_consolidated
                logger.debug(
                    "Token consolidation idle {}: {}/{} via {}, msgs={}",
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    unconsolidated_count,
                )
                return

            round_summaries: list[str] = []
            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    break

                # 找到一个切点，切掉足够的消息让模型总结后能回到安全范围。原则上这个切点必须是一个用户消息的边界
                boundary = self.pick_consolidation_boundary(session, max(1, estimated - target))
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                end_idx = boundary[0]
                # 但是如果这个切点距离上次压缩的位置太远了（超过 _MAX_CHUNK_MESSAGES 条），就收缩到一个更近的安全边界，防止一次性丢给模型太多消息压不动
                end_idx = self._cap_consolidation_boundary(session, end_idx)
                if end_idx is None:
                    logger.debug(
                        "Token consolidation: no capped boundary for {} (round {})",
                        session.key,
                        round_num,
                    )
                    break

                chunk = session.messages[session.last_consolidated:end_idx]
                if not chunk:
                    break

                logger.info(
                    "Token consolidation round {} for {}: {}/{} via {}, chunk={} msgs",
                    round_num,
                    session.key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                summary = await self.archive(chunk, session_key=session.key)
                if summary:
                    round_summaries.append(summary)
                else:
                    break
                session.last_consolidated = end_idx
                self.sessions.save(session)

                try:
                    estimated, source = self.estimate_session_prompt_tokens(
                        session,
                        session_summary=session_summary,
                    )
                except Exception:
                    logger.exception("Token estimation failed for {}", session.key)
                    estimated, source = 0, "error"
                if estimated <= 0:
                    break

            # Persist the last summary to session metadata so it can be injected
            # into the runtime context on the next prepare_session() call, aligning
            # the summary injection strategy with AutoCompact._archive().
            usable_summaries = [
                summary for summary in round_summaries if summary != "(nothing)"
            ]
            if usable_summaries:
                selected: list[str] = []
                selected_chars = 0
                for summary in reversed(usable_summaries):
                    if selected and selected_chars + len(summary) > 16_000:
                        break
                    selected.append(summary)
                    selected_chars += len(summary)
                checkpoint = "\n\n---\n\n".join(reversed(selected))
                session.metadata["_last_summary"] = {
                    "text": checkpoint,
                    "last_active": session.updated_at.isoformat(),
                }
                self.sessions.save(session)


# ---------------------------------------------------------------------------
# Dream — heavyweight cron-scheduled memory consolidation
# ---------------------------------------------------------------------------


# Single source of truth for the staleness threshold used in _annotate_with_ages
# *and* in the Phase 1 prompt template (passed as `stale_threshold_days`).
# Keep code and prompt aligned — if you bump this, the LLM's instruction string
# updates automatically.
_STALE_THRESHOLD_DAYS = 14
_DREAM_PHASE1_MAX_TOKENS = 4096
_DREAM_PHASE2_ATTEMPTS = 2
_DREAM_PHASE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["upsert", "remove"]},
                    "target": {"type": "string", "enum": ["USER", "SOUL", "MEMORY"]},
                    "kind": {"type": "string"},
                    "subject": {"type": "string"},
                    "content": {"type": "string"},
                    "old_content": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "valid_from": {"type": ["string", "null"]},
                    "expires_at": {"type": ["string", "null"]},
                    "reason": {"type": "string"},
                },
                "required": [
                    "action",
                    "target",
                    "kind",
                    "subject",
                    "content",
                    "old_content",
                    "confidence",
                    "valid_from",
                    "expires_at",
                    "reason",
                ],
                "additionalProperties": False,
            },
        },
        "skills": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "when_to_use": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "completion_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "failure_recovery": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "action",
                    "name",
                    "description",
                    "when_to_use",
                    "steps",
                    "completion_criteria",
                    "failure_recovery",
                    "examples",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposals", "skills"],
    "additionalProperties": False,
}


class Dream:
    """Two-phase memory processor with reviewable Skill discovery.

    Phase 1 produces a schema-constrained proposal document.
    Phase 2 delegates to AgentRunner with read_file / edit_file tools so the
    LLM can make targeted, incremental edits instead of replacing entire files.
    Skill proposals are rendered and staged separately by deterministic
    lifecycle code; Dream never writes active Skill files directly.
    
    将用户的历史对话流水账转化为结构化的长期知识、技能或用户画像，整个过程分为两个阶段：分析（Phase 1）与编辑落盘（Phase 2）
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        annotate_line_ages: bool = True,
        skill_candidates: SkillCandidateManager | None = None,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        # Kill switch for the git-blame-based per-line age annotation in Phase 1.
        # Default True keeps the #3212 behavior; set False to feed MEMORY.md raw
        # (e.g. if a specific LLM reacts poorly to the `← Nd` suffix).
        self.annotate_line_ages = annotate_line_ages
        self.skill_candidates = skill_candidates
        self._runner = AgentRunner(provider)
        self._run_lock = asyncio.Lock()
        self._tools = self._build_tools()

    # -- tool registry -------------------------------------------------------

    def _build_tools(self) -> ToolRegistry:
        """Build tools restricted to the three human-readable memory views."""
        from nanobot.agent.skills import BUILTIN_SKILLS_DIR
        from nanobot.agent.tools.filesystem import EditFileTool, ReadFileTool

        tools = ToolRegistry()
        workspace = self.store.workspace
        # Allow reading builtin skills for reference during skill creation
        extra_read = [BUILTIN_SKILLS_DIR] if BUILTIN_SKILLS_DIR.exists() else None
        tools.register(ReadFileTool(
            workspace=workspace,
            allowed_dir=workspace,
            extra_allowed_dirs=extra_read,
        ))
        allowed_memory_files = {
            (workspace / "SOUL.md").resolve(),
            (workspace / "USER.md").resolve(),
            (workspace / "memory" / "MEMORY.md").resolve(),
        }

        class _MemoryOnlyEditFileTool(EditFileTool):
            async def execute(self, path: str | None = None, **kwargs: Any) -> str:
                if not path:
                    return "Error: Unknown path"
                requested_path = path
                normalized_path = path.strip().replace("\\", "/")
                # Some models shorten the documented memory/MEMORY.md path to
                # MEMORY.md. It is unambiguous in Dream's three-file sandbox,
                # so normalize only this exact legacy alias.
                if normalized_path in {"MEMORY.md", "./MEMORY.md"}:
                    path = "memory/MEMORY.md"
                else:
                    candidate = Path(normalized_path).expanduser()
                    if (
                        candidate.is_absolute()
                        and candidate.resolve() == (workspace / "MEMORY.md").resolve()
                    ):
                        path = "memory/MEMORY.md"
                try:
                    resolved = self._resolve(path)
                except PermissionError as exc:
                    return f"Error: {exc}"
                if resolved not in allowed_memory_files:
                    return (
                        f"Error: Dream edit path {requested_path!r} is not allowed; "
                        "use exactly SOUL.md, USER.md, or memory/MEMORY.md"
                    )
                return await super().execute(path=path, **kwargs)

        tools.register(_MemoryOnlyEditFileTool(workspace=workspace, allowed_dir=workspace))
        return tools

    # -- skill listing --------------------------------------------------------

    def _list_existing_skills(self) -> list[str]:
        """List existing skills as 'name — description' for dedup context."""
        import re as _re

        from nanobot.agent.skills import BUILTIN_SKILLS_DIR

        _DESC_RE = _re.compile(r"^description:\s*(.+)$", _re.MULTILINE | _re.IGNORECASE)
        entries: dict[str, str] = {}
        for base in (self.store.workspace / "skills", BUILTIN_SKILLS_DIR):
            if not base.exists():
                continue
            for d in base.iterdir():
                if not d.is_dir():
                    continue
                skill_md = d / "SKILL.md"
                if not skill_md.exists():
                    continue
                # Prefer workspace skills over builtin (same name)
                if d.name in entries and base == BUILTIN_SKILLS_DIR:
                    continue
                content = skill_md.read_text(encoding="utf-8")[:500]
                m = _DESC_RE.search(content)
                desc = m.group(1).strip() if m else "(no description)"
                entries[d.name] = desc
        return [f"{name} — {desc}" for name, desc in sorted(entries.items())]

    # -- main entry ----------------------------------------------------------

    def _annotate_with_ages(self, content: str) -> str:
        """Append per-line age suffixes to MEMORY.md content.

        Each non-blank line whose age exceeds ``_STALE_THRESHOLD_DAYS`` gets a
        suffix like ``← 30d`` indicating days since last modification.
        Returns the original content unchanged if git is unavailable,
        annotate fails, or the line count doesn't match the age count
        (which can happen with an uncommitted working-tree edit — better to
        skip annotation than to tag the wrong line).
        SOUL.md and USER.md are never annotated.
        """
        file_path = "memory/MEMORY.md"
        try:
            # 调取底层包装的 Git 命令 `git blame -p memory/MEMORY.md` 算出了每一行上次被 commit 的时间并算出相对目前的天数
            ages = self.store.git.line_ages(file_path)
        except Exception:
            logger.debug("line_ages failed for {}", file_path)
            return content
        if not ages:
            return content

        had_trailing = content.endswith("\n")
        lines = content.splitlines()
        # If HEAD-blob line count disagrees with the working-tree content we
        # received, ages would be assigned to the wrong lines — skip entirely
        # and feed the LLM un-annotated content rather than misleading data.
        if len(lines) != len(ages):
            logger.debug(
                "line_ages length mismatch for {} (lines={}, ages={}); skipping annotation",
                file_path, len(lines), len(ages),
            )
            return content

        annotated: list[str] = []
        for line, age in zip(lines, ages):
            if not line.strip():
                annotated.append(line)
                continue

            # 当某行记忆超过了这 14 天的判定阈值，就硬编码地在这行字符串后面加一行后缀！
            # 比如 "User likes coffee.   ← 30d"
            if age.age_days > _STALE_THRESHOLD_DAYS:
                annotated.append(f"{line}  \u2190 {age.age_days}d")
            else:
                annotated.append(line)
        result = "\n".join(annotated)
        if had_trailing:
            result += "\n"
        return result

    async def run(self) -> bool:
        """Run one Dream transaction; concurrent manual/cron runs serialize."""
        async with self._run_lock:
            return await self._run_once()

    @staticmethod
    def _analysis_memory_candidates(analysis: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        raw = analysis.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            proposals = payload.get("proposals", [])
            if isinstance(proposals, list):
                for proposal in proposals:
                    if not isinstance(proposal, dict):
                        continue
                    action = str(proposal.get("action", "upsert")).strip().lower()
                    if action not in {"upsert", "remove"}:
                        continue
                    target = str(proposal.get("target", "MEMORY")).lower()
                    content = str(proposal.get("content", "")).strip()
                    old_content = str(proposal.get("old_content", "")).strip()
                    subject = str(proposal.get("subject", "")).strip()
                    if target not in {"user", "memory", "soul"}:
                        continue
                    if action == "upsert" and not content:
                        continue
                    if action == "remove" and not (subject or old_content or content):
                        continue
                    try:
                        confidence = float(proposal.get("confidence", 0.8))
                    except (TypeError, ValueError):
                        confidence = 0.8
                    candidates.append({
                        "action": action,
                        "target": target,
                        "kind": str(proposal.get("kind", "")).strip().lower(),
                        "subject": subject,
                        "content": content,
                        "old_content": old_content,
                        "confidence": min(1.0, max(0.0, confidence)),
                        "valid_from": proposal.get("valid_from"),
                        "expires_at": proposal.get("expires_at"),
                    })
        for line in analysis.splitlines():
            match = re.match(r"^\[(USER|MEMORY|SOUL)\]\s*(.+)$", line.strip())
            if not match:
                continue
            content = match.group(2).strip()
            if content:
                candidates.append({
                    "action": "upsert",
                    "target": match.group(1).lower(),
                    "kind": "",
                    "subject": "",
                    "content": content,
                    "old_content": "",
                    "confidence": 0.8,
                    "valid_from": None,
                    "expires_at": None,
                })
        return candidates

    @staticmethod
    def _analysis_skill_candidates(analysis: str) -> list[dict[str, Any]]:
        """Return only structured Skill proposals from Phase 1 JSON."""
        raw = analysis.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
            return []
        return [item for item in payload["skills"] if isinstance(item, dict)]

    @staticmethod
    def _analysis_is_skip(analysis: str) -> bool:
        raw = analysis.strip()
        if raw.upper() == "[SKIP]":
            return True
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return False
        return bool(
            isinstance(payload, dict)
            and not payload.get("proposals")
        )

    @staticmethod
    def _normalize_analysis_json(analysis: str) -> str | None:
        """Return canonical Phase 1 JSON, or None when output is malformed."""
        raw = strip_think(analysis).strip()
        if raw.upper() == "[SKIP]":
            return json.dumps({"proposals": [], "skills": []})
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            legacy_proposals = Dream._analysis_memory_candidates(raw)
            if not legacy_proposals:
                return None
            return json.dumps(
                {"proposals": legacy_proposals, "skills": []},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if not isinstance(payload, dict):
            return None
        proposals = payload.get("proposals")
        skills = payload.get("skills")
        if (
            not isinstance(proposals, list)
            or not isinstance(skills, list)
            or any(not isinstance(item, dict) for item in proposals)
            or any(not isinstance(item, dict) for item in skills)
        ):
            return None
        return json.dumps(
            {"proposals": proposals, "skills": skills},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def _run_once(self) -> bool:
        """Process unprocessed history entries. Returns True if work was done."""
        last_cursor = self.store.get_last_dream_cursor()
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        first_scope = str(entries[0].get("scope", "workspace"))
        batch: list[dict[str, Any]] = []
        for entry in entries:
            if str(entry.get("scope", "workspace")) != first_scope:
                break
            batch.append(entry)
            if len(batch) >= self.max_batch_size:
                break
        logger.info(
            "Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries), last_cursor, batch[-1]["cursor"], len(batch),
        )

        # Build history text for LLM
        history_text = "\n".join(
            f"[{e['timestamp']}] {e['content']}" for e in batch
        )

        # Current file contents + per-line age annotations (MEMORY.md only)
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.store.read_memory() or "(empty)"
        current_memory = (
            self._annotate_with_ages(raw_memory)
            if self.annotate_line_ages
            else raw_memory
        )
        current_soul = self.store.read_soul() or "(empty)"
        current_user = self.store.read_user() or "(empty)"

        # 将当前的时间、打好老化标签的 MEMORY.md、自我认知 SOUL.md 以及用户画像 USER.md 整合成上下文
        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        # Give Phase 1 the catalog summary so it can reject duplicates before
        # proposing a candidate. Full Skill files remain progressively loaded.
        existing_skills = self._list_existing_skills()
        if self.skill_candidates is not None:
            existing_skills.extend(
                f"{candidate.get('name', '')} [draft] — "
                f"{candidate.get('description', '')}"
                for candidate in self.skill_candidates.list_candidates(status="draft")
            )
        skills_section = (
            "\n\n## Existing Skill Catalog\n"
            + ("\n".join(f"- {skill}" for skill in existing_skills) or "(empty)")
        )
        phase1_prompt = (
            f"## Conversation History\n{history_text}\n\n{file_context}{skills_section}"
        )

        # Phase 1 不调用文件工具；它对比短期历史和现有长期记忆，并通过
        # JSON Schema 返回可验证的 memory / skill proposals。
        try:
            phase1_response = await self.provider.chat_structured_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": render_template(
                            "agent/dream_phase1.md",
                            strip=True,
                            stale_threshold_days=_STALE_THRESHOLD_DAYS,
                        ),
                    },
                    {"role": "user", "content": phase1_prompt},
                ],
                json_schema=_DREAM_PHASE1_SCHEMA,
                max_tokens=_DREAM_PHASE1_MAX_TOKENS,
                temperature=0.1,
                disable_thinking=True,
            )
            if phase1_response.finish_reason in {"error", "length"}:
                raise RuntimeError(
                    "Dream Phase 1 did not complete "
                    f"(finish_reason={phase1_response.finish_reason}): "
                    f"{phase1_response.content}"
                )
            raw_analysis = phase1_response.content or ""
            analysis = self._normalize_analysis_json(raw_analysis)
            if analysis is None:
                logger.warning(
                    "Dream Phase 1 returned invalid JSON ({} chars); "
                    "retaining cursor for retry",
                    len(raw_analysis),
                )
                return False
            logger.debug(
                "Dream Phase 1 analysis ({} chars): {}",
                len(analysis),
                analysis[:500],
            )
        except Exception:
            logger.exception("Dream Phase 1 failed")
            return False

        # Phase 2 applies memory proposals only. Skill proposals are staged by
        # SkillCandidateManager after the memory transaction succeeds.
        phase2_prompt = f"## Analysis Result\n{analysis}\n\n{file_context}"

        tools = self._tools
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "agent/dream_phase2.md",
                    strip=True,
                ),
            },
            {"role": "user", "content": phase2_prompt},
        ]

        managed_paths = (
            self.store.soul_file,
            self.store.user_file,
            self.store.memory_file,
        )
        snapshots = {
            path: path.read_bytes() if path.exists() else None
            for path in managed_paths
        }
        explicit_skip = self._analysis_is_skip(analysis)

        def restore_snapshots() -> None:
            for path, snapshot in snapshots.items():
                if snapshot is None:
                    path.unlink(missing_ok=True)
                else:
                    self.store._atomic_write_text(
                        path,
                        snapshot.decode("utf-8", errors="replace"),
                    )

        result = None
        phase2_succeeded = explicit_skip
        if not explicit_skip:
            attempt_messages = messages
            for attempt in range(1, _DREAM_PHASE2_ATTEMPTS + 1):
                try:
                    result = await self._runner.run(AgentRunSpec(
                        initial_messages=attempt_messages,
                        tools=tools,
                        model=self.model,
                        max_iterations=self.max_iterations,
                        max_tool_result_chars=self.max_tool_result_chars,
                        fail_on_tool_error=False,
                        disable_thinking=True,
                    ))
                except Exception:
                    result = None
                    logger.exception(
                        "Dream Phase 2 attempt {}/{} failed",
                        attempt,
                        _DREAM_PHASE2_ATTEMPTS,
                    )

                failed_events = [
                    event
                    for event in (result.tool_events if result else [])
                    if event.get("status") != "ok"
                ]
                logger.debug(
                    "Dream Phase 2 attempt {}/{} complete: "
                    "stop_reason={}, tool_events={}, failed_events={}",
                    attempt,
                    _DREAM_PHASE2_ATTEMPTS,
                    result.stop_reason if result else "exception",
                    len(result.tool_events) if result else 0,
                    len(failed_events),
                )
                for ev in (result.tool_events if result else []):
                    logger.info(
                        "Dream tool_event (attempt {}/{}): "
                        "name={}, status={}, detail={}",
                        attempt,
                        _DREAM_PHASE2_ATTEMPTS,
                        ev.get("name"),
                        ev.get("status"),
                        ev.get("detail", "")[:200],
                    )

                if result and result.stop_reason == "completed" and not failed_events:
                    phase2_succeeded = True
                    break

                restore_snapshots()
                if attempt < _DREAM_PHASE2_ATTEMPTS:
                    failure_details = "\n".join(
                        f"- {event.get('name')}: {event.get('detail', '')}"
                        for event in failed_events
                    ) or "- Phase 2 did not complete"
                    logger.warning(
                        "Dream Phase 2 attempt {}/{} incomplete; "
                        "restored snapshots and retrying",
                        attempt,
                        _DREAM_PHASE2_ATTEMPTS,
                    )
                    attempt_messages = [
                        messages[0],
                        {
                            "role": "user",
                            "content": (
                                f"{phase2_prompt}\n\n"
                                "## Previous attempt errors\n"
                                f"{failure_details}\n\n"
                                "Retry the complete proposal transaction from the "
                                "original file contents. Use only the exact relative "
                                "paths SOUL.md, USER.md, and memory/MEMORY.md."
                            ),
                        },
                    ]

        if not phase2_succeeded:
            restore_snapshots()
            reason = result.stop_reason if result else "exception"
            logger.warning(
                "Dream incomplete ({}); restored memory files and retained cursor {}",
                reason,
                last_cursor,
            )
            return False

        # Build changelog from tool events
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event["status"] == "ok":
                    changelog.append(f"{event['name']}: {event['detail']}")

        new_cursor = batch[-1]["cursor"]
        batch_scopes = {
            str(entry.get("scope", "workspace")) for entry in batch
        }
        record_scope = batch_scopes.pop() if len(batch_scopes) == 1 else "workspace"
        evidence = [
            {
                "event_id": entry.get("event_id"),
                "cursor": entry.get("cursor"),
                "scope": entry.get("scope", "workspace"),
            }
            for entry in batch
        ]
        try:
            proposals = self._analysis_memory_candidates(analysis)
            for candidate in proposals:
                file_kind = candidate["target"]
                candidate["kind"] = candidate["kind"] or {
                    "user": "preference",
                    "memory": "fact",
                    "soul": "behavior",
                }[file_kind]
            self.store.apply_memory_proposals(
                proposals,
                scope=record_scope,
                evidence=evidence,
                source_cursor=new_cursor,
            )

            if self.skill_candidates is not None:
                for proposal in self._analysis_skill_candidates(analysis):
                    try:
                        staged = self.skill_candidates.stage(
                            proposal,
                            source="dream",
                            evidence=evidence,
                        )
                        logger.info(
                            "Dream staged Skill candidate {} ({})",
                            staged["candidate_id"],
                            staged["name"],
                        )
                    except (OSError, ValueError):
                        # A malformed optional Skill must not roll back an
                        # otherwise valid memory consolidation transaction.
                        logger.exception("Dream rejected an invalid Skill proposal")
        except Exception:
            logger.exception("Dream structured-memory commit failed; retaining cursor")
            for path, snapshot in snapshots.items():
                if snapshot is None:
                    path.unlink(missing_ok=True)
                else:
                    self.store._atomic_write_text(
                        path,
                        snapshot.decode("utf-8", errors="replace"),
                    )
            return False

        # Version the human-readable views after both stores have accepted the
        # update, then advance the cursor as the final commit marker.
        if changelog and self.store.git.is_initialized():
            ts = batch[-1]["timestamp"]
            summary = f"dream: {ts}, {len(changelog)} change(s)"
            commit_msg = f"{summary}\n\n{analysis.strip()}"
            sha = self.store.git.auto_commit(commit_msg)
            if sha:
                logger.info("Dream commit: {}", sha)

        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history(processed_cursor=new_cursor)
        logger.info(
            "Dream done: {} change(s), cursor advanced to {}",
            len(changelog), new_cursor,
        )

        return True
