"""Lifecycle management and usage telemetry for generated Agent Skills."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from nanobot.agent.skills import BUILTIN_SKILLS_DIR


_SKILL_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_FRONTMATTER_RE = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)
_LIFECYCLE_LOCKS: dict[str, threading.RLock] = {}
_LIFECYCLE_LOCKS_GUARD = threading.Lock()


def _shared_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LIFECYCLE_LOCKS_GUARD:
        return _LIFECYCLE_LOCKS.setdefault(key, threading.RLock())


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
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class SkillValidationResult:
    valid: bool
    message: str
    metadata: dict[str, Any] | None = None


def validate_generated_skill(
    skill_dir: Path,
    *,
    max_chars: int = 128_000,
    max_lines: int = 500,
) -> SkillValidationResult:
    """Validate one instruction-only generated skill before publication."""
    if skill_dir.is_symlink():
        return SkillValidationResult(False, "Skill directory is missing or is a symlink")
    skill_dir = skill_dir.resolve()
    if not skill_dir.is_dir():
        return SkillValidationResult(False, "Skill directory is missing or is a symlink")
    if not _SKILL_NAME_RE.fullmatch(skill_dir.name) or len(skill_dir.name) > 64:
        return SkillValidationResult(False, "Skill directory name must be kebab-case and <=64 chars")

    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file() or skill_file.is_symlink():
        return SkillValidationResult(False, "SKILL.md is missing or is a symlink")
    try:
        content = skill_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return SkillValidationResult(False, f"SKILL.md cannot be read as UTF-8: {exc}")
    if len(content) > max_chars:
        return SkillValidationResult(False, f"SKILL.md exceeds {max_chars} characters")
    if len(content.splitlines()) > max_lines:
        return SkillValidationResult(False, f"SKILL.md exceeds {max_lines} lines")

    match = _FRONTMATTER_RE.match(content)
    if not match:
        return SkillValidationResult(False, "SKILL.md requires YAML frontmatter")
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return SkillValidationResult(False, f"Invalid YAML frontmatter: {exc}")
    if not isinstance(metadata, dict):
        return SkillValidationResult(False, "Frontmatter must be a mapping")
    if metadata.get("name") != skill_dir.name:
        return SkillValidationResult(False, "Frontmatter name must match the directory")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        return SkillValidationResult(False, "Frontmatter requires a non-empty description")
    if len(description) > 1024:
        return SkillValidationResult(False, "Skill description exceeds 1024 characters")
    if metadata.get("always") is True:
        return SkillValidationResult(False, "Generated skills cannot be always-on")

    body = content[match.end():].strip()
    required_sections = ("## When to Use", "## Workflow", "## Completion Criteria")
    missing = [section for section in required_sections if section not in body]
    if missing:
        return SkillValidationResult(
            False,
            "Generated skill is missing section(s): " + ", ".join(missing),
        )
    for child in skill_dir.rglob("*"):
        if child.is_symlink():
            return SkillValidationResult(False, f"Symlinks are not allowed: {child.name}")
    unexpected = [child.name for child in skill_dir.iterdir() if child.name != "SKILL.md"]
    if unexpected:
        return SkillValidationResult(
            False,
            "Generated skills must be instruction-only; unexpected entries: "
            + ", ".join(sorted(unexpected)),
        )
    return SkillValidationResult(True, "Skill is valid", metadata)


class SkillCandidateManager:
    """Stage, validate, version and optionally publish generated skills."""

    def __init__(
        self,
        workspace: Path,
        *,
        auto_promote: bool = False,
        max_skill_chars: int = 128_000,
        max_skill_lines: int = 500,
    ):
        self.workspace = workspace.resolve()
        self.skills_dir = self.workspace / "skills"
        self.candidates_dir = self.skills_dir / ".candidates"
        self.history_dir = self.skills_dir / ".history"
        self.auto_promote = auto_promote
        self.max_skill_chars = max_skill_chars
        self.max_skill_lines = max_skill_lines
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _shared_lock(self.skills_dir)
        self._reserved_names = {
            path.name
            for path in BUILTIN_SKILLS_DIR.iterdir()
            if path.is_dir()
        } if BUILTIN_SKILLS_DIR.exists() else set()

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value[:20]:
            text = re.sub(r"\s+", " ", str(item)).strip()
            if text:
                normalized.append(text[:2_000])
        return normalized

    @staticmethod
    def _bullet_lines(items: list[str], fallback: str) -> str:
        values = items or [fallback]
        return "\n".join(f"- {value}" for value in values)

    def _render_skill(self, proposal: dict[str, Any]) -> str:
        name = str(proposal["name"])
        description = str(proposal["description"]).strip()
        when_to_use = self._as_list(proposal.get("when_to_use"))
        steps = self._as_list(proposal.get("steps"))
        completion = self._as_list(proposal.get("completion_criteria"))
        recovery = self._as_list(proposal.get("failure_recovery"))
        examples = self._as_list(proposal.get("examples"))

        workflow = "\n".join(
            f"{index}. {step}" for index, step in enumerate(steps, 1)
        ) or "1. Follow the validated workflow described for this task."
        sections = [
            "---",
            f"name: {name}",
            f"description: {json.dumps(description, ensure_ascii=False)}",
            "---",
            "",
            f"# {name.replace('-', ' ').title()}",
            "",
            "## When to Use",
            "",
            self._bullet_lines(when_to_use, description),
            "",
            "## Workflow",
            "",
            workflow,
            "",
            "## Completion Criteria",
            "",
            self._bullet_lines(completion, "Verify that the requested outcome is complete and grounded."),
        ]
        if recovery:
            sections.extend([
                "",
                "## Failure Recovery",
                "",
                self._bullet_lines(recovery, "Report the failure and preserve recoverable state."),
            ])
        if examples:
            sections.extend([
                "",
                "## Example Queries",
                "",
                self._bullet_lines(examples, ""),
            ])
        return "\n".join(sections).rstrip() + "\n"

    def stage(
        self,
        proposal: dict[str, Any],
        *,
        source: str,
        evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a validated draft candidate and optionally promote it."""
        name = str(proposal.get("name", "")).strip().lower()
        description = re.sub(
            r"\s+",
            " ",
            str(proposal.get("description", "")),
        ).strip()
        action = str(proposal.get("action", "create")).strip().lower()
        if not _SKILL_NAME_RE.fullmatch(name) or len(name) > 64:
            raise ValueError("Generated skill name must be kebab-case and <=64 chars")
        if name in self._reserved_names:
            raise ValueError(f"Generated skill name is reserved by a builtin skill: {name}")
        if not description or len(description) > 1024:
            raise ValueError("Generated skill description must contain 1-1024 characters")
        if action not in {"create", "update"}:
            raise ValueError("Skill proposal action must be create or update")

        when_to_use = self._as_list(proposal.get("when_to_use"))
        steps = self._as_list(proposal.get("steps"))
        completion = self._as_list(proposal.get("completion_criteria"))
        if not when_to_use:
            raise ValueError("Generated skill requires at least one use trigger")
        if len(steps) < 2:
            raise ValueError("Generated skill requires at least two workflow steps")
        if not completion:
            raise ValueError("Generated skill requires at least one completion criterion")

        target_exists = (self.skills_dir / name / "SKILL.md").exists()
        if target_exists and action == "create":
            action = "update"
        if action == "update" and not target_exists:
            action = "create"

        normalized = dict(proposal)
        normalized.update({
            "name": name,
            "description": description,
            "action": action,
            "when_to_use": when_to_use,
            "steps": steps,
            "completion_criteria": completion,
            "failure_recovery": self._as_list(proposal.get("failure_recovery")),
            "examples": self._as_list(proposal.get("examples")),
        })
        content = self._render_skill(normalized)
        candidate_id = (
            datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
            + "-"
            + uuid.uuid4().hex[:10]
        )
        candidate_root = self.candidates_dir / candidate_id
        manifest = {
            "candidate_id": candidate_id,
            "name": name,
            "description": description,
            "action": action,
            "source": source,
            "status": "draft",
            "created_at": datetime.now().astimezone().isoformat(),
            "evidence": evidence or [],
            "proposal": normalized,
        }

        with self._lock:
            if any(
                candidate.get("name") == name
                for candidate in self.list_candidates(status="draft")
            ):
                raise ValueError(f"A draft candidate already exists for skill: {name}")
            temporary_root = Path(tempfile.mkdtemp(prefix=".candidate-", dir=self.candidates_dir))
            try:
                skill_dir = temporary_root / name
                skill_dir.mkdir()
                _atomic_write_text(skill_dir / "SKILL.md", content)
                validation = validate_generated_skill(
                    skill_dir,
                    max_chars=self.max_skill_chars,
                    max_lines=self.max_skill_lines,
                )
                if not validation.valid:
                    raise ValueError(validation.message)
                manifest["validation"] = {
                    "valid": True,
                    "message": validation.message,
                }
                _atomic_write_text(
                    temporary_root / "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                )
                os.replace(temporary_root, candidate_root)
            finally:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)

        result = {
            "candidate_id": candidate_id,
            "name": name,
            "status": "draft",
            "path": str(candidate_root),
        }
        # Updating an existing skill can silently discard operator-authored
        # instructions. Keep update proposals reviewable even when automatic
        # publication is enabled; automatic promotion is limited to new names.
        if self.auto_promote and action == "create":
            result.update(self.promote(candidate_id))
        return result

    def get_candidate(self, candidate_id: str) -> tuple[Path, dict[str, Any]]:
        """Load one candidate root and manifest after path validation."""
        if not re.fullmatch(r"[0-9A-Za-z-]+", candidate_id):
            raise ValueError("Invalid candidate id")
        unresolved_root = self.candidates_dir / candidate_id
        if unresolved_root.is_symlink():
            raise ValueError("Candidate directory cannot be a symlink")
        root = unresolved_root.resolve()
        try:
            root.relative_to(self.candidates_dir.resolve())
        except ValueError as exc:
            raise ValueError("Candidate path escapes the candidate directory") from exc
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise FileNotFoundError(f"Unknown skill candidate: {candidate_id}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Candidate manifest is invalid")
        if manifest.get("candidate_id") != candidate_id:
            raise ValueError("Candidate manifest id does not match its directory")
        name = str(manifest.get("name", ""))
        if not _SKILL_NAME_RE.fullmatch(name) or len(name) > 64:
            raise ValueError("Candidate manifest contains an invalid skill name")
        return root, manifest

    def promote(self, candidate_id: str) -> dict[str, Any]:
        """Atomically publish one validated draft, versioning replaced content."""
        with self._lock:
            root, manifest = self.get_candidate(candidate_id)
            if manifest.get("status") == "promoted":
                return {
                    "candidate_id": candidate_id,
                    "name": manifest.get("name", ""),
                    "status": "promoted",
                }
            if manifest.get("status") != "draft":
                raise ValueError(f"Candidate is not promotable: {manifest.get('status')}")
            name = str(manifest.get("name", ""))
            if not _SKILL_NAME_RE.fullmatch(name) or len(name) > 64:
                raise ValueError("Candidate manifest contains an invalid skill name")
            if name in self._reserved_names:
                raise ValueError(f"Cannot replace builtin skill: {name}")
            candidate_skill_dir = root / name
            validation = validate_generated_skill(
                candidate_skill_dir,
                max_chars=self.max_skill_chars,
                max_lines=self.max_skill_lines,
            )
            if not validation.valid:
                raise ValueError(validation.message)

            target_dir = self.skills_dir / name
            target_file = target_dir / "SKILL.md"
            if target_dir.is_symlink():
                raise ValueError("Cannot publish over a symlinked skill directory")
            if target_file.exists():
                revision_dir = (
                    self.history_dir
                    / name
                    / datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
                )
                revision_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(target_dir, revision_dir, symlinks=True)
                new_content = (candidate_skill_dir / "SKILL.md").read_text(encoding="utf-8")
                _atomic_write_text(target_file, new_content)
            else:
                temporary_target = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.skills_dir))
                try:
                    _atomic_write_text(
                        temporary_target / "SKILL.md",
                        (candidate_skill_dir / "SKILL.md").read_text(encoding="utf-8"),
                    )
                    os.replace(temporary_target, target_dir)
                finally:
                    if temporary_target.exists():
                        shutil.rmtree(temporary_target)

            manifest["status"] = "promoted"
            manifest["promoted_at"] = datetime.now().astimezone().isoformat()
            _atomic_write_text(
                root / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            return {
                "candidate_id": candidate_id,
                "name": name,
                "status": "promoted",
                "path": str(target_dir),
            }

    def reject(self, candidate_id: str, reason: str = "") -> dict[str, Any]:
        with self._lock:
            root, manifest = self.get_candidate(candidate_id)
            if manifest.get("status") == "promoted":
                raise ValueError("A promoted candidate cannot be rejected")
            manifest["status"] = "rejected"
            manifest["rejected_at"] = datetime.now().astimezone().isoformat()
            manifest["rejection_reason"] = reason.strip()
            _atomic_write_text(
                root / "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            return {
                "candidate_id": candidate_id,
                "name": str(manifest.get("name", "")),
                "status": "rejected",
            }

    def list_candidates(self, *, status: str | None = None) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for root in sorted(self.candidates_dir.iterdir()):
            manifest_path = root / "manifest.json"
            if (
                not root.is_dir()
                or root.is_symlink()
                or not manifest_path.is_file()
                or manifest_path.is_symlink()
            ):
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict):
                continue
            if status is not None and manifest.get("status") != status:
                continue
            candidates.append(manifest)
        return candidates


class SkillUsageStore:
    """Durable activation/outcome counters for model-selected skills."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.skills_dir = self.workspace / "skills"
        self.database = self.skills_dir / ".usage.sqlite3"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._lock = _shared_lock(self.database)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_usage (
                    skill_name TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    activation_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_used_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_activation_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    skill_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    session_key TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_outcome_events (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    skill_name TEXT NOT NULL,
                    session_key TEXT,
                    completed INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            activation_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(skill_activation_events)"
                ).fetchall()
            }
            if "run_id" not in activation_columns:
                try:
                    connection.execute(
                        "ALTER TABLE skill_activation_events ADD COLUMN run_id TEXT"
                    )
                except sqlite3.OperationalError as exc:
                    # Another process may have completed the same idempotent
                    # migration after our PRAGMA read.
                    if "duplicate column" not in str(exc).lower():
                        raise

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def identify(self, path: str | Path) -> tuple[str, str] | None:
        try:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = self.workspace / resolved
            resolved = resolved.resolve()
        except (OSError, RuntimeError):
            return None
        if resolved.name != "SKILL.md":
            return None
        for root, source in (
            (self.skills_dir.resolve(), "workspace"),
            (BUILTIN_SKILLS_DIR.resolve(), "builtin"),
        ):
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) == 2 and not relative.parts[0].startswith("."):
                return relative.parts[0], source
        return None

    @staticmethod
    def new_run_id() -> str:
        return f"skill_run_{uuid.uuid4().hex}"

    def record_activation(
        self,
        path: str | Path,
        *,
        session_key: str | None,
        run_id: str | None = None,
    ) -> str | None:
        identified = self.identify(path)
        if identified is None:
            return None
        name, source = identified
        now = datetime.now().astimezone().isoformat()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO skill_usage(
                    skill_name, source, activation_count, last_used_at, updated_at
                ) VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    source = excluded.source,
                    activation_count = skill_usage.activation_count + 1,
                    last_used_at = excluded.last_used_at,
                    updated_at = excluded.updated_at
                """,
                (name, source, now, now),
            )
            connection.execute(
                """
                INSERT INTO skill_activation_events(
                    id, run_id, skill_name, source, session_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"skill_evt_{uuid.uuid4().hex}",
                    run_id,
                    name,
                    source,
                    session_key,
                    now,
                ),
            )
        return name

    def record_outcome(
        self,
        skill_names: set[str],
        *,
        success: bool,
        session_key: str | None = None,
        run_id: str | None = None,
    ) -> None:
        if not skill_names:
            return
        now = datetime.now().astimezone().isoformat()
        column = "success_count" if success else "failure_count"
        with self._lock, self._connect() as connection:
            for name in sorted(skill_names):
                connection.execute(
                    f"UPDATE skill_usage SET {column} = {column} + 1, updated_at = ? "
                    "WHERE skill_name = ?",
                    (now, name),
                )
                connection.execute(
                    """
                    INSERT INTO skill_outcome_events(
                        id, run_id, skill_name, session_key, completed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"skill_outcome_{uuid.uuid4().hex}",
                        run_id,
                        name,
                        session_key,
                        int(success),
                        now,
                    ),
                )

    def stats(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM skill_usage ORDER BY activation_count DESC, skill_name"
                ).fetchall()
            ]
