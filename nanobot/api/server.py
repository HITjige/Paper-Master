"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json as _json
import mimetypes
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import web
from loguru import logger

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import safe_filename
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_JSON_REQUEST_SIZE = 20 * 1024 * 1024
MAX_PAPER_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB per PDF
MAX_PAPER_UPLOAD_TOTAL_SIZE = 200 * 1024 * 1024
MAX_PAPER_UPLOAD_FILES = 10
MAX_PDF_PAGES = 2000
PAPER_UPLOAD_CHUNK_SIZE = 1024 * 1024
_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


class _FileSizeExceeded(Exception):
    """Raised when an uploaded file exceeds the size limit."""


class _PaperUploadError(Exception):
    """Stable, user-safe validation error for one uploaded paper."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "validation",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.retryable = retryable


@dataclass(slots=True)
class _UploadedPaper:
    original_filename: str
    path: Path
    sha256: str
    size_bytes: int
    page_count: int
    pdf_title: str = ""


API_SESSION_KEY = "api:default"
API_CHAT_ID = "default"


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )


def _chat_completion_response(content: str, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _response_text(value: Any) -> str:
    """Normalize process_direct output to plain assistant text."""
    if value is None:
        return ""
    if hasattr(value, "content"):
        return str(getattr(value, "content") or "")
    return str(value)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_chunk(delta: str, model: str, chunk_id: str, finish_reason: str | None = None) -> bytes:
    """Format a single OpenAI-compatible SSE chunk."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {_json.dumps(payload)}\n\n".encode()


_SSE_DONE = b"data: [DONE]\n\n"

# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


def _save_base64_data_url(data_url: str, media_dir: Path) -> str | None:
    """Decode a data:...;base64,... URL and save to disk."""
    m = _DATA_URL_RE.match(data_url)
    if not m:
        return None
    mime_type, b64_payload = m.group(1), m.group(2)
    try:
        raw = base64.b64decode(b64_payload)
    except Exception:
        return None
    if len(raw) > MAX_FILE_SIZE:
        raise _FileSizeExceeded(f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit")
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    filename = f"{uuid.uuid4().hex[:12]}{ext}"
    dest = media_dir / safe_filename(filename)
    dest.write_bytes(raw)
    return str(dest)


def _parse_json_content(body: dict) -> tuple[str, list[str]]:
    """Parse JSON request body. Returns (text, media_paths)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Only a single user message is supported")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("Only a single user message is supported")

    user_content = message.get("content", "")
    media_dir = get_media_dir("api")
    media_paths: list[str] = []

    if isinstance(user_content, list):
        text_parts: list[str] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    saved = _save_base64_data_url(url, media_dir)
                    if saved:
                        media_paths.append(saved)
                elif url:
                    raise ValueError(
                        "Remote image URLs are not supported. "
                        "Use base64 data URLs or upload files via multipart/form-data."
                    )
        text = " ".join(text_parts)
    elif isinstance(user_content, str):
        text = user_content
    else:
        raise ValueError("Invalid content format")

    return text, media_paths


async def _parse_multipart(request: web.Request) -> tuple[str, list[str], str | None, str | None]:
    """Parse multipart/form-data. Returns (text, media_paths, session_id, model)."""
    media_dir = get_media_dir("api")
    reader = await request.multipart()
    text = ""
    session_id = None
    model = None
    media_paths: list[str] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "message":
            text = await _read_multipart_text(part, "message")
        elif part.name == "session_id":
            session_id = (await _read_multipart_text(part, "session_id")).strip()
        elif part.name == "model":
            model = (await _read_multipart_text(part, "model")).strip()
        elif part.name == "files":
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            temp_dest = media_dir / f".{filename}.part"
            size = 0
            try:
                with temp_dest.open("xb") as output:
                    while True:
                        chunk = await part.read_chunk(size=PAPER_UPLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_FILE_SIZE:
                            raise _FileSizeExceeded(
                                f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                            )
                        output.write(chunk)
                os.replace(temp_dest, dest)
            except Exception:
                temp_dest.unlink(missing_ok=True)
                raise
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model


async def _read_multipart_text(
    part: Any,
    field_name: str,
    *,
    max_bytes: int = 64 * 1024,
) -> str:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await part.read_chunk(size=min(16 * 1024, max_bytes + 1))
        if not chunk:
            break
        size += len(chunk)
        if size > max_bytes:
            raise ValueError(f"Multipart field '{field_name}' is too large")
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8")


async def _read_json_body_limited(
    request: web.Request,
    *,
    max_bytes: int = MAX_JSON_REQUEST_SIZE,
) -> dict[str, Any]:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.content.iter_chunked(64 * 1024):
        size += len(chunk)
        if size > max_bytes:
            raise _FileSizeExceeded(
                f"JSON request exceeds {max_bytes // (1024 * 1024)}MB limit"
            )
        chunks.append(chunk)
    payload = _json.loads(b"".join(chunks))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _inspect_pdf(path: Path, max_pages: int = MAX_PDF_PAGES) -> tuple[int, str]:
    """Open a PDF without extracting its body and return basic trusted metadata."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path, strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise _PaperUploadError(
                "PDF_ENCRYPTED",
                "Password-protected PDFs are not supported",
            )
        page_count = len(reader.pages)
        if page_count < 1:
            raise _PaperUploadError("PDF_EMPTY", "PDF contains no pages")
        if page_count > max_pages:
            raise _PaperUploadError(
                "PDF_TOO_MANY_PAGES",
                f"PDF exceeds the {max_pages}-page limit",
            )
        metadata = reader.metadata
        title = str(getattr(metadata, "title", "") or "").strip()[:200]
        return page_count, title
    except _PaperUploadError:
        raise
    except Exception as exc:
        logger.info("Rejected invalid uploaded PDF {}: {}", path.name, exc)
        raise _PaperUploadError(
            "PDF_INVALID",
            "The uploaded file is not a valid PDF",
        ) from exc


async def _stream_paper_upload(
    part: Any,
    uploads_dir: Path,
    *,
    max_size: int = MAX_PAPER_UPLOAD_SIZE,
    max_pages: int = MAX_PDF_PAGES,
) -> _UploadedPaper:
    """Stream one multipart PDF to a content-addressed file with early limits."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    original_filename = str(part.filename or "paper.pdf")[:500]
    temp_path = uploads_dir / f".upload.{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    header = bytearray()
    size = 0
    try:
        with temp_path.open("xb") as output:
            while True:
                chunk = await part.read_chunk(size=PAPER_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_size:
                    raise _PaperUploadError(
                        "PDF_TOO_LARGE",
                        f"PDF exceeds the {max_size // (1024 * 1024)}MB limit",
                    )
                if len(header) < 1024:
                    header.extend(chunk[: 1024 - len(header)])
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if not size:
            raise _PaperUploadError("PDF_EMPTY", "Uploaded PDF is empty")
        if b"%PDF-" not in bytes(header):
            raise _PaperUploadError(
                "PDF_SIGNATURE_INVALID",
                "Uploaded file does not have a valid PDF signature",
            )

        # This is a metadata-only pass after the streaming byte/page limits.
        # Full extraction remains in the bounded background ingestion worker.
        page_count, pdf_title = _inspect_pdf(temp_path, max_pages)

        sha256 = digest.hexdigest()
        final_path = uploads_dir / f"{sha256}.pdf"
        if final_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            os.replace(temp_path, final_path)
        return _UploadedPaper(
            original_filename=original_filename,
            path=final_path,
            sha256=sha256,
            size_bytes=size,
            page_count=page_count,
            pdf_title=pdf_title,
        )
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _paper_error_result(
    filename: str | None,
    error: _PaperUploadError,
) -> dict[str, Any]:
    return {
        "filename": filename or "paper.pdf",
        "status": "error",
        "stage": error.stage,
        "error_code": error.code,
        "error": error.message,
        "retryable": error.retryable,
    }


def _request_bearer_token(request: web.Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _origin_is_allowed(origin: str, allowed_origins: set[str]) -> bool:
    if not origin:
        return True
    if origin in allowed_origins:
        return True
    try:
        parsed = urlparse(origin)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _positive_int_setting(owner: Any, name: str, default: int) -> int:
    value = getattr(owner, name, default) if owner is not None else default
    return value if isinstance(value, int) and value > 0 else default


async def _ingest_uploaded_paper(
    agent_loop: Any,
    kb: Any,
    uploaded: _UploadedPaper,
    ingest_locks: dict[str, asyncio.Lock],
) -> dict[str, Any]:
    """Ingest one validated, content-addressed PDF and normalize its result."""
    paper_id = f"upload:{uploaded.sha256}"
    local_md_path = uploaded.path.with_suffix(".md")
    filename_title = Path(
        safe_filename(uploaded.original_filename) or "paper.pdf"
    ).stem[:200]
    doc = {
        "paper_id": paper_id,
        "title": uploaded.pdf_title or filename_title or "Uploaded paper",
        "source": "upload",
        "url": str(local_md_path),
        "year": None,
        "uploaded_at": datetime.now().astimezone().isoformat(),
        "original_filename": uploaded.original_filename,
        "content_sha256": uploaded.sha256,
        "size_bytes": uploaded.size_bytes,
        "page_count": uploaded.page_count,
        "metadata_provenance": {
            "title": "pdf_metadata" if uploaded.pdf_title else "filename",
            "year": "unknown",
        },
    }

    lock = ingest_locks.setdefault(paper_id, asyncio.Lock())
    async with lock:
        existing_meta = kb.load_docs_meta().get(paper_id)
        existing_chunk_count = sum(
            1
            for row in kb._read_jsonl(kb.chunks_file)
            if str(row.get("paper_id", "")) == paper_id
        )
        if existing_meta and existing_chunk_count:
            return {
                "filename": uploaded.original_filename,
                "paper_id": paper_id,
                "title": existing_meta.get("title") or doc["title"],
                "status": "ok",
                "stage": "completed",
                "chunk_count": existing_chunk_count,
                "deduplicated": True,
                "content_sha256": uploaded.sha256,
            }

        ingest_result = await agent_loop.kb_ingest_local(
            doc=doc,
            local_pdf_path=str(uploaded.path),
        )
        status = str(ingest_result.get("status", "error"))
        row: dict[str, Any] = {
            "filename": uploaded.original_filename,
            "paper_id": paper_id,
            "title": doc.get("title", ""),
            "status": status,
            "stage": "completed" if status == "ok" else "ingestion",
            "chunk_count": int(ingest_result.get("chunk_count", 0) or 0),
            "deduplicated": False,
            "content_sha256": uploaded.sha256,
        }
        if status == "ok":
            row.update({
                "degraded": bool(ingest_result.get("degraded", False)),
                "storage_backend": ingest_result.get("storage_backend", "unknown"),
                "degradation_reasons": ingest_result.get("degradation_reasons", []),
                "parser_name": ingest_result.get("parser_name", ""),
                "parse_quality_score": ingest_result.get("parse_quality_score"),
            })
        else:
            internal_error = str(ingest_result.get("error", "INGEST_FAILED"))
            public_messages = {
                "failed_to_parse_content": "No usable text could be extracted from the PDF",
                "no_chunks_generated": "No indexable sections were found in the PDF",
                "Knowledge base not enabled": "Knowledge base is not enabled",
                "Paper ingest tool not available": "Paper ingestion is not available",
            }
            row.update({
                "error_code": internal_error.upper().replace(" ", "_"),
                "error": public_messages.get(internal_error, "PDF ingestion failed"),
                "retryable": internal_error not in {"no_chunks_generated"},
            })
        return row


class _PaperIngestJobManager:
    """Small in-process job registry with bounded expensive ingestion work."""

    def __init__(self, concurrency: int = 2, max_jobs: int = 200) -> None:
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._max_jobs = max(10, max_jobs)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    def submit(
        self,
        *,
        agent_loop: Any,
        kb: Any,
        uploads: list[_UploadedPaper],
        ingest_locks: dict[str, asyncio.Lock],
        validation_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        job_id = f"ingest_{uuid.uuid4().hex}"
        now = datetime.now().astimezone().isoformat()
        pending = [
            {
                "filename": upload.original_filename,
                "paper_id": f"upload:{upload.sha256}",
                "status": "pending",
                "stage": "queued",
                "content_sha256": upload.sha256,
            }
            for upload in uploads
        ]
        job = {
            "job_id": job_id,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "total": len(validation_results) + len(uploads),
            "completed": len(validation_results),
            "results": [*validation_results, *pending],
        }
        self._jobs[job_id] = job
        task = asyncio.create_task(
            self._run(job, agent_loop, kb, uploads, ingest_locks),
            name=job_id,
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._prune()
        return dict(job)

    async def _run(
        self,
        job: dict[str, Any],
        agent_loop: Any,
        kb: Any,
        uploads: list[_UploadedPaper],
        ingest_locks: dict[str, asyncio.Lock],
    ) -> None:
        job["status"] = "running"
        job["updated_at"] = datetime.now().astimezone().isoformat()
        validation_count = len(job["results"]) - len(uploads)
        try:
            for index, upload in enumerate(uploads):
                job["results"][validation_count + index]["stage"] = "ingestion"
                async with self._semaphore:
                    try:
                        result = await _ingest_uploaded_paper(
                            agent_loop,
                            kb,
                            upload,
                            ingest_locks,
                        )
                    except Exception:
                        logger.exception("Background PDF ingestion failed for {}", upload.original_filename)
                        result = _paper_error_result(
                            upload.original_filename,
                            _PaperUploadError(
                                "INGEST_INTERNAL_ERROR",
                                "PDF ingestion failed due to an internal error",
                                stage="ingestion",
                                retryable=True,
                            ),
                        )
                    job["results"][validation_count + index] = result
                    job["completed"] += 1
                    job["updated_at"] = datetime.now().astimezone().isoformat()
            succeeded = sum(
                result.get("status") == "ok" for result in job["results"]
            )
            job["succeeded"] = succeeded
            job["failed"] = len(job["results"]) - succeeded
            job["status"] = "completed"
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            raise
        finally:
            job["updated_at"] = datetime.now().astimezone().isoformat()

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return dict(job) if job is not None else None

    def _prune(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        completed = [
            job_id
            for job_id, job in self._jobs.items()
            if job.get("status") in {"completed", "cancelled"}
        ]
        for job_id in completed[: len(self._jobs) - self._max_jobs]:
            self._jobs.pop(job_id, None)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def handle_chat_completions(request: web.Request) -> web.Response:
    """POST /v1/chat/completions — supports JSON and multipart/form-data."""
    content_type = request.content_type or ""
    if not isinstance(content_type, str):
        content_type = ""

    agent_loop = request.app["agent_loop"]
    timeout_s: float = request.app.get("request_timeout", 120.0)
    model_name: str = request.app.get("model_name", "nanobot")

    stream = False
    try:
        if content_type.startswith("multipart/"):
            text, media_paths, session_id, requested_model = await _parse_multipart(request)
        else:
            try:
                body = await _read_json_body_limited(request)
            except _FileSizeExceeded:
                raise
            except Exception:
                return _error_json(400, "Invalid JSON body")
            stream = body.get("stream", False)
            requested_model = body.get("model")
            text, media_paths = _parse_json_content(body)
            session_id = body.get("session_id")
    except ValueError as e:
        return _error_json(400, str(e))
    except _FileSizeExceeded as e:
        return _error_json(413, str(e), err_type="invalid_request_error")
    except Exception:
        logger.exception("Error parsing upload")
        return _error_json(413, "File too large or invalid upload")

    if requested_model and requested_model != model_name:
        return _error_json(400, f"Only configured model '{model_name}' is available")

    session_key = f"api:{session_id}" if session_id else API_SESSION_KEY
    session_locks: dict[str, asyncio.Lock] = request.app["session_locks"]
    session_lock = session_locks.setdefault(session_key, asyncio.Lock())

    logger.info(
        "API request session_key={} media={} text={} stream={}",
        session_key, len(media_paths), text[:80], stream,
    )
    # -- streaming path --
    if stream:
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["Connection"] = "keep-alive"
        resp.enable_compression()
        await resp.prepare(request)

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_failed = False

        async def _on_stream(token: str) -> None:
            await queue.put(token)

        async def _on_stream_end(*_a: Any, **_kw: Any) -> None:
            await queue.put(None)

        async def _run() -> None:
            nonlocal stream_failed
            try:
                async with session_lock:
                    await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                            on_stream=_on_stream,
                            on_stream_end=_on_stream_end,
                        ),
                        timeout=timeout_s,
                    )
            except Exception:
                stream_failed = True
                logger.exception("Streaming error for session {}", session_key)
                await queue.put(None)

        task = asyncio.create_task(_run())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    break
                await resp.write(_sse_chunk(token, model_name, chunk_id))
        finally:
            task.cancel()

        if not stream_failed:
            await resp.write(_sse_chunk("", model_name, chunk_id, finish_reason="stop"))
            await resp.write(_SSE_DONE)
        return resp

    # -- non-streaming path (original logic) --
    _FALLBACK = EMPTY_FINAL_RESPONSE_MESSAGE

    try:
        async with session_lock:
            try:
                response = await asyncio.wait_for(
                    agent_loop.process_direct(
                        content=text,
                        media=media_paths if media_paths else None,
                        session_key=session_key,
                        channel="api",
                        chat_id=API_CHAT_ID,
                    ),
                    timeout=timeout_s,
                )
                response_text = _response_text(response)

                if not response_text or not response_text.strip():
                    logger.warning("Empty response for session {}, retrying", session_key)
                    retry_response = await asyncio.wait_for(
                        agent_loop.process_direct(
                            content=text,
                            media=media_paths if media_paths else None,
                            session_key=session_key,
                            channel="api",
                            chat_id=API_CHAT_ID,
                        ),
                        timeout=timeout_s,
                    )
                    response_text = _response_text(retry_response)
                    if not response_text or not response_text.strip():
                        logger.warning("Empty response after retry, using fallback")
                        response_text = _FALLBACK

            except asyncio.TimeoutError:
                return _error_json(504, f"Request timed out after {timeout_s}s")
            except Exception:
                logger.exception("Error processing request for session {}", session_key)
                return _error_json(500, "Internal server error", err_type="server_error")
    except Exception:
        logger.exception("Unexpected API lock error for session {}", session_key)
        return _error_json(500, "Internal server error", err_type="server_error")

    return web.json_response(_chat_completion_response(response_text, model_name))


async def handle_models(request: web.Request) -> web.Response:
    """GET /v1/models"""
    model_name = request.app.get("model_name", "nanobot")
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanobot",
                }
            ],
        }
    )


async def handle_health(request: web.Request) -> web.Response:
    """GET /health"""
    return web.json_response({"status": "ok"})


# ---------------------------------------------------------------------------
# KB endpoints
# ---------------------------------------------------------------------------


async def handle_papers_upload(request: web.Request) -> web.Response:
    """POST /api/papers/upload — multipart PDF upload for KB ingestion.

    Accepts one or more PDF files via the ``files`` field.
    Saves them locally then ingests through PaperKnowledgeBase.

    Returns JSON with per-file results.
    """
    logger.info("Beginning paper upload handling")
    agent_loop = request.app["agent_loop"]
    kb = getattr(agent_loop, "kb", None)
    if kb is None:
        return _error_json(400, "Knowledge base not available (paper tools disabled)")

    if not request.content_type.startswith("multipart/"):
        return _error_json(400, "Expected multipart/form-data")
    try:
        reader = await request.multipart()
    except Exception:
        return _error_json(400, "Invalid multipart upload")

    results: list[dict[str, Any]] = []
    uploads: list[_UploadedPaper] = []
    total_size = 0
    file_count = 0
    paper_config = getattr(getattr(agent_loop, "tools_config", None), "paper", None)
    max_file_size = _positive_int_setting(
        paper_config,
        "max_upload_mb",
        MAX_PAPER_UPLOAD_SIZE // (1024 * 1024),
    ) * 1024 * 1024
    max_total_size = _positive_int_setting(
        paper_config,
        "max_upload_total_mb",
        MAX_PAPER_UPLOAD_TOTAL_SIZE // (1024 * 1024),
    ) * 1024 * 1024
    max_files = _positive_int_setting(
        paper_config,
        "max_upload_files",
        MAX_PAPER_UPLOAD_FILES,
    )
    max_pdf_pages = _positive_int_setting(
        paper_config,
        "max_pdf_pages",
        MAX_PDF_PAGES,
    )
    uploads_dir = kb.base_dir / "uploads"
    ingest_locks: dict[str, asyncio.Lock] = request.app["paper_ingest_locks"]

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name != "files":
            continue
        file_count += 1
        if file_count > max_files:
            results.append(_paper_error_result(
                part.filename,
                _PaperUploadError(
                    "TOO_MANY_FILES",
                    f"A maximum of {max_files} PDFs can be uploaded at once",
                ),
            ))
            continue

        try:
            remaining_total = max_total_size - total_size
            if remaining_total <= 0:
                raise _PaperUploadError(
                    "UPLOAD_TOTAL_TOO_LARGE",
                    f"Upload batch exceeds the {max_total_size // (1024 * 1024)}MB limit",
                )
            uploaded = await _stream_paper_upload(
                part,
                uploads_dir,
                max_size=min(max_file_size, remaining_total),
                max_pages=max_pdf_pages,
            )
            total_size += uploaded.size_bytes
            uploads.append(uploaded)
        except _PaperUploadError as exc:
            results.append(_paper_error_result(part.filename, exc))
        except Exception:
            logger.exception("KB ingest failed for {}", part.filename)
            results.append(_paper_error_result(
                part.filename,
                _PaperUploadError(
                    "INGEST_INTERNAL_ERROR",
                    "PDF ingestion failed due to an internal error",
                    stage="ingestion",
                    retryable=True,
                ),
            ))

    if not results and not uploads:
        return _error_json(400, "No PDF files were provided")

    wait_for_completion = request.query.get("wait", "").lower() in {"1", "true", "yes"}
    if wait_for_completion:
        for uploaded in uploads:
            try:
                results.append(await _ingest_uploaded_paper(
                    agent_loop,
                    kb,
                    uploaded,
                    ingest_locks,
                ))
            except Exception:
                logger.exception("PDF ingestion failed for {}", uploaded.original_filename)
                results.append(_paper_error_result(
                    uploaded.original_filename,
                    _PaperUploadError(
                        "INGEST_INTERNAL_ERROR",
                        "PDF ingestion failed due to an internal error",
                        stage="ingestion",
                        retryable=True,
                    ),
                ))
        succeeded = sum(result.get("status") == "ok" for result in results)
        return web.json_response({
            "status": "ok" if succeeded == len(results) else "partial" if succeeded else "error",
            "total": len(results),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
            "results": results,
        })

    if not uploads:
        return web.json_response({
            "status": "error",
            "total": len(results),
            "succeeded": 0,
            "failed": len(results),
            "results": results,
        })
    job_manager: _PaperIngestJobManager = request.app["paper_ingest_jobs"]
    job = job_manager.submit(
        agent_loop=agent_loop,
        kb=kb,
        uploads=uploads,
        ingest_locks=ingest_locks,
        validation_results=results,
    )
    return web.json_response(job, status=202)


async def handle_paper_ingest_job(request: web.Request) -> web.Response:
    """GET /api/papers/jobs/{job_id} — return bounded background job state."""
    manager: _PaperIngestJobManager = request.app["paper_ingest_jobs"]
    job = manager.get(request.match_info["job_id"])
    if job is None:
        return _error_json(404, "Paper ingestion job was not found")
    return web.json_response(job)


async def handle_paper_delete(request: web.Request) -> web.Response:
    """DELETE /api/papers/{paper_id} — remove one paper from every KB backend."""
    agent_loop = request.app["agent_loop"]
    kb = getattr(agent_loop, "kb", None)
    if kb is None:
        return _error_json(400, "Knowledge base not available (paper tools disabled)")

    paper_id = str(request.match_info.get("paper_id", "")).strip()
    if not paper_id:
        return _error_json(400, "paper_id is required", err_type="invalid_request_error")

    ingest_locks: dict[str, asyncio.Lock] = request.app["paper_ingest_locks"]
    ingest_lock = ingest_locks.setdefault(paper_id, asyncio.Lock())
    try:
        async with ingest_lock:
            result = await kb.delete_paper(paper_id)
    except ValueError as exc:
        return _error_json(400, str(exc), err_type="invalid_request_error")
    except Exception:
        logger.exception("Failed to delete paper {}", paper_id)
        return _error_json(
            500,
            "Paper deletion failed; storage snapshots were restored where possible",
            err_type="server_error",
        )

    if not result.get("deleted"):
        return _error_json(404, "Paper was not found", err_type="not_found_error")
    return web.json_response(result)


async def handle_kb_stats(request: web.Request) -> web.Response:
    """GET /api/kb/stats — return knowledge base statistics."""
    agent_loop = request.app["agent_loop"]
    kb = getattr(agent_loop, "kb", None)
    if kb is None:
        return web.json_response({
            "paper_count": 0,
            "chunk_count": 0,
            "chroma_chunk_count": None,
            "lexical_chunk_count": None,
            "storage_backend": "disabled",
            "chroma_consistent": None,
            "lexical_consistent": None,
            "backends_consistent": None,
            "embedding": {
                "backend": "disabled",
                "model": "",
                "batch_size": 0,
                "degraded": True,
                "reason": "paper_kb_disabled",
            },
            "lexical": {
                "backend": "disabled",
                "document_count": None,
                "degraded": True,
                "reason": "paper_kb_disabled",
            },
            "degraded": True,
            "degradation_reasons": ["paper_kb_disabled"],
            "recent_papers": [],
        })

    return web.json_response(kb.get_stats())


# ---------------------------------------------------------------------------
# CORS middleware (local-only tool — allow cross-origin from WebUI gateway)
# ---------------------------------------------------------------------------


@web.middleware
async def _cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Protect KB routes and add CORS only for explicitly trusted origins."""
    origin = request.headers.get("Origin", "")
    allowed_origins = request.app.get("allowed_origins", set())
    origin_allowed = _origin_is_allowed(origin, allowed_origins)
    if origin and not origin_allowed:
        return _error_json(403, "Origin is not allowed", err_type="forbidden")

    if request.path.startswith("/api/") and request.method != "OPTIONS":
        token_validator = request.app.get("kb_token_validator")
        if token_validator is not None:
            token = _request_bearer_token(request)
            try:
                authorized = bool(token and token_validator(token))
            except Exception:
                logger.exception("KB API token validation failed")
                authorized = False
            if not authorized:
                return _error_json(401, "Unauthorized", err_type="authentication_error")

    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    if origin and origin_allowed:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    resp.headers["Access-Control-Max-Age"] = "3600"
    return resp


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop,
    model_name: str = "nanobot",
    request_timeout: float = 120.0,
    *,
    kb_token_validator: Any | None = None,
    allowed_origins: set[str] | None = None,
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    paper_config = getattr(getattr(agent_loop, "tools_config", None), "paper", None)
    max_upload_total = _positive_int_setting(
        paper_config,
        "max_upload_total_mb",
        MAX_PAPER_UPLOAD_TOTAL_SIZE // (1024 * 1024),
    ) * 1024 * 1024
    app = web.Application(
        client_max_size=max_upload_total + 1024 * 1024,
        middlewares=[_cors_middleware],
    )
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key
    app["paper_ingest_locks"] = {}
    app["paper_ingest_jobs"] = _PaperIngestJobManager()
    app["kb_token_validator"] = kb_token_validator
    app["allowed_origins"] = allowed_origins or set()

    # OpenAI-compatible API
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)

    # Knowledge Base management
    app.router.add_post("/api/papers/upload", handle_papers_upload)
    app.router.add_get("/api/papers/jobs/{job_id}", handle_paper_ingest_job)
    app.router.add_delete(r"/api/papers/{paper_id:.+}", handle_paper_delete)
    app.router.add_get("/api/kb/stats", handle_kb_stats)

    async def _close_ingest_jobs(current_app: web.Application) -> None:
        await current_app["paper_ingest_jobs"].close()

    app.on_cleanup.append(_close_ingest_jobs)

    return app
