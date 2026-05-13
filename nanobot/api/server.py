"""OpenAI-compatible HTTP API server for a fixed nanobot session.

Provides /v1/chat/completions and /v1/models endpoints.
All requests route to a single persistent API session.
"""

from __future__ import annotations

import asyncio
import base64
import json as _json
import mimetypes
import re
import time
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web
from loguru import logger

from datetime import datetime

from nanobot.config.paths import get_media_dir
from nanobot.utils.helpers import safe_filename
from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PAPER_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB per PDF
_DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.+)$", re.DOTALL)


class _FileSizeExceeded(Exception):
    """Raised when an uploaded file exceeds the size limit."""


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
            text = (await part.read()).decode("utf-8")
        elif part.name == "session_id":
            session_id = (await part.read()).decode("utf-8").strip()
        elif part.name == "model":
            model = (await part.read()).decode("utf-8").strip()
        elif part.name == "files":
            raw = await part.read()
            if len(raw) > MAX_FILE_SIZE:
                raise _FileSizeExceeded(
                    f"File '{part.filename}' exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit"
                )
            base = safe_filename(part.filename or "upload.bin")
            filename = f"{uuid.uuid4().hex[:12]}_{base}"
            dest = media_dir / filename
            dest.write_bytes(raw)
            media_paths.append(str(dest))

    if not text:
        text = "请分析上传的文件"

    return text, media_paths, session_id, model


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
                body = await request.json()
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
    logger.info("Beginning paper upload handling")  # Debug log to trace request handling
    agent_loop = request.app["agent_loop"]
    kb = getattr(agent_loop, "kb", None)
    if kb is None:
        return _error_json(400, "Knowledge base not available (paper tools disabled)")

    reader = await request.multipart()
    results: list[dict[str, Any]] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name != "files":
            continue

        raw = await part.read()
        if len(raw) > MAX_PAPER_UPLOAD_SIZE:
            results.append({
                "filename": part.filename,
                "status": "error",
                "error": f"File exceeds {MAX_PAPER_UPLOAD_SIZE // (1024 * 1024)}MB limit",
            })
            continue

        # 1. Save to workspace/kb/uploads/
        uploads_dir = kb.base_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        filename = safe_filename(part.filename or f"paper_{uuid.uuid4().hex[:8]}.pdf")
        local_path = uploads_dir / filename
        local_path.write_bytes(raw)
        local_md_path = local_path.with_suffix(".md")

        # 2. Build paper metadata (title will be extracted from markdown heading)
        stem = (Path(part.filename or "paper").stem)[:200]
        import hashlib
        paper_id = hashlib.md5(stem.encode()).hexdigest()[:12]
        doc = {
            "paper_id": paper_id,
            "title": Path(part.filename or "paper").stem,
            "source": "upload",
            "url": str(local_md_path),
            "year": datetime.now().year,
        }

        # 3. Ingest via AgentLoop helper
        try:
            ingest_result = await agent_loop.kb_ingest_local(
                doc=doc,
                local_pdf_path=str(local_path),
            )
            results.append({
                "filename": part.filename,
                "paper_id": doc["paper_id"],
                "title": doc.get("title", ""),
                "status": ingest_result.get("status", "ok"),
                "chunk_count": ingest_result.get("chunk_count", 0),
            })
        except Exception as exc:
            logger.exception("KB ingest failed for {}", part.filename)
            results.append({
                "filename": part.filename,
                "status": "error",
                "error": str(exc),
            })

    return web.json_response({"results": results})


async def handle_kb_stats(request: web.Request) -> web.Response:
    """GET /api/kb/stats — return knowledge base statistics."""
    agent_loop = request.app["agent_loop"]
    kb = getattr(agent_loop, "kb", None)
    if kb is None:
        return web.json_response({"paper_count": 0, "chunk_count": 0, "recent_papers": []})

    # Count from JSONL files
    docs = kb._read_jsonl(kb.docs_file)
    chunks = kb._read_jsonl(kb.chunks_file)

    # Count unique papers per doc
    paper_ids = {d.get("paper_id") for d in docs if d.get("paper_id")}
    chunk_count = len(chunks)

    # Recent papers (last 10)
    recent = sorted(docs, key=lambda d: d.get("updated_at", ""), reverse=True)[:10]
    recent_papers = [
        {
            "paper_id": d.get("paper_id", ""),
            "title": d.get("title", ""),
            "source": d.get("source", ""),
            "year": d.get("year"),
            "chunk_count": sum(1 for c in chunks if c.get("paper_id") == d.get("paper_id")),
            "updated_at": d.get("updated_at", ""),
        }
        for d in recent
    ]

    return web.json_response({
        "paper_count": len(paper_ids),
        "chunk_count": chunk_count,
        "recent_papers": recent_papers,
    })


# ---------------------------------------------------------------------------
# CORS middleware (local-only tool — allow cross-origin from WebUI gateway)
# ---------------------------------------------------------------------------


@web.middleware
async def _cors_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    """Add permissive CORS headers for the embedded WebUI."""
    if request.method == "OPTIONS":
        # Preflight
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    resp.headers["Access-Control-Max-Age"] = "3600"
    return resp


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    agent_loop, model_name: str = "nanobot", request_timeout: float = 120.0
) -> web.Application:
    """Create the aiohttp application.

    Args:
        agent_loop: An initialized AgentLoop instance.
        model_name: Model name reported in responses.
        request_timeout: Per-request timeout in seconds.
    """
    app = web.Application(
        client_max_size=20 * 1024 * 1024,  # 20MB for base64 images
        middlewares=[_cors_middleware],
    )
    app["agent_loop"] = agent_loop
    app["model_name"] = model_name
    app["request_timeout"] = request_timeout
    app["session_locks"] = {}  # per-user locks, keyed by session_key

    # OpenAI-compatible API
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_get("/health", handle_health)

    # Knowledge Base management
    app.router.add_post("/api/papers/upload", handle_papers_upload)
    app.router.add_get("/api/kb/stats", handle_kb_stats)

    return app
