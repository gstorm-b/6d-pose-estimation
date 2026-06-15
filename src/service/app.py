"""Stdlib HTTP layer for the pose-estimation backend.

Zero third-party dependencies (no fastapi/uvicorn): a ThreadingHTTPServer with a
small admission gate in front of the runtime's single GPU lock. Requests beyond
`max_queue_depth` in flight are rejected with 429 instead of piling up on the
lock. The handlers are thin — all inference logic lives in ServiceRuntime.

Endpoints:
  POST /v1/poses   -> ranked object poses for one scene
  GET  /v1/skus    -> available SKUs and versions
  GET  /v1/health  -> liveness + loaded bundles + device
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.service.runtime import ServiceRuntime
from src.service.schemas import ErrorCode

# Reject bodies larger than this to bound memory from a hostile Content-Length.
MAX_CONTENT_LENGTH = 256 * 1024 * 1024


class ServiceState:
    """Shared, thread-safe state: the runtime plus an admission counter."""

    def __init__(self, runtime: ServiceRuntime, *, max_queue_depth: int = 8) -> None:
        self.runtime = runtime
        self.max_queue_depth = max(1, int(max_queue_depth))
        self._inflight = 0
        self._lock = threading.Lock()

    def try_admit(self) -> bool:
        with self._lock:
            if self._inflight >= self.max_queue_depth:
                return False
            self._inflight += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight


class PoseHandler(BaseHTTPRequestHandler):
    """Routes the three endpoints; delegates all work to ServiceState/runtime."""

    protocol_version = "HTTP/1.1"
    server_version = "PoseBackend/1.0"

    def __init__(self, state: ServiceState, *args: Any, **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    # -- helpers ---------------------------------------------------------------
    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    # -- routes ----------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        try:
            if self.path == "/v1/health":
                self._send_json(200, self.state.runtime.health())
            elif self.path == "/v1/skus":
                self._send_json(200, self.state.runtime.list_skus())
            else:
                self._error(404, ErrorCode.INVALID_INPUT, f"unknown path: {self.path}")
        except Exception as exc:  # noqa: BLE001 - never leak a stack trace to the client
            self._error(500, ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/v1/poses":
            self._error(404, ErrorCode.INVALID_INPUT, f"unknown path: {self.path}")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._error(400, ErrorCode.INVALID_INPUT, "invalid Content-Length")
            return
        if length <= 0:
            self._error(400, ErrorCode.INVALID_INPUT, "empty request body")
            return
        if length > MAX_CONTENT_LENGTH:
            self._error(413, ErrorCode.INVALID_INPUT, "request body too large")
            return

        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._error(400, ErrorCode.INVALID_INPUT, f"invalid JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self._error(400, ErrorCode.INVALID_INPUT, "body must be a JSON object")
            return

        if not self.state.try_admit():
            self._error(429, ErrorCode.INTERNAL, "service overloaded; retry later")
            return
        try:
            status, body = self.state.runtime.infer(payload)
        except Exception as exc:  # noqa: BLE001 - structured 500, never a stack trace
            status, body = 500, {"error": {"code": ErrorCode.INTERNAL, "message": f"{type(exc).__name__}: {exc}"}}
        finally:
            self.state.release()
        self._send_json(status, body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - http.server API
        # Quiet by default; the launcher can attach its own logging if desired.
        return


def build_server(
    runtime: ServiceRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_queue_depth: int = 8,
) -> tuple[ThreadingHTTPServer, ServiceState]:
    """Construct (but do not start) the threaded HTTP server and its state."""

    state = ServiceState(runtime, max_queue_depth=max_queue_depth)
    handler = partial(PoseHandler, state)
    server = ThreadingHTTPServer((host, port), handler)
    return server, state
