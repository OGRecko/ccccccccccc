"""A fake Ollama server, so the brain can be tested without a GPU or a model.

It implements just enough of the API used by GARVIS:

* ``GET  /api/tags``    - model list
* ``GET  /api/version`` - version string
* ``POST /api/chat``    - streaming NDJSON chat, with scripted replies

Usage in a test::

    server = MockOllama(models=["llama3.1:8b"])
    server.queue_text(["Hello ", "Boss."])
    cfg.set("brain.host", server.url)

Scripting helpers:
    ``queue_text(chunks)``            - stream text, token by token
    ``queue_tool_call(name, args)``   - emit a tool call, then wait for the next scripted reply
    ``queue_raw(chunks)``             - emit literal chunk dicts
    ``queue_error(message)``          - emit an error chunk
    ``requests``                      - everything received, for assertions
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def text_chunks(text: str, size: int = 4) -> list[dict[str, Any]]:
    """Split a reply into small NDJSON chunks, the way a real model streams."""
    chunks = []
    for i in range(0, len(text), size):
        chunks.append({"message": {"role": "assistant", "content": text[i : i + size]}, "done": False})
    return chunks


def tool_call_chunks(name: str, args: dict[str, Any], preamble: str = "") -> list[dict[str, Any]]:
    """A tool call the way Ollama reports it: one chunk with empty content."""
    chunks = []
    if preamble:
        chunks.extend(text_chunks(preamble))
    chunks.append(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": name, "arguments": args}}],
            },
            "done": False,
        }
    )
    return chunks


class MockOllama:
    """Threaded fake server. Use as a context manager."""

    def __init__(
        self,
        models: list[str] | None = None,
        version: str = "0.4.0-mock",
        responder: "Callable[[dict[str, Any]], list[dict[str, Any]]] | None" = None,
    ) -> None:
        self.models = models or ["llama3.1:8b", "llava:7b"]
        self.version = version
        #: Optional dynamic responder(request_payload) -> chunks. Used by the
        #: standalone demo server; tests use the script queue instead.
        self.responder = responder
        self.script: list[list[dict[str, Any]]] = []
        self.requests: list[dict[str, Any]] = []
        self.fail_next: int = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # -- scripting ---------------------------------------------------------
    def queue_text(self, text: str | list[str], size: int = 4) -> "MockOllama":
        chunks: list[dict[str, Any]] = []
        parts = text if isinstance(text, list) else [text]
        for part in parts:
            chunks.extend(text_chunks(part, size))
        self.script.append(chunks + [{"done": True, "done_reason": "stop", "eval_count": 12}])
        return self

    def queue_tool_call(self, name: str, args: dict[str, Any] | None = None, preamble: str = "") -> "MockOllama":
        self.script.append(tool_call_chunks(name, args or {}, preamble) + [{"done": True}])
        return self

    def queue_raw(self, chunks: list[dict[str, Any]]) -> "MockOllama":
        self.script.append(list(chunks))
        return self

    def queue_error(self, message: str) -> "MockOllama":
        self.script.append([{"error": message}])
        return self

    def fail_requests(self, count: int) -> "MockOllama":
        """Make the next N requests fail at the HTTP layer (connection reset)."""
        self.fail_next = count
        return self

    # -- lifecycle ---------------------------------------------------------
    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, port: int = 0) -> "MockOllama":
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # silence the test output
                pass

            def _json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/tags":
                    self._json({"models": [{"name": m} for m in outer.models]})
                elif self.path == "/api/version":
                    self._json({"version": outer.version})
                else:
                    self._json({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    payload = {}

                with outer._lock:
                    outer.requests.append(payload)
                    if outer.fail_next > 0:
                        outer.fail_next -= 1
                        self.close_connection = True
                        return  # drop the connection: simulates Ollama dying
                    if outer.script:
                        chunks = outer.script.pop(0)
                    elif outer.responder is not None:
                        chunks = outer.responder(payload)
                    else:
                        chunks = [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "(mock: no script queued)",
                                },
                                "done": True,
                            }
                        ]

                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                try:
                    for chunk in chunks:
                        if chunk.get("done") and "done_reason" not in chunk:
                            chunk = dict(chunk)
                            chunk["message"] = chunk.get("message", {"role": "assistant", "content": ""})
                        line = (json.dumps(chunk) + "\n").encode()
                        self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class QuietServer(ThreadingHTTPServer):
            """Dropped connections are part of the test, not an error to print."""

            def handle_error(self, request: Any, client_address: Any) -> None:
                return

        self._server = QuietServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "MockOllama":
        return self.start()

    def __exit__(self, *exc: object) -> bool:
        self.stop()
        return False

    # -- assertions --------------------------------------------------------
    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1] if self.requests else {}

    def last_messages(self) -> list[dict[str, Any]]:
        return list(self.last_request.get("messages", []))

    def last_system_prompt(self) -> str:
        for message in self.last_messages():
            if message.get("role") == "system":
                return str(message.get("content", ""))
        return ""

    def last_tool_names(self) -> list[str]:
        return [
            (t.get("function") or {}).get("name", "")
            for t in (self.last_request.get("tools") or [])
        ]


def _main() -> int:
    """Run a standalone fake Ollama, so the plumbing can be tested without a model.

        python tests/mock_ollama.py --port 11999
        python main.py --ollama http://127.0.0.1:11999 --ask "hello"

    Canned behaviour: on the first call it emits a clock.now tool call (so you can
    watch the gate and the tool loop); once it sees a tool result, it answers.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Fake Ollama server for GARVIS plumbing tests.")
    parser.add_argument("--port", type=int, default=11999)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--no-tools", action="store_true", help="always answer directly")
    args = parser.parse_args()

    def responder(payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = payload.get("messages") or []
        last = messages[-1] if messages else {}
        print(
            f"  <- {payload.get('model')}: {len(messages)} messages, "
            f"last_role={last.get('role')}, tools={'yes' if payload.get('tools') else 'no'}",
            flush=True,
        )
        if last.get("role") == "tool":
            return text_chunks("I ran that tool. All good, Boss.") + [{"done": True, "eval_count": 11}]
        if payload.get("tools") and not args.no_tools:
            return tool_call_chunks("clock.now", {}, preamble="One moment. ") + [{"done": True}]
        return text_chunks(
            "This is the mock model, so I am not really thinking. "
            "The plumbing works: config, prompt, streaming, tool loop and logging."
        ) + [{"done": True, "eval_count": 30}]

    mock = MockOllama(models=[args.model, "llava:7b"], responder=responder)
    mock.start(port=args.port)
    print(f"mock Ollama listening on {mock.url}")
    print(f"try:  python main.py --ollama {mock.url} --ask \"hello\"")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        mock.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
