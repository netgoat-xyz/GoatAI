"""Threaded local HTTP target for Koda-WAF and GoTestWAF."""

from __future__ import annotations

import argparse
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from engine import KodaWAF, RequestView

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "koda-waf.pkl"
MAX_BODY_BYTES = 2 * 1024 * 1024


class KodaWAFHandler(BaseHTTPRequestHandler):
    engine: KodaWAF | None = None
    server_version = "KodaWAF/1.0"

    def _body(self) -> str:
        try:
            length = min(max(int(self.headers.get("content-length", "0")), 0), MAX_BODY_BYTES)
        except ValueError:
            length = 0
        return self.rfile.read(length).decode("utf-8", errors="replace") if length else ""

    def _handle(self) -> None:
        started = time.perf_counter()
        parsed = urlsplit(self.path)
        headers = " ".join(f"{key}:{value}" for key, value in self.headers.items())
        if self.engine is None:
            raise RuntimeError("Koda-WAF model is not loaded")
        decision = self.engine.inspect(RequestView(
            method=self.command,
            path=parsed.path or "/",
            query=parsed.query,
            headers=headers,
            body=self._body(),
        ))
        status = HTTPStatus.FORBIDDEN if decision.blocked else HTTPStatus.OK
        response = {
            "blocked": decision.blocked,
            "score": round(decision.score, 4),
            "reasons": decision.reasons,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        body = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("x-koda-waf", "block" if decision.blocked else "allow")
        self.send_header("x-koda-waf-score", f"{decision.score:.4f}")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = _handle

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dependency-free Koda-WAF HTTP service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    KodaWAFHandler.engine = KodaWAF.from_model(args.model)
    server = ThreadingHTTPServer((args.host, args.port), KodaWAFHandler)
    print(f"Koda-WAF listening on http://{args.host}:{args.port}", flush=True)
    print("Malicious requests return 403; benign requests return 200.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
