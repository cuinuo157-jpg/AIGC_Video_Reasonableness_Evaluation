from __future__ import annotations

import argparse
import cgi
import json
import logging
import mimetypes
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .service import (
    DEFAULT_UPLOAD_DIR,
    DEFAULT_RESULTS_DIR,
    WebUIJobManager,
    build_frontend_config,
    build_run_config,
    save_uploaded_file,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class WebUIHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        *,
        upload_dir: Path | None = None,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.upload_dir = upload_dir or DEFAULT_UPLOAD_DIR
        self.frontend_config = build_frontend_config()
        self.job_manager = WebUIJobManager(results_dir=DEFAULT_RESULTS_DIR)


class WebUIRequestHandler(BaseHTTPRequestHandler):
    server_version = "AIGCVideoWebUI/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html", content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return
        if parsed.path == "/api/config":
            self._send_json(self.server.frontend_config)
            return
        if parsed.path.startswith("/api/jobs/"):
            self._handle_job_get(parsed)
            return
        if parsed.path.startswith("/static/"):
            target = (STATIC_DIR / parsed.path.removeprefix("/static/")).resolve()
            if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
                self._send_error_json(HTTPStatus.FORBIDDEN, "invalid static path")
                return
            self._serve_file(target)
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "route not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/evaluate":
            self._handle_evaluate()
            return
        self._send_error_json(HTTPStatus.NOT_FOUND, "route not found")

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _handle_evaluate(self) -> None:
        try:
            payload, uploaded_video_path = self._parse_request_payload()
            run_config = build_run_config(payload, uploaded_video_path=uploaded_video_path)
            job = self.server.job_manager.create_job(run_config)
            self._send_json(
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "video_name": Path(run_config.video_path).name,
                },
                status=HTTPStatus.ACCEPTED,
            )
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("webui evaluate failed")
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                str(exc),
                details=traceback.format_exc(limit=8),
            )

    def _handle_job_get(self, parsed: Any) -> None:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3:
            self._send_error_json(HTTPStatus.NOT_FOUND, "job route not found")
            return
        _, _, job_id, *rest = parts
        try:
            if not rest:
                self._send_json(self.server.job_manager.get_job_snapshot(job_id))
                return
            if len(rest) == 1 and rest[0] == "logs":
                query = parse_qs(parsed.query, keep_blank_values=True)
                offset = int(query.get("offset", ["0"])[0] or 0)
                self._send_json(self.server.job_manager.get_job_logs(job_id, offset=offset))
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "job subroute not found")
        except KeyError:
            self._send_error_json(HTTPStatus.NOT_FOUND, f"job not found: {job_id}")
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def _parse_request_payload(self) -> tuple[dict[str, object], str | None]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" in content_type:
            return self._parse_multipart_payload()
        if "application/json" in content_type:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            return json.loads(raw.decode("utf-8") or "{}"), None
        if "application/x-www-form-urlencoded" in content_type:
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(content_length).decode("utf-8")
            form = parse_qs(raw, keep_blank_values=True)
            payload = {
                key: value if len(value) > 1 else value[0]
                for key, value in form.items()
            }
            return payload, None
        raise ValueError(f"unsupported content type: {content_type or 'unknown'}")

    def _parse_multipart_payload(self) -> tuple[dict[str, object], str | None]:
        environ = {
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
        }
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ=environ,
            keep_blank_values=True,
        )

        payload: dict[str, object] = {}
        uploaded_video_path: str | None = None
        if not getattr(form, "list", None):
            return payload, None

        multi_value_fields = {"anomaly_types", "selected_dimensions"}
        for item in form.list:
            if item.filename and item.name == "video_file":
                if item.file and item.filename:
                    uploaded_video_path = save_uploaded_file(
                        item.file,
                        item.filename,
                        upload_dir=self.server.upload_dir,
                    )
                continue

            value = item.value
            if item.name in multi_value_fields:
                payload.setdefault(item.name, [])
                assert isinstance(payload[item.name], list)
                payload[item.name].append(value)
            else:
                payload[item.name] = value
        return payload, uploaded_video_path

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_error_json(HTTPStatus.NOT_FOUND, "file not found")
            return
        guessed, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or guessed or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(
        self,
        status: HTTPStatus,
        message: str,
        *,
        details: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "error": message,
            "status": int(status),
        }
        if details:
            payload["details"] = details
        self._send_json(payload, status=status)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    upload_dir: Path | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    server = WebUIHTTPServer((host, port), WebUIRequestHandler, upload_dir=upload_dir)
    logger.info("Web UI running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Web UI")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AIGC 视频合理性评测 Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument(
        "--upload-dir",
        default=str(DEFAULT_UPLOAD_DIR),
        help="上传视频暂存目录",
    )
    args = parser.parse_args()
    run_server(args.host, args.port, upload_dir=Path(args.upload_dir))


if __name__ == "__main__":
    main()
