"""Disk-backed SQLite FTS5 search service (standard library only).

CLI: python rlab/search_server.py build-index CORPUS.jsonl INDEX.sqlite
     python rlab/search_server.py serve INDEX.sqlite --host 0.0.0.0 --port 8000
Optional authentication: --auth-env SEARCH_BEARER_TOKEN (value stays in env).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_QUERY_CHARS = 1024
MAX_BATCH = 64
MAX_TOP_K = 20
MAX_CORPUS_LINE_BYTES = 8 * 1024 * 1024
QUERY_SECONDS = 10.0


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_index(corpus_path: str | os.PathLike, index_path: str | os.PathLike) -> str:
    """Stream {id,title,text} JSONL into a new FTS5 database; never overwrite.

    Return a SHA256 corpus_id over canonical records in corpus order. On failure,
    remove only the newly created partial database, not an existing target.
    """
    target = Path(index_path).resolve()
    # Exclusive creation, including an existing empty file, precedes SQLite open.
    fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    connection = None
    try:
        connection = sqlite3.connect(str(target))
        connection.execute("CREATE VIRTUAL TABLE documents USING fts5(id UNINDEXED, title, text, tokenize='unicode61')")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        fingerprint = hashlib.sha256(b"rlab-search-fts5-unicode61-v1\n")
        count = 0
        with open(corpus_path, "rb") as corpus:
            while True:
                raw = corpus.readline(MAX_CORPUS_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_CORPUS_LINE_BYTES:
                    raise ValueError("corpus line exceeds size limit")
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if (not isinstance(row, dict) or isinstance(row.get("id"), bool)
                        or not isinstance(row.get("id"), (str, int))
                        or not isinstance(row.get("title"), str)
                        or not isinstance(row.get("text"), str)):
                    raise ValueError("corpus records require id (string/int), title and text (strings)")
                record = {"id": str(row["id"]), "title": row["title"], "text": row["text"]}
                fingerprint.update(_json_bytes(record) + b"\n")
                connection.execute("INSERT INTO documents(id,title,text) VALUES(?,?,?)",
                                   (record["id"], record["title"], record["text"]))
                count += 1
                if count % 1000 == 0:
                    connection.commit()
        corpus_id = fingerprint.hexdigest()
        connection.executemany("INSERT INTO metadata VALUES(?,?)",
                               [("corpus_id", corpus_id), ("documents", str(count))])
        connection.execute("INSERT INTO documents(documents) VALUES('optimize')")
        connection.commit()
        connection.close()
        connection = None
        return corpus_id
    except BaseException:
        if connection is not None:
            connection.close()
        # target is the resolved path exclusively created above, never a preexisting file.
        target.unlink()
        raise


def _top_k(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOP_K:
        raise ValueError("top_k must be an integer between 1 and %d" % MAX_TOP_K)
    return value


def _literal_query(query: object) -> str:
    """Model queries are words, not FTS programs: quoted Unicode OR terms only."""
    if not isinstance(query, str) or len(query) > MAX_QUERY_CHARS:
        return ""
    # Normalize compatibility forms; Unicode-aware word extraction excludes FTS syntax.
    words = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", query), re.UNICODE)
    words = list(dict.fromkeys(words))[:64]
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in words)


def _bounded_result(title: str, text: str, budget: int) -> dict:
    result = {"title": title[:512], "text": text[:8192]}
    # Byte budget accounts for JSON escaping, not merely Python character count.
    while len(_json_bytes(result)) > budget:
        field = "text" if result["text"] else "title"
        result[field] = result[field][:len(result[field]) // 2]
    return result


class SearchIndex:
    """Thread-safe by using a fresh read-only connection per operation.

    The database corpus_id is pinned at construction and checked on every open;
    replacing the index beneath a running service raises instead of mixing corpora.
    """
    def __init__(self, index_path: str | os.PathLike):
        self.path = Path(index_path).resolve()
        self.corpus_id = None
        with self._connect() as connection:
            self.corpus_id = self._identity(connection)

    @staticmethod
    def _identity(connection: sqlite3.Connection) -> str:
        row = connection.execute("SELECT value FROM metadata WHERE key='corpus_id'").fetchone()
        if not row or not re.fullmatch(r"[0-9a-f]{64}", row[0]):
            raise ValueError("invalid search index corpus_id")
        return row[0]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            if self.corpus_id is not None and self._identity(connection) != self.corpus_id:
                raise RuntimeError("search index corpus_id changed")
        except BaseException:
            connection.close()
            raise
        return connection

    def health(self) -> dict:
        connection = self._connect()
        try:
            count = int(connection.execute("SELECT value FROM metadata WHERE key='documents'").fetchone()[0])
            return {"ok": True, "corpus_id": self.corpus_id, "documents": count}
        finally:
            connection.close()

    def search(self, query: object, top_k: int = 3) -> dict:
        payload = self.search_batch([query], top_k)
        return {"corpus_id": payload["corpus_id"], "results": payload["results"][0]}

    def search_batch(self, queries: list, top_k: int = 3) -> dict:
        top_k = _top_k(top_k)
        if not isinstance(queries, list) or len(queries) > MAX_BATCH:
            raise ValueError("queries must be a list with at most %d entries" % MAX_BATCH)
        connection = self._connect()
        try:
            deadline = time.monotonic() + QUERY_SECONDS
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            budget = (MAX_RESPONSE_BYTES - 4096) // (max(1, len(queries)) * top_k)
            groups = []
            for query in queries:
                expression = _literal_query(query)
                if not expression:
                    groups.append([])
                    continue
                rows = connection.execute(
                    "SELECT title,text FROM documents WHERE documents MATCH ? "
                    "ORDER BY bm25(documents),rowid LIMIT ?", (expression, top_k))
                groups.append([_bounded_result(title, text, budget) for title, text in rows])
            return {"corpus_id": self.corpus_id, "results": groups}
        finally:
            connection.close()


def make_server(index_path: str | os.PathLike, host: str = "127.0.0.1", port: int = 8000,
                *, auth_env: str | None = None) -> ThreadingHTTPServer:
    """Construct, but do not start, the server; port=0 supports integration tests."""
    index = SearchIndex(index_path)
    token = os.environ.get(auth_env) if auth_env else None
    if auth_env and not token:
        raise ValueError("authentication environment variable is empty")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(QUERY_SECONDS)

        def log_message(self, format, *args):
            # No model queries, request headers or credentials in server logs.
            pass

        def _reply(self, status: int, payload: dict):
            body = _json_bytes(payload)
            if len(body) > MAX_RESPONSE_BYTES:
                status, body = 503, _json_bytes({"error": "response size limit"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _authorized(self) -> bool:
            if token is None:
                return True
            headers = self.headers.get_all("Authorization", [])
            allowed = len(headers) == 1 and hmac.compare_digest(
                headers[0].encode("utf-8"), ("Bearer " + token).encode("utf-8"))
            if not allowed:
                self._reply(401, {"error": "unauthorized"})
            return allowed

        def do_GET(self):
            if not self._authorized():
                return
            if self.path != "/health":
                self._reply(404, {"error": "not found"})
                return
            try:
                self._reply(200, index.health())
            except (sqlite3.Error, ValueError, RuntimeError, OSError):
                self._reply(503, {"error": "search index unavailable or changed"})

        def do_POST(self):
            if not self._authorized():
                return
            if self.path not in ("/search", "/search/batch"):
                self._reply(404, {"error": "not found"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
                self._reply(400, {"error": "one Content-Length required; chunked requests unsupported"})
                return
            try:
                length = int(lengths[0])
            except ValueError:
                self._reply(400, {"error": "invalid Content-Length"})
                return
            if not 0 <= length <= MAX_REQUEST_BYTES:
                self._reply(413, {"error": "request size limit"})
                return
            if self.headers.get_content_type() != "application/json":
                self._reply(415, {"error": "application/json required"})
                return
            try:
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete request")
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                top_k = _top_k(request.get("top_k", 3))
                if self.path == "/search":
                    result = index.search(request.get("query"), top_k)
                else:
                    result = index.search_batch(request.get("queries"), top_k)
                self._reply(200, result)
            except (ValueError, UnicodeError, RecursionError):
                self._reply(400, {"error": "invalid request"})
            except (sqlite3.Error, RuntimeError, OSError):
                self._reply(503, {"error": "search index unavailable or changed"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-index")
    build.add_argument("corpus_path")
    build.add_argument("index_path")
    serve = commands.add_parser("serve")
    serve.add_argument("index_path")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--auth-env", default=None)
    args = parser.parse_args(argv)
    if args.command == "build-index":
        print(json.dumps({"corpus_id": build_index(args.corpus_path, args.index_path)}))
        return
    server = make_server(args.index_path, args.host, args.port, auth_env=args.auth_env)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
