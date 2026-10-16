"""HTTP/local search client and run_code-shaped runner; no training imports.

Use local:<SQLite index path> or an HTTP(S) service base URL. Network, schema,
index identity and local SQLite failures raise SearchServiceError, never a
successful empty retrieval. Model-generated invalid/empty queries return [].
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from urllib import error, parse, request

MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_QUERY_CHARS = 1024
MAX_BATCH = 64
MAX_TOP_K = 20


class SearchServiceError(RuntimeError):
    """Infrastructure failure: callers must stop/retry training, not assign zero reward."""


def _positive_seconds(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("search timeout must be a finite positive number")
    return value


def _top_k(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_TOP_K:
        raise ValueError("search_top_k must be an integer between 1 and %d" % MAX_TOP_K)
    return value


def _query(value: object) -> str:
    # Invalid model arguments are normal task data, not infrastructure errors.
    if not isinstance(value, str) or len(value) > MAX_QUERY_CHARS:
        return ""
    return value


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an optional bearer credential to another server.
        return None


class SearchClient:
    def __init__(self, endpoint: str, *, timeout: float = 5.0, top_k: int = 3,
                 retries: int = 2, auth_env: str | None = None):
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError("search_endpoint is required")
        self.timeout = _positive_seconds(timeout)
        self.top_k = _top_k(top_k)
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
            raise ValueError("search_retries must be an integer from 0 to 10 (additional attempts)")
        self.retries = retries
        self.corpus_id = None
        self._identity_lock = threading.RLock()
        self._index = None
        self._token = os.environ.get(auth_env) if auth_env else None
        if auth_env and not self._token:
            raise ValueError("authentication environment variable is empty")
        if endpoint.startswith("local:"):
            path = endpoint[len("local:"):]
            if not path:
                raise ValueError("local endpoint requires a SQLite index path")
            # Lazy: the HTTP client never loads the server or any training dependency.
            from .search_server import SearchIndex
            try:
                self._index = SearchIndex(path)
            except (OSError, ValueError, RuntimeError, __import__("sqlite3").Error) as exc:
                raise SearchServiceError("local search index unavailable") from exc
            self.endpoint = endpoint
            self._opener = None
        else:
            parts = parse.urlsplit(endpoint)
            if (parts.scheme not in ("http", "https") or not parts.hostname
                    or parts.username or parts.password or parts.query or parts.fragment):
                raise ValueError("search_endpoint must be a base HTTP(S) URL without credentials/query/fragment")
            self.endpoint = endpoint.rstrip("/")
            self._opener = request.build_opener(_NoRedirect())

    def _pin(self, payload: dict):
        corpus_id = payload.get("corpus_id")
        if not isinstance(corpus_id, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_id):
            raise SearchServiceError("search response has invalid corpus_id")
        with self._identity_lock:
            if self.corpus_id is not None and self.corpus_id != corpus_id:
                raise SearchServiceError("search corpus_id changed; refusing mixed-index training")
            self.corpus_id = corpus_id

    def _http(self, route: str, payload: dict | None, timeout: float) -> dict:
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise ValueError("search request exceeds size limit")
        headers = {"Accept": "application/json", "Accept-Encoding": "identity"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self._token is not None:
            headers["Authorization"] = "Bearer " + self._token
        for attempt in range(self.retries + 1):
            try:
                req = request.Request(self.endpoint + route, data=body, headers=headers)
                with self._opener.open(req, timeout=timeout) as response:
                    if response.status != 200:
                        raise SearchServiceError("search service returned non-200 status")
                    if response.headers.get_content_type() != "application/json":
                        raise SearchServiceError("search response is not JSON")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise SearchServiceError("compressed search responses unsupported")
                    lengths = response.headers.get_all("Content-Length", [])
                    if len(lengths) > 1:
                        raise SearchServiceError("ambiguous search response length")
                    length = int(lengths[0]) if lengths else None
                    if length is not None and not 0 <= length <= MAX_RESPONSE_BYTES:
                        raise SearchServiceError("search response exceeds size limit")
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise SearchServiceError("search response exceeds size limit")
                    if length is not None and len(raw) != length:
                        raise SearchServiceError("incomplete search response")
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise SearchServiceError("search response must be an object")
                    return value
            except error.HTTPError as exc:
                exc.close()
                # Authentication and other 4xx errors are permanent, not retriable.
                if exc.code < 500 or attempt == self.retries:
                    raise SearchServiceError("search service HTTP status %d" % exc.code) from None
            except (error.URLError, OSError, TimeoutError):
                if attempt == self.retries:
                    raise SearchServiceError("search service network failure after %d attempts" % (attempt + 1)) from None
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise SearchServiceError("invalid search response encoding/JSON") from exc
        raise AssertionError("unreachable retry loop")

    def check_health(self) -> dict:
        if self._index is None:
            value = self._http("/health", None, self.timeout)
        else:
            try:
                value = self._index.health()
            except (OSError, ValueError, RuntimeError, __import__("sqlite3").Error) as exc:
                raise SearchServiceError("local search index unavailable or changed") from exc
        if (value.get("ok") is not True or isinstance(value.get("documents"), bool)
                or not isinstance(value.get("documents"), int) or value["documents"] < 0):
            raise SearchServiceError("invalid search health response")
        self._pin(value)
        return value

    def _ensure_health(self):
        # First health pins identity before querying; explicit check_health is also
        # available as the training startup fail-fast gate.
        with self._identity_lock:
            if self.corpus_id is None:
                self.check_health()

    def _results(self, value: dict, *, batch_size: int | None, top_k: int) -> list:
        self._pin(value)
        results = value.get("results")
        if not isinstance(results, list):
            raise SearchServiceError("search results must be a list")
        if batch_size is not None and len(results) != batch_size:
            raise SearchServiceError("search batch response cardinality mismatch")
        groups = [results] if batch_size is None else results
        for group in groups:
            if not isinstance(group, list) or len(group) > top_k:
                raise SearchServiceError("invalid search result group")
            for row in group:
                if (not isinstance(row, dict) or not isinstance(row.get("title"), str)
                        or not isinstance(row.get("text"), str)):
                    raise SearchServiceError("search results require title/text strings")
        return results

    def search(self, query: object, *, top_k: int | None = None,
               timeout: float | None = None) -> list[dict]:
        self._ensure_health()
        count = self.top_k if top_k is None else _top_k(top_k)
        seconds = self.timeout if timeout is None else _positive_seconds(timeout)
        if self._index is None:
            value = self._http("/search", {"query": _query(query), "top_k": count}, seconds)
        else:
            try:
                value = self._index.search(_query(query), count)
            except (OSError, ValueError, RuntimeError, __import__("sqlite3").Error) as exc:
                raise SearchServiceError("local search query failed or index changed") from exc
        return self._results(value, batch_size=None, top_k=count)

    def search_batch(self, queries: list, *, top_k: int | None = None,
                     timeout: float | None = None) -> list[list[dict]]:
        if not isinstance(queries, list) or len(queries) > MAX_BATCH:
            raise ValueError("queries must be a list with at most %d entries" % MAX_BATCH)
        self._ensure_health()
        count = self.top_k if top_k is None else _top_k(top_k)
        seconds = self.timeout if timeout is None else _positive_seconds(timeout)
        clean = [_query(query) for query in queries]
        if self._index is None:
            value = self._http("/search/batch", {"queries": clean, "top_k": count}, seconds)
        else:
            try:
                value = self._index.search_batch(clean, count)
            except (OSError, ValueError, RuntimeError, __import__("sqlite3").Error) as exc:
                raise SearchServiceError("local search query failed or index changed") from exc
        return self._results(value, batch_size=len(queries), top_k=count)


class SearchRunner:
    """run_code-compatible callable; max_chars truncates tool display only.

    mem_mb is accepted for executor compatibility, not a remote memory quota.
    Infrastructure exceptions intentionally propagate through both execution paths.
    """
    def __init__(self, client: SearchClient):
        self.client = client

    @staticmethod
    def _display(results: list[dict], max_chars: int) -> str:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
            raise ValueError("max_chars must be a nonnegative integer")
        text = "\n\n".join("[%d] %s\n%s" % (i + 1, row["title"], row["text"])
                           for i, row in enumerate(results)) or "No search results."
        return text[:max_chars]

    def __call__(self, query: object, timeout: float | None = None, mem_mb: int = 256,
                 max_chars: int = 500) -> dict:
        start = time.monotonic()
        results = self.client.search(query, timeout=timeout)
        return {"ok": True, "display": self._display(results, max_chars),
                "duration": time.monotonic() - start, "error_type": "ok"}

    def run_batch(self, queries: list, max_chars: int = 500) -> list[dict]:
        start = time.monotonic()
        groups = self.client.search_batch(queries)
        duration = time.monotonic() - start
        return [{"ok": True, "display": self._display(rows, max_chars),
                 "duration": duration, "error_type": "ok"} for rows in groups]

    def check_health(self) -> dict:
        return self.client.check_health()


def make_search_runner(cfg: dict) -> SearchRunner:
    """Read search_endpoint/timeout/top_k/retries and optional search_auth_env."""
    return SearchRunner(SearchClient(
        cfg["search_endpoint"], timeout=cfg.get("search_timeout", 5.0),
        top_k=cfg.get("search_top_k", 3), retries=cfg.get("search_retries", 2),
        auth_env=cfg.get("search_auth_env")))
