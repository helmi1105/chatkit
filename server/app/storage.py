# app/storage.py
"""Durable state for a scale-to-zero container.

Everything the tutor knows about a learner (mastery, current notion, pending
quiz), every ChatKit thread, every evidence event and the shared content cache
used to live in process memory or on the container disk. On Scaleway
Serverless Containers (min_scale=0) that state disappears at the first idle
period: a trainer coming back the next day restarted from zero (cahier
2026-09-15). This module is a tiny key/value layer over an S3-compatible
bucket (Scaleway Object Storage) with a local-directory fallback for
development.

Keys are plain paths: sessions/<user>.json, threads/<user>.json,
events/<yyyy-mm-dd>/<ts>-<uuid>.json, cache/<...>.json, bank/questions.json.
Writes are synchronous boto3 calls run in a worker thread."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = Path(os.getenv("STATE_LOCAL_DIR", str(APP_DIR / "data")))

S3_BUCKET = os.getenv("STATE_S3_BUCKET", "")
S3_ENDPOINT = os.getenv("STATE_S3_ENDPOINT", "https://s3.fr-par.scw.cloud")
S3_REGION = os.getenv("STATE_S3_REGION", "fr-par")
S3_ACCESS_KEY = os.getenv("STATE_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("STATE_S3_SECRET_KEY", "")


class _LocalBackend:
    name = "local"

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"bad key {key!r}")
        return p

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.exists() else None

    def put(self, key: str, data: bytes) -> None:
        with self._write_lock:
            self._put(key, data)

    def _put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent requests must not share the same temporary filename.
        with tempfile.NamedTemporaryFile(dir=p.parent, prefix=p.name + ".", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
        try:
            for attempt in range(6):
                try:
                    os.replace(tmp, p)
                    break
                except OSError as exc:
                    # Windows readers/sync software can briefly lock the destination.
                    if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                        raise
                    time.sleep(0.02 * (2 ** attempt))
        finally:
            tmp.unlink(missing_ok=True)

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Atomically publish a complete new file without replacing an account."""
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p.parent, suffix='.tmp', delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
        try:
            try:
                os.link(tmp, p)
                return True
            except FileExistsError:
                return False
        finally:
            tmp.unlink(missing_ok=True)

    def list(self, prefix: str) -> List[str]:
        base = self._path(prefix.rstrip("/")) if prefix else self.root
        if not base.exists():
            return []
        out: List[str] = []
        for f in base.rglob("*"):
            if f.is_file() and not f.name.endswith(".tmp"):
                out.append(str(f.relative_to(self.root)).replace(os.sep, "/"))
        return sorted(out)


class _S3Backend:
    name = "s3"

    def __init__(self) -> None:
        import boto3  # imported lazily: only needed when the bucket is configured

        self.bucket = S3_BUCKET
        self.client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            region_name=S3_REGION,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

    def get(self, key: str) -> Optional[bytes]:
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=key)
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as exc:  # pragma: no cover - network
            if "NoSuchKey" in str(exc) or "Not Found" in str(exc):
                return None
            raise
        return obj["Body"].read()

    def put(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="application/json")

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        from botocore.exceptions import ClientError
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                                   ContentType='application/json', IfNoneMatch='*')
            return True
        except ClientError as exc:
            if exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode') in (409, 412):
                return False
            raise

    def list(self, prefix: str) -> List[str]:
        keys: List[str] = []
        token: Optional[str] = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kwargs)
            keys.extend(item["Key"] for item in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys


class StateStore:
    """JSON documents keyed by path, with a small in-process read cache so a
    request that reads a session ten times pays one backend round trip."""

    def __init__(self) -> None:
        if S3_BUCKET and S3_ACCESS_KEY and S3_SECRET_KEY:
            try:
                self.backend: Any = _S3Backend()
            except Exception as exc:
                print(f"[storage] S3 backend unavailable ({exc}); falling back to local dir")
                self.backend = _LocalBackend(LOCAL_DATA_DIR)
        else:
            self.backend = _LocalBackend(LOCAL_DATA_DIR)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        print(f"[storage] backend={self.backend.name}")

    # ---- sync API ----------------------------------------------------------
    def get_json(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            raw = self.backend.get(key)
        except Exception as exc:
            print(f"[storage] get {key} failed: {exc}")
            raw = None
        value = default if raw is None else json.loads(raw.decode("utf-8"))
        with self._lock:
            self._cache[key] = value
        return value

    def put_json(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
        data = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        try:
            self.backend.put(key, data)
        except Exception as exc:
            print(f"[storage] put {key} failed: {exc}")

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
        try:
            self.backend.delete(key)
        except Exception as exc:
            print(f"[storage] delete {key} failed: {exc}")

    def list_keys(self, prefix: str) -> List[str]:
        try:
            return self.backend.list(prefix)
        except Exception as exc:
            print(f"[storage] list {prefix} failed: {exc}")
            return []

    def append_event(self, event: Dict[str, Any]) -> str:
        ts = datetime.now(timezone.utc)
        key = f"events/{ts:%Y-%m-%d}/{ts:%H%M%S}-{int(time.time() * 1000) % 1000:03d}-{uuid.uuid4().hex[:8]}.json"
        data = json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")
        try:
            self.backend.put(key, data)
        except Exception as exc:
            print(f"[storage] event put failed: {exc}")
        return key

    # ---- async wrappers (never block the event loop on network I/O) -------
    async def aget_json(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        return await asyncio.to_thread(self.get_json, key, default)

    async def aput_json(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
        await asyncio.to_thread(self.put_json, key, value)

    async def aappend_event(self, event: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.append_event, event)


_STORE: Optional[StateStore] = None


def store() -> StateStore:
    global _STORE
    if _STORE is None:
        _STORE = StateStore()
    return _STORE
