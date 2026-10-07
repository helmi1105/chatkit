# app/data_store.py
"""ChatKit Store + AttachmentStore.

Threads and their items are kept in memory for speed and WRITTEN THROUGH to
the shared state store (storage.py: S3 bucket, or a local directory) as one
JSON document per learner, `threads/<userId>.json`, so a conversation
survives the scale-to-zero of the Scaleway container. Items are serialised
with pydantic's TypeAdapter over the ThreadItem union (checked 2026-09-15:
a UserMessageItem round-trips through dump_json / validate_json).

Images stay on the container disk (uploads/) and are handed to the model as
data URLs at request time; they are not persisted across restarts, which is
acceptable for a visual question asked and answered in the same session."""
from __future__ import annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from chatkit.store import AttachmentStore, NotFoundError, Store
from chatkit.types import (
    Attachment,
    AttachmentCreateParams,
    AttachmentUploadDescriptor,
    ImageAttachment,
    Page,
    ThreadItem,
    ThreadMetadata,
)
from pydantic import TypeAdapter

from app.storage import store as state_store

USER_ID_KEY = "userId"
APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_IMAGE_ATTACHMENT_BYTES = int(os.getenv("MAX_IMAGE_ATTACHMENT_BYTES", str(5 * 1024 * 1024)))
MAX_ITEMS_PER_THREAD = int(os.getenv("MAX_ITEMS_PER_THREAD", "400"))

_ITEM_ADAPTER: TypeAdapter = TypeAdapter(ThreadItem)


@dataclass
class _ThreadState:
    thread: ThreadMetadata
    items: List[ThreadItem]


@dataclass
class _UserState:
    threads: Dict[str, _ThreadState]
    loaded: bool = False


class MyDataStore(Store[dict[str, Any]], AttachmentStore[dict[str, Any]]):
    def __init__(self) -> None:
        self._users: Dict[str, _UserState] = {}
        self._attachments: Dict[str, Attachment] = {}
        self._attachment_paths: Dict[str, Path] = {}
        self._attachment_owners: Dict[str, str] = {}
        self._state = state_store()
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ===========================
    # persistence helpers
    # ===========================
    @staticmethod
    def _key(user_id: str) -> str:
        return f"threads/{user_id}.json"

    def _serialize(self, state: _UserState) -> Dict[str, Any]:
        out: Dict[str, Any] = {"threads": []}
        for ts in state.threads.values():
            out["threads"].append({
                "thread": ts.thread.model_dump(mode="json"),
                "items": [_ITEM_ADAPTER.dump_python(it, mode="json") for it in ts.items[-MAX_ITEMS_PER_THREAD:]],
            })
        return out

    def _deserialize(self, data: Any) -> Dict[str, _ThreadState]:
        threads: Dict[str, _ThreadState] = {}
        if not isinstance(data, dict):
            return threads
        for entry in data.get("threads") or []:
            try:
                thread = ThreadMetadata.model_validate(entry.get("thread") or {})
                items: List[ThreadItem] = []
                for raw in entry.get("items") or []:
                    try:
                        items.append(_ITEM_ADAPTER.validate_python(raw))
                    except Exception as exc:  # one bad item must not lose the thread
                        print(f"[data_store] skipped item: {exc}")
                threads[thread.id] = _ThreadState(thread=thread, items=items)
            except Exception as exc:
                print(f"[data_store] skipped thread: {exc}")
        return threads

    def _get_user_state(self, user_id: str) -> _UserState:
        state = self._users.get(user_id)
        if state is None:
            state = _UserState(threads={})
            self._users[user_id] = state
        if not state.loaded:
            state.loaded = True
            data = self._state.get_json(self._key(user_id), None)
            if data:
                for tid, ts in self._deserialize(data).items():
                    state.threads.setdefault(tid, ts)
        return state

    def _persist(self, user_id: str) -> None:
        state = self._users.get(user_id)
        if state is None:
            return
        payload = self._serialize(state)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._state.aput_json(self._key(user_id), payload))
        except RuntimeError:
            self._state.put_json(self._key(user_id), payload)

    def _get_user_id(self, context: dict[str, Any]) -> str:
        user_id = context.get(USER_ID_KEY)
        if user_id is None:
            raise Exception("User id required")
        return str(user_id)

    def _get_thread_items(self, user_id: str, thread_id: str) -> List[ThreadItem]:
        user_state = self._get_user_state(user_id)
        state = user_state.threads.get(thread_id)
        if state is None:
            state = _ThreadState(thread=ThreadMetadata(id=thread_id, created_at=datetime.now(timezone.utc)), items=[])
            user_state.threads[thread_id] = state
        return state.items

    # ===========================
    # Thread metadata
    # ===========================
    async def load_thread(self, thread_id: str, context: dict[str, Any]) -> ThreadMetadata:
        user_id = self._get_user_id(context)
        state = self._get_user_state(user_id).threads.get(thread_id)
        if not state:
            raise NotFoundError(f"Thread {thread_id} not found")
        return self._with_image_origin(state.thread.model_copy(deep=True))

    @staticmethod
    def _with_image_origin(thread: ThreadMetadata) -> ThreadMetadata:
        parsed = urlparse(os.getenv('PUBLIC_BASE_URL', 'http://127.0.0.1:8000'))
        if parsed.scheme in {'http', 'https'} and parsed.netloc:
            origin = f'{parsed.scheme}://{parsed.netloc}'
            thread.allowed_image_domains = list(dict.fromkeys([*(thread.allowed_image_domains or []), origin]))
        return thread

    async def save_thread(self, thread: ThreadMetadata, context: dict[str, Any]) -> None:
        self._with_image_origin(thread)
        user_id = self._get_user_id(context)
        user_state = self._get_user_state(user_id)
        state = user_state.threads.get(thread.id)
        if state:
            state.thread = thread
        else:
            user_state.threads[thread.id] = _ThreadState(thread=thread, items=[])
        self._persist(user_id)

    async def load_threads(self, limit: int, after: str | None, order: str, context: dict[str, Any]) -> Page[ThreadMetadata]:
        user_id = self._get_user_id(context)
        threads = sorted(
            (self._with_image_origin(state.thread) for state in self._get_user_state(user_id).threads.values()),
            key=lambda t: t.created_at or datetime.min,
            reverse=(order == "desc"),
        )
        start = 0
        if after:
            index_map = {t.id: idx for idx, t in enumerate(threads)}
            start = index_map.get(after, -1) + 1
        slice_threads = threads[start : start + limit + 1]
        has_more = len(slice_threads) > limit
        slice_threads = slice_threads[:limit]
        return Page(data=slice_threads, has_more=has_more, after=slice_threads[-1].id if has_more and slice_threads else None)

    async def delete_thread(self, thread_id: str, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context)
        self._get_user_state(user_id).threads.pop(thread_id, None)
        self._persist(user_id)

    # ===========================
    # Thread items
    # ===========================
    async def load_thread_items(self, thread_id: str, after: str | None, limit: int, order: str, context: dict[str, Any]) -> Page[ThreadItem]:
        user_id = self._get_user_id(context)
        items = [item.model_copy(deep=True) for item in self._get_thread_items(user_id, thread_id)]
        items.sort(key=lambda item: getattr(item, "created_at", datetime.now(timezone.utc)), reverse=(order == "desc"))
        start = 0
        if after:
            index_map = {item.id: idx for idx, item in enumerate(items)}
            start = index_map.get(after, -1) + 1
        slice_items = items[start : start + limit + 1]
        has_more = len(slice_items) > limit
        slice_items = slice_items[:limit]
        return Page(data=slice_items, has_more=has_more, after=slice_items[-1].id if has_more and slice_items else None)

    async def add_thread_item(self, thread_id: str, item: ThreadItem, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context)
        self._get_thread_items(user_id, thread_id).append(item.model_copy(deep=True))
        self._persist(user_id)

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context)
        items = self._get_thread_items(user_id, thread_id)
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item.model_copy(deep=True)
                self._persist(user_id)
                return
        items.append(item.model_copy(deep=True))
        self._persist(user_id)

    async def load_item(self, thread_id: str, item_id: str, context: dict[str, Any]) -> ThreadItem:
        user_id = self._get_user_id(context)
        for item in self._get_thread_items(user_id, thread_id):
            if item.id == item_id:
                return item.model_copy(deep=True)
        raise NotFoundError(f"Item {item_id} not found")

    async def delete_thread_item(self, thread_id: str, item_id: str, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context)
        state = self._get_user_state(user_id).threads.get(thread_id)
        if state:
            state.items = [item for item in state.items if item.id != item_id]
            self._persist(user_id)

    # ===========================
    # Attachments (images only, local disk)
    # ===========================
    def _attachment_extension(self, name: str, mime_type: str) -> str:
        suffix = Path(name).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return suffix
        return {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(mime_type.lower(), ".img")

    async def create_attachment(self, input: AttachmentCreateParams, context: dict[str, Any]) -> Attachment:
        if not input.mime_type.lower().startswith("image/"):
            raise ValueError("Seules les images sont acceptées (photo d'un symbole).")
        if input.size > MAX_IMAGE_ATTACHMENT_BYTES:
            raise ValueError("Image trop lourde (5 Mo maximum).")
        attachment_id = self.generate_attachment_id(input.mime_type, context)
        filename = f"{attachment_id}{self._attachment_extension(input.name, input.mime_type)}"
        path = UPLOAD_DIR / filename
        frontend_base = str(context.get('frontend_origin') or 'http://localhost:3000').rstrip('/') + '/backend'
        attachment = ImageAttachment(
            id=attachment_id,
            name=input.name,
            mime_type=input.mime_type,
            upload_descriptor=AttachmentUploadDescriptor(
                url=f"{frontend_base}/attachments/{attachment_id}/upload",
                method="PUT",
                headers={"Content-Type": input.mime_type, "X-ChatKit-Request": "1"},
            ),
            preview_url=f"{frontend_base}/static/uploads/{filename}",
        )
        self._attachments[attachment_id] = attachment
        self._attachment_paths[attachment_id] = path
        self._attachment_owners[attachment_id] = self._get_user_id(context)
        return attachment

    async def upload_attachment_bytes(self, attachment_id: str, content: bytes, content_type: str | None = None) -> None:
        attachment = self._attachments.get(attachment_id)
        path = self._attachment_paths.get(attachment_id)
        if attachment is None or path is None:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        if len(content) > MAX_IMAGE_ATTACHMENT_BYTES:
            raise ValueError("Image trop lourde (5 Mo maximum).")
        path.write_bytes(content)
        # The hosted ChatKit iframe cannot use our first-party session cookie.
        # Return the preview inline after an authenticated upload instead.
        preview = 'data:' + attachment.mime_type + ';base64,' + base64.b64encode(content).decode('ascii')
        self._attachments[attachment_id] = attachment.model_copy(update={"upload_descriptor": None, "preview_url": preview})

    async def save_attachment(self, attachment: Attachment, context: dict[str, Any]) -> None:
        if self._attachment_owners.get(attachment.id) != self._get_user_id(context):
            raise NotFoundError(f"Attachment {attachment.id} not found")
        self._attachments[attachment.id] = attachment

    async def load_attachment(self, attachment_id: str, context: dict[str, Any]) -> Attachment:
        if self._attachment_owners.get(attachment_id) != self._get_user_id(context):
            raise NotFoundError(f"Attachment {attachment_id} not found")
        attachment = self._attachments.get(attachment_id)
        if attachment is None:
            raise NotFoundError(f"Attachment {attachment_id} not found")
        return attachment.model_copy(update={"upload_descriptor": None})

    async def delete_attachment(self, attachment_id: str, context: dict[str, Any]) -> None:
        if self._attachment_owners.get(attachment_id) != self._get_user_id(context):
            raise NotFoundError(f"Attachment {attachment_id} not found")
        self._attachment_owners.pop(attachment_id, None)
        path = self._attachment_paths.pop(attachment_id, None)
        self._attachments.pop(attachment_id, None)
        if path and path.exists():
            path.unlink()
