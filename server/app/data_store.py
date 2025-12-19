# my_data_store.py (or wherever you keep it)

"""
Implement chatkit.store.Store to persist threads, messages, and files using your preferred database.
You are responsible for implementing the chatkit.store.Store class using the data store of your choice.
When implementing the store, you must allow for the Thread/Attachment/ThreadItem type shapes changing between library versions.
The recommended approach for relational databases is to serialize models into JSON-typed columns
instead of separating model fields across multiple columns.
"""

from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime, timezone

from chatkit.store import NotFoundError, Store, AttachmentStore
from chatkit.types import Attachment, Page, ThreadItem, ThreadMetadata

USER_ID_KEY = "userId"


@dataclass
class _ThreadState:
    thread: ThreadMetadata
    items: List[ThreadItem]


@dataclass
class _UserState:
    threads: Dict[str, _ThreadState]


class MyDataStore(Store[dict[str, Any]]):
    """Simple in-memory store compatible with the ChatKit server interface."""

    def __init__(self) -> None:
        self._users: Dict[str, _UserState] = {}

    # ===========================
    # Helpers
    # ===========================
    def _get_user_state(self, user_id: str) -> _UserState:
        state = self._users.get(user_id)
        if state:
            return state
        state = _UserState(threads={})
        self._users[user_id] = state
        return state

    def _get_thread_metatdata(self, user_id: str, thread_id: str) -> ThreadMetadata:
        user_state = self._get_user_state(user_id=user_id)
        thread_state = user_state.threads.get(thread_id)
        if not thread_state:
            raise NotFoundError(f"Thread {thread_id} not found")
        return thread_state.thread.model_copy(deep=True)

    def _get_user_id(self, context: dict[str, Any]) -> str:
        id = context.get(USER_ID_KEY)
        if id is None:
            raise Exception("User id required")
        return id

    def _get_thread_items(self, user_id: str, thread_id: str) -> List[ThreadItem]:
        user_state = self._get_user_state(user_id=user_id)
        state = user_state.threads.get(thread_id)
        if state is None:
            state = _ThreadState(
                thread=ThreadMetadata(
                    id=thread_id,
                    created_at=datetime.now(timezone.utc),
                ),
                items=[],
            )
            user_state.threads[thread_id] = state
        return state.items

    # ===========================
    # Thread metadata
    # ===========================
    async def load_thread(self, thread_id: str, context: dict[str, Any]) -> ThreadMetadata:
        user_id = self._get_user_id(context=context)
        return self._get_thread_metatdata(user_id=user_id, thread_id=thread_id)

    async def save_thread(self, thread: ThreadMetadata, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context=context)
        user_state = self._get_user_state(user_id=user_id)
        state = user_state.threads.get(thread.id)
        if state:
            state.thread = thread
        else:
            state = _ThreadState(
                thread=thread,
                items=[],
            )
            user_state.threads[thread.id] = state  # 🔹 attach new thread
        self._users[user_id] = user_state

    async def load_threads(
        self,
        limit: int,
        after: str | None,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadMetadata]:
        user_id = self._get_user_id(context=context)
        user_state = self._get_user_state(user_id=user_id)
        threads_map = user_state.threads

        threads = sorted(
            (state.thread for state in threads_map.values()),
            key=lambda t: t.created_at or datetime.min,
            reverse=(order == "desc"),
        )

        if after:
            index_map = {thread.id: idx for idx, thread in enumerate(threads)}
            start = index_map.get(after, -1) + 1
        else:
            start = 0

        slice_threads = threads[start : start + limit + 1]
        has_more = len(slice_threads) > limit
        slice_threads = slice_threads[:limit]
        next_after = slice_threads[-1].id if has_more and slice_threads else None
        return Page(
            data=slice_threads,
            has_more=has_more,
            after=next_after,
        )

    async def delete_thread(self, thread_id: str, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context=context)
        user_state = self._get_user_state(user_id=user_id)
        user_state.threads.pop(thread_id, None)
        self._users[user_id] = user_state

    # ===========================
    # Thread items
    # ===========================
    async def load_thread_items(
        self,
        thread_id: str,
        after: str | None,
        limit: int,
        order: str,
        context: dict[str, Any],
    ) -> Page[ThreadItem]:
        user_id = self._get_user_id(context=context)
        items = [
            item.model_copy(deep=True)
            for item in self._get_thread_items(user_id=user_id, thread_id=thread_id)
        ]
        items.sort(
            key=lambda item: getattr(item, "created_at", datetime.now(timezone.utc)),
            reverse=(order == "desc"),
        )

        if after:
            index_map = {item.id: idx for idx, item in enumerate(items)}
            start = index_map.get(after, -1) + 1
        else:
            start = 0

        slice_items = items[start : start + limit + 1]
        has_more = len(slice_items) > limit
        slice_items = slice_items[:limit]
        next_after = slice_items[-1].id if has_more and slice_items else None
        return Page(data=slice_items, has_more=has_more, after=next_after)

    async def add_thread_item(
        self, thread_id: str, item: ThreadItem, context: dict[str, Any]
    ) -> None:
        user_id = self._get_user_id(context=context)
        self._get_thread_items(user_id=user_id, thread_id=thread_id).append(
            item.model_copy(deep=True)
        )

    async def save_item(self, thread_id: str, item: ThreadItem, context: dict[str, Any]) -> None:
        user_id = self._get_user_id(context=context)
        items = self._get_thread_items(user_id=user_id, thread_id=thread_id)
        for idx, existing in enumerate(items):
            if existing.id == item.id:
                items[idx] = item.model_copy(deep=True)
                return
        items.append(item.model_copy(deep=True))

    async def load_item(self, thread_id: str, item_id: str, context: dict[str, Any]) -> ThreadItem:
        user_id = self._get_user_id(context=context)
        for item in self._get_thread_items(user_id=user_id, thread_id=thread_id):
            if item.id == item_id:
                return item.model_copy(deep=True)
        raise NotFoundError(f"Item {item_id} not found")

    async def delete_thread_item(
        self, thread_id: str, item_id: str, context: dict[str, Any]
    ) -> None:
        user_id = self._get_user_id(context=context)
        user_state = self._get_user_state(user_id=user_id)
        state = user_state.threads.get(thread_id)
        if not state:
            return
        state.items = [item for item in state.items if item.id != item_id]

    # ===========================
    # Attachments
    # ===========================
    async def save_attachment(
        self,
        attachment: Attachment,
        context: dict[str, Any],
    ) -> None:
        raise NotImplementedError(
            "MyDataStore does not persist attachments. Provide a Store implementation "
            "that enforces authentication and authorization before enabling uploads."
        )

    async def load_attachment(
        self,
        attachment_id: str,
        context: dict[str, Any],
    ) -> Attachment:
        raise NotImplementedError(
            "MyDataStore does not load attachments. Provide a Store implementation "
            "that enforces authentication and authorization before enabling uploads."
        )

    async def delete_attachment(self, attachment_id: str, context: dict[str, Any]) -> None:
        raise NotImplementedError(
            "MyDataStore does not delete attachments because they are never stored."
        )
