# chatkit_server.py
"""ChatKit server adapter: turns the orchestrator's blocks (text, widgets,
QCM) into thread items, streams progress updates while a generation runs,
handles widget actions (quiz submission, action buttons, question reports)
and records learner feedback."""
from __future__ import annotations

import asyncio
import os
from contextlib import suppress
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import unquote, urlparse

from agents import TContext
from chatkit.agents import AgentContext, ThreadItemConverter
from chatkit.server import ChatKitServer
from chatkit.store import AttachmentStore, Store
from chatkit.types import (
    Action,
    AssistantMessageContent,
    AssistantMessageItem,
    ImageAttachment,
    ProgressUpdateEvent,
    ThreadItemDoneEvent,
    ThreadItemReplacedEvent,
    ThreadMetadata,
    ThreadStreamEvent,
    UserMessageItem,
    WidgetItem,
)
from openai.types.responses import ResponseInputImageParam, ResponseInputTextParam

from app.orchestrator import USER_ID_KEY, Orchestrator, extract_latest_user_text
from app.widgets.qcmwidget import build_qcm_widget_from_data


class VisualThreadItemConverter(ThreadItemConverter):
    """Images are stored locally: hand them to the model as data URLs."""

    def _image_url_for_model(self, attachment: ImageAttachment) -> str:
        url = str(attachment.preview_url)
        parsed = urlparse(url)
        marker = "/static/"
        if marker not in parsed.path:
            return url
        static_rel = unquote(parsed.path.split(marker, 1)[1]).lstrip("/")
        app_dir = Path(__file__).resolve().parent
        path = (app_dir / static_rel).resolve()
        try:
            path.relative_to(app_dir.resolve())
        except ValueError:
            return url
        if not path.exists():
            return url
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{attachment.mime_type};base64,{encoded}"

    async def attachment_to_message_content(self, attachment):
        if isinstance(attachment, ImageAttachment):
            return ResponseInputImageParam(type="input_image", image_url=self._image_url_for_model(attachment), detail="auto")
        return ResponseInputTextParam(type="input_text", text=f"Pièce jointe non prise en charge : {attachment.name} ({attachment.mime_type})")


converter = VisualThreadItemConverter()


class MyChatKitServer(ChatKitServer[dict[str, Any]]):
    def __init__(self, store: Store[TContext], attachment_store: AttachmentStore[TContext] | None = None):
        super().__init__(store=store, attachment_store=attachment_store)
        self.orch = Orchestrator()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _message(self, thread: ThreadMetadata, context: Any, text: str) -> ThreadItemDoneEvent:
        return ThreadItemDoneEvent(item=AssistantMessageItem(
            thread_id=thread.id,
            id=self.store.generate_item_id("message", thread, context),
            created_at=datetime.now(),
            content=[AssistantMessageContent(text=text)],
        ))

    def _widget(self, thread: ThreadMetadata, context: Any, widget: Any, title: str) -> ThreadItemDoneEvent:
        return ThreadItemDoneEvent(item=WidgetItem(
            thread_id=thread.id,
            id=self.store.generate_item_id("message", thread, context),
            created_at=datetime.now(),
            widget=widget,
            title=title,
        ))

    def _render(self, thread: ThreadMetadata, context: Any, blocks: Any) -> List[ThreadStreamEvent]:
        events: List[ThreadStreamEvent] = []
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        for block in blocks or []:
            if not isinstance(block, dict):
                events.append(self._message(thread, context, str(block)))
                continue
            kind = block.get("type")
            if kind == "text":
                text = str(block.get("text") or "").strip()
                if text:
                    events.append(self._message(thread, context, text))
            elif kind == "widget":
                events.append(self._widget(thread, context, block["widget"], str(block.get("title") or "")))
            elif kind == "qcm":
                data = block["data"]
                events.append(self._widget(thread, context, build_qcm_widget_from_data(data), str(data.get("title") or "Quiz")))
            else:
                events.append(self._message(thread, context, str(block)))
        return events

    async def _run_with_progress(self, thread: ThreadMetadata, context: Any, coro_factory) -> AsyncIterator[ThreadStreamEvent]:
        """Run an orchestrator call while relaying its progress messages as
        ProgressUpdateEvents, then render its blocks."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def progress(text: str, icon: str = "sparkle") -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (text, icon))

        task = asyncio.create_task(asyncio.wait_for(
            coro_factory(progress), timeout=float(os.getenv('CHAT_RESPONSE_TIMEOUT_SECONDS', '180'))))
        try:
            while not (task.done() and queue.empty()):
                try:
                    text, icon = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                yield ProgressUpdateEvent(text=text, icon=icon)
            try:
                blocks = await task
            except TimeoutError:
                blocks = [{'type': 'text', 'text': 'La réponse a pris trop de temps. Veuillez réessayer votre demande.'}]
        finally:
            # Closing a stream must not leave model calls running in the background.
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        for ev in self._render(thread, context, blocks):
            yield ev

    # ------------------------------------------------------------------
    # main chat response
    # ------------------------------------------------------------------
    async def respond(self, thread: ThreadMetadata, item: UserMessageItem | None, context: TContext) -> AsyncIterator[ThreadStreamEvent]:
        agent_context = AgentContext(thread=thread, store=self.store, request_context=context)
        items_page = await self.store.load_thread_items(thread_id=thread.id, after=None, limit=6, order="desc", context=context)
        items = list(reversed(items_page.data))
        input_items = await converter.to_agent_input(items)
        transition = self.orch.peek_transition_message(agent_context, extract_latest_user_text(input_items))
        if transition:
            yield self._message(thread, context, transition)
        async for ev in self._run_with_progress(thread, context, lambda progress: self.orch.handle(user_input=input_items, ctx=agent_context, progress=progress)):
            yield ev

    # ------------------------------------------------------------------
    # widget actions
    # ------------------------------------------------------------------
    async def action(self, _thread: ThreadMetadata, _action: Action[str, Any], _sender: WidgetItem | None, _context: TContext) -> AsyncIterator[ThreadStreamEvent]:
        if _action.type in ("map.open_external", "map.show_inline", "report.open", "radar.click"):
            return
        agent_context = AgentContext(thread=_thread, store=self.store, request_context=_context)
        if _action.type == "qcm.submit":
            submitted = self._extract_answers_from_payload(_action.payload)
            if not submitted:
                yield self._message(_thread, _context, "Aucune réponse reçue : cochez une proposition par question puis validez.")
                return
            async for ev in self._run_with_progress(_thread, _context, lambda progress: self.orch.handle_qcm_submit(submitted, agent_context, progress)):
                yield ev
            return
        if _action.type == "its.command":
            payload = _action.payload if isinstance(_action.payload, dict) else {}
            command = str(payload.get("command") or "").strip()
            if not command:
                return
            if _sender is not None and command.startswith('fiche '):
                blocks = await self.orch.handle_command(command, agent_context)
                if len(blocks) == 1 and blocks[0].get('flashcard'):
                    updated = _sender.model_copy(update={
                        'widget': blocks[0]['widget'], 'title': blocks[0]['title']})
                    # ChatKit persists replacement events with the same item id.
                    yield ThreadItemReplacedEvent(item=updated)
                else:
                    for event in self._render(_thread, _context, blocks):
                        yield event
                return
            transition = self.orch.peek_transition_message(agent_context, command)
            if transition:
                yield self._message(_thread, _context, transition)
            async for ev in self._run_with_progress(_thread, _context, lambda progress: self.orch.handle_command(command, agent_context, progress)):
                yield ev
            return
        if _action.type == "qcm.report":
            payload = _action.payload if isinstance(_action.payload, dict) else {}
            blocks = await self.orch.handle_report(payload, agent_context)
            for ev in self._render(_thread, _context, blocks):
                yield ev
            return
        raise RuntimeError(f"Unsupported action type: {_action.type}")

    async def add_feedback(self, thread_id: str, item_ids: list[str], feedback: str, context: TContext) -> None:
        await self.store.load_thread(thread_id, context)
        user_id = str((context or {}).get(USER_ID_KEY) or "anonymous")
        try:
            self.orch.record_feedback(user_id, thread_id, list(item_ids), str(feedback))
        except Exception as exc:  # noqa: BLE001
            print(f"[feedback] {exc}")

    def _extract_answers_from_payload(self, payload: Any) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        data = payload if isinstance(payload, dict) else {}
        values = data.get("values")
        if not isinstance(values, dict):
            values = data
        raw_answers: Dict[str, Any] = {}
        answers_section = values.get("answers")
        if isinstance(answers_section, dict):
            raw_answers = answers_section
        else:
            for key, value in values.items():
                if isinstance(key, str) and key.startswith("answers."):
                    raw_answers[key.split(".", 1)[1]] = value
        for key, value in raw_answers.items():
            try:
                number = int(str(key))
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value:
                normalized[number] = value.strip().upper()
        return normalized
