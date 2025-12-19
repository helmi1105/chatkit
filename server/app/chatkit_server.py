# chatkit_server.py

from __future__ import annotations
from datetime import datetime
from typing import Any, AsyncIterator
from pprint import pprint
import webbrowser  # for local dev only

from agents import TContext  # type variable for context
from chatkit.agents import AgentContext, ThreadItemConverter
from chatkit.server import ChatKitServer
from chatkit.types import (
    ThreadMetadata,
    UserMessageItem,
    Action,
    WidgetItem,
    ThreadStreamEvent,
    AssistantMessageItem,
    AssistantMessageContent,
    ThreadItemDoneEvent,
    ThreadItem,
    AssistantMessageContent,
    UserMessageItem,
)
from chatkit.store import Store, AttachmentStore
from chatkit.types import UserMessageItem as CKUserMessageItem, AssistantMessageItem as CKAssistantMessageItem
# 🔹 Import your multi-agent orchestrator
from app.orchestrator import Orchestrator
from app.widgets.qcmwidget import build_qcm_widget_from_data
from app.widgets.studywidget import build_study_widget_from_data
from app.widgets.mapwidget import build_map_widget_from_data   
from app.widgets.plotlywidget import build_plotly_widget_from_data

# ChatKit helper for converting thread messages
converter = ThreadItemConverter()



class MyChatKitServer(ChatKitServer[dict[str, Any]]):
    """ChatKit server that delegates logic to your Orchestrator."""

    def __init__(
        self,
        store: Store[TContext],
        attachment_store: AttachmentStore[TContext] | None = None,
    ):
        super().__init__(store=store, attachment_store=attachment_store)
        self.orch = Orchestrator()  # 🔹 attach orchestrator instance

    # ----------------------------------------------------------------------
    # MAIN CHAT RESPONSE
    # ----------------------------------------------------------------------
    async def respond(
        self,
        thread: ThreadMetadata,
        item: UserMessageItem | None,
        context: TContext,
    ) -> AsyncIterator[ThreadStreamEvent]:

        print("......////::",item)
        agent_context = AgentContext(
            thread=thread,
            store=self.store,
            request_context=context,
        )

        # 1) Load recent items (history)
        items_page = await self.store.load_thread_items(
            thread_id=thread.id,
            after=None,
            limit=20,
            order="desc",
            context=context,
        )
        items = list(reversed(items_page.data))

        # 2) Convert to input_items for Agents SDK
        input_items = await converter.to_agent_input(items)

        # 3) Call your orchestrator with input_items instead of plain string
        try:
            result_text = await self.orch.handle(user_input=input_items, ctx=agent_context)
        except Exception as e:
            result_text = f"⚠️ Internal error: {e}"
      

        if isinstance(result_text, dict):
            result_type = result_text.get("type")

            if result_type == "qcm":
                qcm_data = result_text["data"]
                print("QCM data:", qcm_data)

                widget_root = build_qcm_widget_from_data(qcm_data)

                widget_item = WidgetItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    widget=widget_root,
                    title=qcm_data.get("title", "QCM"),
                )

                yield ThreadItemDoneEvent(item=widget_item)
                return

            if result_type == "study":
                study_data = result_text["data"]
                widget_root = build_study_widget_from_data(study_data)
                payload = study_data.get("payload", {})

                widget_item = WidgetItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    widget=widget_root,
                    title=payload.get("title", "Study card"),
                )

                yield ThreadItemDoneEvent(item=widget_item)
                return
            if result_type == "map":
                map_data = result_text["data"]
                print("MAP DATA:", map_data)
                widget_root = build_map_widget_from_data(map_data)

                widget_item = WidgetItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    widget=widget_root,
                    title="Static map",
                )

                yield ThreadItemDoneEvent(item=widget_item)
                return
            if result_type == "plotly":
                plot_data = result_text["data"]  # expected: {"html": "...", "title": "..."} or similar
                print("PLOTLY DATA:", plot_data)

                widget_root = build_plotly_widget_from_data(plot_data)

                widget_item = WidgetItem(
                    thread_id=thread.id,
                    id=self.store.generate_item_id("message", thread, context),
                    created_at=datetime.now(),
                    widget=widget_root,
                    title=plot_data.get("title", "Plotly chart"),
                )

                yield ThreadItemDoneEvent(item=widget_item)
                return

        # else: normal text answer
        message_item = AssistantMessageItem(
            thread_id=thread.id,
            id=self.store.generate_item_id("message", thread, context),
            created_at=datetime.now(),
            content=[AssistantMessageContent(text=str(result_text))],
        )
        yield ThreadItemDoneEvent(item=message_item)
        
    # ----------------------------------------------------------------------
    # ACTION HANDLING (not used yet)
    # ----------------------------------------------------------------------
    async def action(
        self,
        _thread: ThreadMetadata,
        _action: Action[str, Any],
        _sender: WidgetItem | None,
        _context: TContext,
    ) -> AsyncIterator[ThreadStreamEvent]:

        if _action.type in ("map.open_external", "map.show_inline", "report.open"):
            # Map/report actions are rendered client-side; log and ignore if they arrive here.
            print(
                "[MyChatKitServer.action] plotly/map action received (ignored on server):",
                _action.type,
                _action.payload,    
            )
            return

        if _action.type == "qcm.submit":
            if not self.orch.hidden_answers:
                evaluation_text = "⚠️ Please generate and answer a quiz before submitting."
            else:
                submitted_answers = self._extract_answers_from_payload(_action.payload)
                if not submitted_answers:
                    evaluation_text = "⚠️ No answers received with the submission."
                else:
                    score, score20, details = self.orch.evaluator.evaluate(
                        submitted_answers, self.orch.hidden_answers
                    )
                    evaluation_text = self.orch.format_evaluation(score, score20, details)

            message_item = AssistantMessageItem(
                thread_id=_thread.id,
                id=self.store.generate_item_id("message", _thread, _context),
                created_at=datetime.now(),
                content=[AssistantMessageContent(text=evaluation_text)],
            )
            yield ThreadItemDoneEvent(item=message_item)
            return

        raise RuntimeError(f"Unsupported action type: {_action.type}")
    
    def _extract_answers_from_payload(self, payload: Any) -> dict[int, str]:
        """
        Normalize action payload from the QCM widget into {question_number: choice}.
        Supports payloads where answers live under `values.answers` or flat keys
        like `answers.1`.
        """
        normalized: dict[int, str] = {}
        data = payload if isinstance(payload, dict) else {}

        values = data.get("values")
        if not isinstance(values, dict):
            values = data

        raw_answers: dict[str, Any] = {}
        answers_section = values.get("answers")
        if isinstance(answers_section, dict):
            raw_answers = answers_section
        else:
            for key, value in values.items():
                if isinstance(key, str) and key.startswith("answers."):
                    raw_answers[key.split(".", 1)[1]] = value

        for key, value in raw_answers.items():
            try:
                question_number = int(str(key))
            except (TypeError, ValueError):
                continue

            if isinstance(value, str) and value:
                normalized[question_number] = value.strip().upper()

        return normalized
