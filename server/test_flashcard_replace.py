import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from chatkit.types import Action, ThreadMetadata, WidgetItem, ThreadItemReplacedEvent
from chatkit.widgets import Card, Text
from app.chatkit_server import MyChatKitServer


class ReplacementTests(unittest.IsolatedAsyncioTestCase):
    async def test_flashcard_reuses_sender_id(self):
        server = MyChatKitServer.__new__(MyChatKitServer)
        server.store = object()
        widget = Card(children=[Text(value='Revealed answer')])
        server.orch = SimpleNamespace(handle_command=AsyncMock(return_value=[
            {'type': 'widget', 'flashcard': True, 'title': 'Revision', 'widget': widget}]))
        thread = ThreadMetadata(id='thr_test', created_at=datetime.now())
        sender = WidgetItem(id='w_test', thread_id=thread.id, created_at=datetime.now(), widget=Card(children=[]))
        events = [event async for event in server.action(thread,
            Action(type='its.command', payload={'command': 'fiche reveler deck 0'}), sender, {})]
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ThreadItemReplacedEvent)
        self.assertEqual(events[0].item.id, sender.id)
        self.assertEqual(events[0].item.widget, widget)
