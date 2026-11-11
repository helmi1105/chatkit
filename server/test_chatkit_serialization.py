import unittest
from datetime import datetime, timezone

from chatkit.types import Page

from app.data_store import _ITEM_ADAPTER


class PersistedChatSerializationTests(unittest.TestCase):
    def test_rehydrated_messages_serialize_in_a_page(self) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        raw_items = [
            {
                "id": "msg_user",
                "thread_id": "thr_test",
                "created_at": created_at,
                "type": "user_message",
                "content": [{"type": "input_text", "text": "hello"}],
                "attachments": [],
                "inference_options": {},
            },
            {
                "id": "msg_assistant",
                "thread_id": "thr_test",
                "created_at": created_at,
                "type": "assistant_message",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            },
        ]

        items = [_ITEM_ADAPTER.validate_python(raw) for raw in raw_items]
        payload = Page(data=items, has_more=False).model_dump_json(
            by_alias=True,
            exclude_none=True,
            context={"exclude_metadata": True},
        )

        self.assertIn('"user_message"', payload)
        self.assertIn('"assistant_message"', payload)


if __name__ == "__main__":
    unittest.main()
