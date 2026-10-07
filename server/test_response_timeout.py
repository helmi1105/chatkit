import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel
from app import providers
from app.chatkit_server import MyChatKitServer


class Result(BaseModel):
    text: str


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_timeout_cancels_and_does_not_start_fallback(self):
        cancelled = asyncio.Event()
        async def stalled(*args, **kwargs):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        with patch.object(providers, 'LLM_TIMEOUT_SECONDS', .02), patch.object(providers, '_sem', return_value=asyncio.Semaphore(1)), patch.object(providers, 'make_agent', return_value=Mock()) as make, patch.object(providers.Runner, 'run', side_effect=stalled):
            with self.assertRaises(TimeoutError):
                await providers.run_structured('test', 'instructions', 'prompt', Result)
            self.assertTrue(cancelled.is_set())
            self.assertEqual(make.call_count, 1)
        self.assertIn('réessayer', providers.friendly_llm_error(TimeoutError()))

    async def test_semaphore_queue_has_deadline(self):
        with patch.object(providers, 'LLM_TIMEOUT_SECONDS', .02), patch.object(providers, '_sem', return_value=asyncio.Semaphore(0)), patch.object(providers.Runner, 'run', new_callable=AsyncMock) as run:
            with self.assertRaises(TimeoutError):
                await providers._run_with_retry(Mock(), '', None)
            run.assert_not_called()

    async def test_response_timeout_finishes_with_message(self):
        server = MyChatKitServer.__new__(MyChatKitServer)
        server._render = lambda thread, context, blocks: blocks
        async def stalled(progress):
            await asyncio.Future()
        with patch.dict('os.environ', {'CHAT_RESPONSE_TIMEOUT_SECONDS': '.02'}):
            events = [event async for event in server._run_with_progress(None, None, stalled)]
        self.assertIn('réessayer', events[-1]['text'])

    async def test_closing_stream_cancels_work(self):
        server = MyChatKitServer.__new__(MyChatKitServer)
        cancelled = asyncio.Event()
        async def stalled(progress):
            progress('Working')
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
        stream = server._run_with_progress(None, None, stalled)
        await anext(stream)
        await stream.aclose()
        self.assertTrue(cancelled.is_set())
