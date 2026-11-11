import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel
from chatkit.types import ProgressUpdateEvent
from app import providers
from app.chatkit_server import MyChatKitServer


class Result(BaseModel):
    text: str


class TimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_generation_budget_caps_structured_calls(self):
        budget = {"remaining": 2, "used": 0}
        response = Mock(final_output=Result(text="ok"))
        with patch.object(providers, 'make_agent', return_value=Mock()), \
                patch.object(providers.Runner, 'run', new_callable=AsyncMock,
                             return_value=response) as run:
            await providers.run_structured(
                'quiz', 'instructions', 'prompt', Result, call_budget=budget)
            await providers.run_structured(
                'quiz-repair', 'instructions', 'prompt', Result,
                call_budget=budget)
            with self.assertRaisesRegex(RuntimeError, 'limite de generations'):
                await providers.run_structured(
                    'quiz-third-call', 'instructions', 'prompt', Result,
                    call_budget=budget)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(budget, {"remaining": 0, "used": 2})

    async def test_structured_fallback_uses_last_budgeted_call(self):
        budget = {"remaining": 2, "used": 0}
        responses = [
            Mock(final_output='not json'),
            Mock(final_output='{"text": "ok"}'),
        ]
        with patch.object(providers, 'make_agent', return_value=Mock()), \
                patch.object(providers.Runner, 'run', new_callable=AsyncMock,
                             side_effect=responses) as run:
            result = await providers.run_structured(
                'quiz', 'instructions', 'prompt', Result, call_budget=budget)
            self.assertEqual(result.text, 'ok')
            self.assertEqual(run.call_count, 2)
            self.assertEqual(budget, {"remaining": 0, "used": 2})

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

    async def test_progress_heartbeat_keeps_slow_stream_active(self):
        server = MyChatKitServer.__new__(MyChatKitServer)
        server._render = lambda thread, context, blocks: blocks

        async def slow_success(progress):
            await asyncio.sleep(.08)
            return [{'type': 'text', 'text': 'done'}]

        with patch.dict('os.environ', {
            'CHAT_RESPONSE_TIMEOUT_SECONDS': '1',
            'CHAT_PROGRESS_HEARTBEAT_SECONDS': '.02',
        }):
            events = [event async for event in server._run_with_progress(
                None, None, slow_success
            )]

        self.assertTrue(any(isinstance(event, ProgressUpdateEvent) for event in events))
        self.assertEqual(events[-1]['text'], 'done')

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
