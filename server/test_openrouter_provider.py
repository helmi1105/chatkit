import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents import OpenAIChatCompletionsModel
from pydantic import BaseModel

from app import providers


class OpenRouterProviderTests(unittest.TestCase):
    def test_context_accepts_openrouter(self) -> None:
        choice = providers.set_provider_from_context(
            {"provider": "openrouter", "api_key": "personal-key"}
        )

        self.assertEqual(choice.provider, providers.PROVIDER_OPENROUTER)
        self.assertEqual(choice.api_key, "personal-key")
        self.assertEqual(providers.provider_label(choice), "OpenRouter")

    def test_build_model_uses_openrouter_endpoint_and_personal_key(self) -> None:
        choice = providers.ProviderChoice(
            provider=providers.PROVIDER_OPENROUTER,
            api_key="personal-key",
        )

        with patch.object(providers, "OPENROUTER_MODEL", "anthropic/test-model"):
            model = providers.build_model(choice)

        self.assertIsInstance(model, OpenAIChatCompletionsModel)
        self.assertEqual(model.model, "anthropic/test-model")
        self.assertEqual(str(model._client.base_url), "https://openrouter.ai/api/v1/")
        self.assertEqual(model._client.api_key, "personal-key")

    def test_build_model_falls_back_to_server_key_and_vision_model(self) -> None:
        choice = providers.ProviderChoice(provider=providers.PROVIDER_OPENROUTER)

        with (
            patch.object(providers, "OPENROUTER_API_KEY", "server-key"),
            patch.object(providers, "OPENROUTER_VISION_MODEL", "google/test-vision"),
        ):
            model = providers.build_model(choice, vision=True)

        self.assertEqual(model.model, "google/test-vision")
        self.assertEqual(model._client.api_key, "server-key")

    def test_openrouter_never_receives_openai_file_search(self) -> None:
        choice = providers.ProviderChoice(provider=providers.PROVIDER_OPENROUTER)
        self.assertEqual(providers.build_tools(choice), [])


class OpenRouterStructuredOutputTests(unittest.IsolatedAsyncioTestCase):
    async def test_openrouter_skips_native_schema_and_validates_json_text(self) -> None:
        class Result(BaseModel):
            value: str

        token = providers.current_provider.set(
            providers.ProviderChoice(provider=providers.PROVIDER_OPENROUTER)
        )
        output_types = []

        def make_agent(*args, **kwargs):
            output_types.append(kwargs.get("output_type"))
            return object()

        try:
            with (
                patch.object(providers, "make_agent", side_effect=make_agent),
                patch.object(
                    providers,
                    "_run_with_retry",
                    new=AsyncMock(return_value=SimpleNamespace(
                        final_output='{"value":"ok"}'
                    )),
                ) as run,
            ):
                result = await providers.run_structured(
                    "quiz", "instructions", "prompt", Result
                )
        finally:
            providers.current_provider.reset(token)

        self.assertEqual(result.value, "ok")
        self.assertEqual(output_types, [None])
        self.assertEqual(run.await_count, 1)


if __name__ == "__main__":
    unittest.main()
