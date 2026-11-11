# app/providers.py
"""Model-provider selection (Mistral by default, OpenAI or OpenRouter on
request, optional personal key per session) and the ONE way every generator talks to a model:
`run_structured()`, which asks for a typed (Pydantic) output, retries on
rate limits and falls back to lenient text parsing.

History (cahier de labo 2026-09-15): the generators used to be free-text
JSON parsed with json.loads(); Mistral wrapped it in code fences or returned
an empty tool-calling turn, OpenAI had no credit left, and every generation
went through a slow file_search tool loop. Grounding is now injected in the
prompt (see content.py), so the Mistral path needs no tool at all."""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, List, MutableMapping, Optional, Type, TypeVar

from agents import (
    Agent,
    AgentOutputSchema,
    AsyncOpenAI,
    FileSearchTool,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    Runner,
)
from pydantic import BaseModel, ValidationError

PROVIDER_MISTRAL = "mistral"
PROVIDER_OPENAI = "openai"
PROVIDER_OPENROUTER = "openrouter"
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", PROVIDER_MISTRAL)
KNOWN_PROVIDERS = (PROVIDER_MISTRAL, PROVIDER_OPENAI, PROVIDER_OPENROUTER)

VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "vs_6a116b3869e08191aa26f247b322a8c1")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral/mistral-large-latest")
# pixtral-large-latest was retired: "Invalid model" on 2026-09-15; mistral-large
# now carries the vision capability itself (checked on /v1/models).
MISTRAL_VISION_MODEL = os.getenv("MISTRAL_VISION_MODEL", "mistral/mistral-large-latest")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini-2024-07-18")
OPENAI_INVENTORY_MODEL = os.getenv("OPENAI_INVENTORY_MODEL", "gpt-5.4")
OPENAI_VISUAL_EXPLANATION_MODEL = os.getenv("OPENAI_VISUAL_EXPLANATION_MODEL", "gpt-5.4")
OPENAI_FREE_QUESTION_MODEL = os.getenv("OPENAI_FREE_QUESTION_MODEL", "gpt-5.4")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", OPENROUTER_MODEL)
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "PDF2ITS")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
# Free Mistral tier: ~1 request/second. Parallel generation is throttled here.
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "2"))

T = TypeVar("T", bound=BaseModel)


class GenerationBudgetExceeded(RuntimeError):
    """Raised before a model invocation would exceed a local generation cap."""


@dataclass
class ProviderChoice:
    provider: str = DEFAULT_PROVIDER
    api_key: Optional[str] = None  # user-supplied override; never logged, never persisted


current_provider: ContextVar[ProviderChoice] = ContextVar("current_provider", default=ProviderChoice())
_semaphore: Optional[asyncio.Semaphore] = None


def _sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(max(1, LLM_CONCURRENCY))
    return _semaphore


def set_provider_from_context(request_context: Any) -> ProviderChoice:
    request_context = request_context or {}
    provider = request_context.get("provider")
    choice = ProviderChoice(
        provider=provider if provider in KNOWN_PROVIDERS else DEFAULT_PROVIDER,
        api_key=request_context.get("api_key") or None,
    )
    current_provider.set(choice)
    return choice


def provider_label(choice: Optional[ProviderChoice] = None) -> str:
    choice = choice or current_provider.get()
    return {
        PROVIDER_MISTRAL: "Mistral",
        PROVIDER_OPENAI: "OpenAI",
        PROVIDER_OPENROUTER: "OpenRouter",
    }.get(choice.provider, choice.provider)


def uses_shared_openai_key(choice: ProviderChoice) -> bool:
    return choice.provider == PROVIDER_OPENAI and not choice.api_key


def build_model(choice: ProviderChoice, *, vision: bool = False) -> Any:
    if choice.provider == PROVIDER_OPENAI:
        name = OPENAI_VISION_MODEL if vision else OPENAI_MODEL
        if choice.api_key:
            return OpenAIResponsesModel(model=name, openai_client=AsyncOpenAI(api_key=choice.api_key))
        return name
    if choice.provider == PROVIDER_OPENROUTER:
        name = OPENROUTER_VISION_MODEL if vision else OPENROUTER_MODEL
        headers = {}
        if OPENROUTER_HTTP_REFERER:
            headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
        if OPENROUTER_APP_NAME:
            headers["X-Title"] = OPENROUTER_APP_NAME
        client = AsyncOpenAI(
            api_key=choice.api_key or OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers,
        )
        return OpenAIChatCompletionsModel(model=name, openai_client=client)
    model_name = MISTRAL_VISION_MODEL if vision else MISTRAL_MODEL
    from agents.extensions.models.litellm_model import LitellmModel

    return LitellmModel(model=model_name, api_key=choice.api_key or MISTRAL_API_KEY)


def build_tools(choice: ProviderChoice, *, max_results: int = 6) -> List[Any]:
    """Only the shared-OpenAI path still gets the hosted file_search over the
    private vector store; every other path is grounded by prompt injection."""
    if uses_shared_openai_key(choice):
        return [FileSearchTool(max_num_results=max_results, vector_store_ids=[VECTOR_STORE_ID])]
    return []


def build_model_settings(choice: ProviderChoice) -> ModelSettings:
    if choice.provider == PROVIDER_OPENAI:
        return ModelSettings(store=False)
    return ModelSettings()


def make_agent(
    name: str,
    instructions: str,
    *,
    output_type: Optional[Type[BaseModel]] = None,
    vision: bool = False,
    with_tools: bool = False,
) -> Agent[Any]:
    choice = current_provider.get()
    dedicated_models = {
        "Visual-inventory": OPENAI_INVENTORY_MODEL,
        "Visual-symbol": OPENAI_VISUAL_EXPLANATION_MODEL,
        "Learner-question": OPENAI_FREE_QUESTION_MODEL,
    }
    if name in dedicated_models:
        # Dedicated OpenAI models for visual tasks and free-answer SVG generation.
        # A personal Mistral key must never be sent to OpenAI.
        key = choice.api_key if choice.provider == PROVIDER_OPENAI else None
        choice = ProviderChoice(provider=PROVIDER_OPENAI, api_key=key)
        model = OpenAIResponsesModel(
            model=dedicated_models[name],
            openai_client=AsyncOpenAI(api_key=key or os.getenv("OPENAI_API_KEY")),
        )
    else:
        model = build_model(choice, vision=vision)
    kwargs: dict = {}
    if output_type is not None:
        # OpenAI models support strict Structured Outputs.  LiteLLM and
        # OpenRouter models vary by upstream provider, so retain the lenient
        # schema plus the existing validation/fallback path for them.
        kwargs["output_type"] = AgentOutputSchema(
            output_type,
            strict_json_schema=choice.provider == PROVIDER_OPENAI,
        )
    return Agent[Any](
        name=name,
        model=model,
        tools=build_tools(choice) if with_tools else [],
        instructions=instructions,
        model_settings=build_model_settings(choice),
        **kwargs,
    )


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_RATE_HINTS = ("429", "rate limit", "rate_limit", "capacity", "overloaded", "503", "timeout", "timed out")


def strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def _extract_json_object(text: str) -> str:
    raw = strip_code_fence(text)
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        return raw
    start = min(starts)
    end = max(raw.rfind("}"), raw.rfind("]"))
    return raw[start : end + 1] if end > start else raw


def _example_for(model_cls: Any) -> Any:
    """Placeholder instance of a Pydantic model, used as the shape hint in
    the text fallback ("summary": "...", "hints": ["...", "..."])."""
    import typing

    def for_annotation(ann: Any) -> Any:
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)
        if origin is typing.Union or str(origin) == "types.UnionType":
            non_none = [a for a in args if a is not type(None)]
            return for_annotation(non_none[0]) if non_none else None
        if origin in (list, typing.List):
            return [for_annotation(args[0])] if args else ["..."]
        if origin in (dict, typing.Dict):
            return {}
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return _example_for(ann)
        if ann is int:
            return 1
        if ann is float:
            return 0.5
        if ann is bool:
            return True
        return "..."

    return {name: for_annotation(f.annotation) for name, f in model_cls.model_fields.items()}


def is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(h in text for h in _RATE_HINTS) and "credit" not in text


async def _run_with_retry(
    agent: Agent[Any],
    prompt: Any,
    ctx: Any,
    *,
    call_budget: Optional[MutableMapping[str, int]] = None,
    budget_name: str = "generation",
) -> Any:
    # Include queueing, SDK retries and our backoff in one deadline.
    return await asyncio.wait_for(
        _run_attempts(
            agent, prompt, ctx,
            call_budget=call_budget,
            budget_name=budget_name,
        ),
        timeout=LLM_TIMEOUT_SECONDS,
    )


async def _run_attempts(
    agent: Agent[Any],
    prompt: Any,
    ctx: Any,
    *,
    call_budget: Optional[MutableMapping[str, int]] = None,
    budget_name: str = "generation",
) -> Any:
    delay = 2.0
    last: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            if call_budget is not None:
                remaining = int(call_budget.get("remaining", 0))
                if remaining <= 0:
                    raise GenerationBudgetExceeded(
                        f"{budget_name}: limite de generations atteinte"
                    )
                call_budget["remaining"] = remaining - 1
                call_budget["used"] = int(call_budget.get("used", 0)) + 1
            async with _sem():
                # Structured QCM calls do not use tools. One turn prevents a
                # single Runner invocation from hiding extra model turns.
                max_turns = 1 if call_budget is not None else 4
                return await Runner.run(agent, prompt, context=ctx, max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= LLM_MAX_RETRIES or not is_rate_limit(exc):
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 12.0)
    assert last is not None
    raise last


async def run_structured(
    name: str,
    instructions: str,
    prompt: Any,
    model_cls: Type[T],
    ctx: Any = None,
    *,
    vision: bool = False,
    with_tools: bool = False,
    call_budget: Optional[MutableMapping[str, int]] = None,
) -> T:
    """Generate and locally validate a typed response.

    OpenRouter routes can advertise structured output at the model level while
    the selected upstream rejects the SDK's response-format parameter. Skip
    that known-failing request and go directly to JSON text for OpenRouter.
    Dedicated OpenAI-only paths keep native structured output.
    """
    dedicated_openai = name in {
        "Visual-inventory", "Visual-symbol", "Learner-question",
    }
    use_native_structured = (
        current_provider.get().provider != PROVIDER_OPENROUTER
        or dedicated_openai
    )
    if use_native_structured:
        try:
            agent = make_agent(name, instructions, output_type=model_cls, vision=vision, with_tools=with_tools)
            res = await _run_with_retry(
                agent, prompt, ctx,
                call_budget=call_budget,
                budget_name=name,
            )
            out = res.final_output
            if isinstance(out, model_cls):
                return out
            if isinstance(out, BaseModel):
                return model_cls.model_validate(out.model_dump())
            if isinstance(out, (dict, list)):
                return model_cls.model_validate(out)
            if isinstance(out, str) and out.strip():
                return model_cls.model_validate_json(_extract_json_object(out))
            raise ValueError("empty structured output")
        except Exception as first_exc:  # noqa: BLE001
            if (isinstance(first_exc, (TimeoutError, GenerationBudgetExceeded))
                    or is_rate_limit(first_exc) or "credit" in str(first_exc).lower()
                    or "401" in str(first_exc)):
                raise
            print(f"[{name}] structured output failed ({type(first_exc).__name__}: {str(first_exc)[:160]}), text fallback")
    # An EXAMPLE object, not the JSON schema: given the schema, Mistral echoed
    # the schema itself ({"properties": {...}}) instead of an instance
    # (smoke test 2026-09-15).
    text_instructions = (
        instructions
        + "\nRéponds UNIQUEMENT par un objet JSON valide ayant exactement cette forme (remplace les valeurs), sans texte autour ni bloc de code :\n"
        + json.dumps(_example_for(model_cls), ensure_ascii=False)
    )
    agent = make_agent(name, text_instructions, vision=vision, with_tools=with_tools)
    last_err: Optional[Exception] = None
    for _ in range(2):
        res = await _run_with_retry(
            agent, prompt, ctx,
            call_budget=call_budget,
            budget_name=name,
        )
        raw = _extract_json_object(str(res.final_output or ""))
        if not raw:
            last_err = ValueError("empty output")
            continue
        try:
            return model_cls.model_validate_json(raw)
        except ValidationError as exc:
            last_err = exc
    raise RuntimeError(f"{name}: sortie du modèle inexploitable ({last_err})")


async def run_text(name: str, instructions: str, prompt: Any, ctx: Any = None, *, vision: bool = False, with_tools: bool = False) -> str:
    agent = make_agent(name, instructions, vision=vision, with_tools=with_tools)
    res = await _run_with_retry(agent, prompt, ctx)
    return str(res.final_output or "").strip()


_QUOTA_ERROR_HINTS = ("rate limit", "rate_limit", "quota", "429", "insufficient_quota", "capacity", "credit")
_AUTH_ERROR_HINTS = ("401", "unauthorized", "invalid api key", "authentication")


def friendly_llm_error(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "Le modèle a mis trop de temps à répondre. Veuillez réessayer dans un instant."
    from app.embeddings import EmbeddingError
    if isinstance(exc, EmbeddingError):
        return str(exc)
    text = str(exc).lower()
    label = provider_label()
    if "credit" in text or "insufficient_quota" in text:
        return (
            f"Le crédit {label} partagé est épuisé. "
            "Ouvrez les réglages (icône engrenage) pour utiliser votre propre clé, ou changez de modèle."
        )
    if any(hint in text for hint in _QUOTA_ERROR_HINTS):
        return (
            f"Le service {label} est saturé pour le moment (limite de requêtes). "
            "Réessayez dans une minute, ou ajoutez votre propre clé dans les réglages."
        )
    if any(hint in text for hint in _AUTH_ERROR_HINTS):
        return (
            f"La clé {label} a été refusée. "
            "Vérifiez la clé collée dans les réglages, ou retirez-la pour revenir à la clé partagée."
        )
    return f"Une erreur est survenue côté {label} : {str(exc)[:200]}"
