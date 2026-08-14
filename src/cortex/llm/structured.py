"""Structured output: schema-constrained generation with validation and repair.

The LLM is asked for JSON matching a Pydantic model's schema. If parsing or
validation fails, the error is fed back as a repair hint and the model retries.
This is the production pattern behind extraction, classification and planning.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from cortex.errors import StructuredOutputError
from cortex.llm.base import ChatMessage, LLMResponse
from cortex.llm.gateway import LLMGateway
from cortex.logging import log, trace_span
from cortex.metrics import METRICS

logger = logging.getLogger("cortex.llm.structured")

ModelT = TypeVar("ModelT", bound=BaseModel)


def repair_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from arbitrary model output.

    Handles markdown fences, surrounding prose and trailing commas.
    """
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # first balanced {...} span
    start = cleaned.find("{")
    if start == -1:
        raise StructuredOutputError(f"no JSON object found in output: {text[:200]!r}")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                cleaned = cleaned[start : i + 1]
                break
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError("output JSON is not an object")
    return value


def _model_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


async def generate_structured(
    gateway: LLMGateway,
    messages: list[ChatMessage],
    model: type[ModelT],
    *,
    provider: str | None = None,
    system_prompt: str | None = None,
    max_attempts: int = 2,
    temperature: float = 0.0,
) -> ModelT:
    """Generate one Pydantic-validated object, repairing on failure."""
    if not issubclass(model, BaseModel):
        raise TypeError("model must be a pydantic BaseModel subclass")
    schema = _model_schema(model)
    request: list[ChatMessage] = []
    if system_prompt:
        request.append(ChatMessage(role="system", content=system_prompt))
    request.extend(messages)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        with trace_span("llm.structured", model=model.__name__, attempt=attempt):
            response: LLMResponse = await gateway.chat(
                request,
                provider=provider,
                response_format={"type": "json_schema", "json_schema": schema},
                temperature=temperature,
            )
        raw = response.content or ""
        try:
            data = repair_json(raw)
            instance = model.model_validate(data)
            METRICS.inc("structured_output_total", model=model.__name__, status="ok")
            return instance
        except (StructuredOutputError, ValidationError) as exc:
            last_error = exc
            METRICS.inc("structured_output_total", model=model.__name__, status="repair")
            log(
                logger,
                logging.WARNING,
                "structured output invalid, repairing",
                model=model.__name__,
                attempt=attempt,
                error=str(exc),
            )
            request.append(ChatMessage(role="assistant", content=raw))
            request.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous output could not be parsed. Fix it and respond with "
                        f"only a JSON object matching the schema. Error: {exc}"
                    ),
                )
            )
    METRICS.inc("structured_output_total", model=model.__name__, status="failed")
    raise StructuredOutputError(f"{model.__name__} generation failed after {max_attempts} attempts: {last_error}")
