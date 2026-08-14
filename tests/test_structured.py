"""Structured output tests: JSON repair and schema-validated generation."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from cortex.errors import StructuredOutputError
from cortex.llm.base import ChatMessage, LLMResponse
from cortex.llm.gateway import LLMGateway
from cortex.llm.mock import MockProvider
from cortex.llm.structured import generate_structured, repair_json


class Person(BaseModel):
    name: str
    age: int


def test_repair_json_plain():
    assert repair_json('{"a": 1}') == {"a": 1}


def test_repair_json_fenced_and_prose():
    raw = '好的，结果如下：\n```json\n{"name": "张三", "age": 30}\n```\n希望有帮助'
    assert repair_json(raw) == {"name": "张三", "age": 30}


def test_repair_json_trailing_commas():
    assert repair_json('{"name": "张三", "age": 30,}') == {"name": "张三", "age": 30}


def test_repair_json_no_object_raises():
    with pytest.raises(StructuredOutputError):
        repair_json("no json here at all")


async def test_generate_structured_success():
    def handler(messages, tools=None, response_format=None):
        assert response_format is not None
        return LLMResponse(content=json.dumps({"name": "李四", "age": 25}, ensure_ascii=False))

    gateway = LLMGateway([MockProvider(handler=handler)])
    person = await generate_structured(
        gateway, [ChatMessage(role="user", content="提取人物")], Person
    )
    assert person == Person(name="李四", age=25)


async def test_generate_structured_repairs_then_succeeds():
    def handler(messages, tools=None, response_format=None):
        if len(messages) <= 2:
            return LLMResponse(content="```json\n{\"name\": \"王五\"}\n```")  # missing age
        return LLMResponse(content=json.dumps({"name": "王五", "age": 40}, ensure_ascii=False))

    gateway = LLMGateway([MockProvider(handler=handler)])
    person = await generate_structured(
        gateway, [ChatMessage(role="user", content="提取人物")], Person, max_attempts=2
    )
    assert person == Person(name="王五", age=40)


async def test_generate_structured_exhausted_raises():
    def handler(messages, tools=None, response_format=None):
        return LLMResponse(content="not json at all")

    gateway = LLMGateway([MockProvider(handler=handler)])
    with pytest.raises(StructuredOutputError):
        await generate_structured(
            gateway, [ChatMessage(role="user", content="x")], Person, max_attempts=2
        )
