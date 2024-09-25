import importlib
import logging
import sys
from contextlib import contextmanager
from typing import AsyncIterator, Tuple

import pytest
from openai import AsyncClient
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from tests.client import TestHTTPClient
from tests.test_case import IncludeTest, TestCase

logging.getLogger().setLevel(logging.DEBUG)

inc = IncludeTest.create


def create_test_case(incs: Tuple[bool, bool, bool, bool]):
    return TestCase(
        request_top_level_extra=inc(
            incs[0], {"custom_fields": {"configuration": {"a": "b"}}}
        ),
        request_message_extra={
            0: inc(incs[1], {"custom_content": {"state": "foobar"}})
        },
        response_top_level_extra=inc(incs[2], {"statistics": {"a": "b"}}),
        response_message_extra=inc(
            incs[3], {"custom_content": {"attachments": []}}
        ),
    )


test_case_openai = create_test_case((True, True, True, True))


def unload_langchain():
    for name in list(sys.modules):
        if any(s in name for s in ["langchain", "langsmith", "pydantic"]):
            del sys.modules[name]


def unload_module(module: str):
    if module in sys.modules:
        del sys.modules[module]


@contextmanager
def with_patch():
    module = "aidial_integration_langchain.patch"
    importlib.import_module(module)
    yield
    unload_module(module)


@contextmanager
def with_langchain():
    langchain_core = importlib.import_module("langchain_core")
    langchain_openai = importlib.import_module("langchain_openai")

    def get_langchain_chat_client(test_case: TestCase):
        return langchain_openai.ChatOpenAI(
            api_key="dummy-key",
            http_async_client=TestHTTPClient(test_case=test_case),
            max_retries=0,
        )

    yield langchain_core.messages.HumanMessage, get_langchain_chat_client

    unload_langchain()


@contextmanager
def get_langchain_manager(patched: bool):
    if patched:
        with with_patch():
            with with_langchain() as lc:
                yield lc
    else:
        with with_langchain() as lc:
            yield lc


def get_langchain_test_case(patched: bool) -> TestCase:
    if patched:
        return create_test_case((True, True, True, True))
    else:
        return create_test_case((True, False, False, False))


@pytest.mark.asyncio
@pytest.mark.parametrize("patched", [True, False])
async def test_langchain_block(patched):
    test_case = get_langchain_test_case(patched)
    with get_langchain_manager(patched) as lc:
        HumanMessage, get_client = lc

        message = HumanMessage(
            content="question",
            additional_kwargs=test_case.request_message_extra[0].value,
        )

        output = await get_client(test_case).agenerate(
            messages=[[message]],
            extra_body=test_case.request_top_level_extra.value,
        )

        generation = output.generations[0][0]
        response = generation.message

        assert test_case.response_top_level_extra.is_valid(
            response.response_metadata
        )

        assert test_case.response_message_extra.is_valid(
            response.additional_kwargs
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("patched", [True, False])
async def test_langchain_streaming(patched):
    test_case = get_langchain_test_case(patched)
    with get_langchain_manager(patched) as lc:
        HumanMessage, get_client = lc

        request_message = HumanMessage(
            content="question",
            additional_kwargs=test_case.request_message_extra[0].value,
        )

        stream = get_client(test_case).astream(
            input=[request_message],
            extra_body=test_case.request_top_level_extra.value,
        )

        async for chunk in stream:
            assert test_case.response_top_level_extra.is_valid(
                chunk.response_metadata
            )

            assert test_case.response_message_extra.is_valid(
                chunk.additional_kwargs
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", [test_case_openai])
async def test_openai_stream(test_case: TestCase):

    http_client = TestHTTPClient(test_case=test_case)
    openai_client = AsyncClient(api_key="dummy-key", http_client=http_client)

    stream: AsyncIterator[
        ChatCompletionChunk
    ] = await openai_client.chat.completions.create(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": "question",
                **test_case.request_message_extra[0].value,
            }  # type: ignore
        ],
        stream=True,
        extra_body=test_case.request_top_level_extra.value,
    )  # type: ignore

    async for c in stream:
        chunk = c.model_dump()
        assert test_case.response_top_level_extra.is_valid(chunk)
        assert test_case.response_message_extra.is_valid(
            chunk["choices"][0]["delta"]
        )
