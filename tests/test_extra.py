import aidial_integration_langchain.patch  # isort:skip  # noqa: F401

import logging
from typing import AsyncIterator, cast

import pytest
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_openai import ChatOpenAI
from openai import AsyncClient
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pydantic import SecretStr

from tests.client import TestHTTPClient
from tests.test_case import TestCase
from tests.utils.json import is_subdict

logging.getLogger().setLevel(logging.DEBUG)


test_case = TestCase(
    request_top_level_extra={"custom_fields": {"configuration": {"a": "b"}}},
    request_message_extra={0: {"custom_content": {"state": "foobar"}}},
    response_top_level_extra={"statistics": {"a": "b"}},
    response_message_extra={"custom_content": {"attachments": []}},
)

http_client = TestHTTPClient(test_case=test_case)

langchain_chat_client = ChatOpenAI(
    api_key=SecretStr("dummy-key"),
    http_async_client=http_client,
    max_retries=0,
)

openai_client = AsyncClient(api_key="dummy-key", http_client=http_client)


@pytest.mark.asyncio
async def test_langchain_block():
    message = HumanMessage(
        content="question",
        additional_kwargs=test_case.request_message_extra[0],
    )

    output = await langchain_chat_client.agenerate(
        messages=[[message]],
        extra_body=test_case.request_top_level_extra,
    )

    generation = cast(ChatGeneration, output.generations[0][0])
    response: BaseMessage = generation.message

    assert is_subdict(
        test_case.response_top_level_extra, response.response_metadata
    )

    assert is_subdict(
        test_case.response_message_extra, response.additional_kwargs
    )


@pytest.mark.asyncio
async def test_langchain_streaming():

    request_message = HumanMessage(
        content="question",
        additional_kwargs=test_case.request_message_extra[0],
    )

    stream: AsyncIterator[BaseMessageChunk] = langchain_chat_client.astream(
        input=[request_message],
        extra_body=test_case.request_top_level_extra,
    )

    async for chunk in stream:
        assert is_subdict(
            test_case.response_top_level_extra, chunk.response_metadata
        )

        assert is_subdict(
            test_case.response_message_extra, chunk.additional_kwargs
        )


@pytest.mark.asyncio
async def test_openai_stream():

    stream: AsyncIterator[
        ChatCompletionChunk
    ] = await openai_client.chat.completions.create(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": "question",
                **test_case.request_message_extra[0],
            }  # type: ignore
        ],
        stream=True,
        extra_body=test_case.request_top_level_extra,
    )  # type: ignore

    async for c in stream:
        chunk = c.model_dump()
        assert is_subdict(test_case.response_top_level_extra, chunk)
        assert is_subdict(
            test_case.response_message_extra, chunk["choices"][0]["delta"]
        )
