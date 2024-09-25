import aidial_integration_langchain.patch  # isort:skip  # noqa: F401

import json
import logging
from typing import AsyncIterator, List

import httpx
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

logging.getLogger().setLevel(logging.DEBUG)


class MockHTTPClient(httpx.AsyncClient):
    async def _to_sse_stream(self, chunks: List[dict]) -> AsyncIterator[bytes]:
        for chunk in chunks:
            print(f"CHUNK: {chunk}")
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    async def send(self, request, **kwargs):
        request_dict = json.loads(request.content.decode())

        if request_dict.get("custom_fields") != {"configuration": {"a": "b"}}:
            return httpx.Response(
                request=request,
                status_code=400,
                content="Didn't receive top-level extra fields",
            )

        if request_dict["messages"][0].get("custom_content") != {
            "state": "foobar"
        }:
            return httpx.Response(
                request=request,
                status_code=400,
                content="Didn't receive per-message extra fields",
            )

        message = {
            "role": "assistant",
            "content": "answer",
            "custom_content": {"attachments": []},
        }

        stream = request_dict.get("stream") is True
        obj = "chat.completion" if not stream else "chat.completion.chunk"
        message_key = "message" if not stream else "delta"

        response = {
            "id": "chatcmpl-123",
            "created": 1677652288,
            "model": "test-model",
            "object": obj,
            "choices": [{"index": 0, message_key: message}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            "statistics": {"a": "b"},
        }

        if stream:
            return httpx.Response(
                status_code=200,
                request=request,
                content=self._to_sse_stream([response]),
                headers={"Content-Type": "text/event-stream"},
            )
        else:
            return httpx.Response(
                request=request,
                status_code=200,
                headers={"Content-Type": "application/json"},
                json=response,
            )


chat_client = ChatOpenAI(
    api_key=SecretStr("-"),
    http_async_client=MockHTTPClient(),
    max_retries=0,
)


@pytest.mark.asyncio
async def test_langchain_block():
    request_message = HumanMessage(
        content="question",
        additional_kwargs={"custom_content": {"state": "foobar"}},
    )

    output = await chat_client.agenerate(
        messages=[[request_message]],
        extra_body={"custom_fields": {"configuration": {"a": "b"}}},
    )

    generation: ChatGeneration = output.generations[0][0]  # type: ignore
    response: BaseMessage = generation.message

    assert response.response_metadata.get("statistics") == {"a": "b"}

    assert response.additional_kwargs.get("custom_content") == {
        "attachments": []
    }


@pytest.mark.asyncio
async def test_langchain_streaming():

    request_message = HumanMessage(
        content="question",
        additional_kwargs={"custom_content": {"state": "foobar"}},
    )

    stream: AsyncIterator[BaseMessageChunk] = chat_client.astream(
        input=[request_message],
        extra_body={"custom_fields": {"configuration": {"a": "b"}}},
    )

    async for chunk in stream:
        print(f"CHUNK={chunk}")
        assert chunk.additional_kwargs.get("custom_content") == {
            "attachments": []
        }
        assert chunk.response_metadata.get("statistics") == {"a": "b"}


@pytest.mark.asyncio
async def test_openai_stream():
    client = AsyncClient(api_key="dummy", http_client=MockHTTPClient())

    stream: AsyncIterator[
        ChatCompletionChunk
    ] = await client.chat.completions.create(
        model="dummy",
        messages=[
            {
                "role": "user",
                "content": "question",
                "custom_content": {"state": "foobar"},
            }  # type: ignore
        ],
        stream=True,
        extra_body={"custom_fields": {"configuration": {"a": "b"}}},
    )  # type: ignore

    async for c in stream:
        print(c)
        chunk = c.model_dump()
        assert chunk["statistics"] == {"a": "b"}
        assert chunk["choices"][0]["delta"]["custom_content"] == {
            "attachments": []
        }
