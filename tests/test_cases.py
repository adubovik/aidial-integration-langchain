from typing import AsyncIterator, Optional

from openai import AsyncClient
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from tests.client import TestHTTPClient
from tests.test_case import TestCase
from tests.utils import (
    PatchType,
    get_langchain_manager,
    get_langchain_test_case,
)

_TOP_LEVEL_ERROR = "Unexpected result for the response top-level extra field"
_MESSAGE_ERROR = "Unexpected result for the response message extra field"


async def run_test_langchain_block(
    patch_mode: Optional[PatchType], is_azure: bool
):
    with get_langchain_manager(patch_mode, is_azure) as lc:
        test_case = get_langchain_test_case(patch_mode, False)
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
        ), _TOP_LEVEL_ERROR

        assert test_case.response_message_extra.is_valid(
            response.additional_kwargs
        ), _MESSAGE_ERROR


async def run_test_langchain_streaming(
    patch_mode: Optional[PatchType], is_azure: bool
):
    with get_langchain_manager(patch_mode, is_azure) as lc:
        test_case = get_langchain_test_case(patch_mode, True)
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
            ), _TOP_LEVEL_ERROR

            assert test_case.response_message_extra.is_valid(
                chunk.additional_kwargs
            ), _MESSAGE_ERROR


async def run_test_openai_stream(test_case: TestCase):
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
        assert test_case.response_top_level_extra.is_valid(
            chunk
        ), _TOP_LEVEL_ERROR

        assert test_case.response_message_extra.is_valid(
            chunk["choices"][0]["delta"]
        ), _MESSAGE_ERROR
