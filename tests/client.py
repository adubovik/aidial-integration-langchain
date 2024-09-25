import json
from typing import AsyncIterator, List

import httpx

from tests.test_case import TestCase


class TestHTTPClient(httpx.AsyncClient):
    __test__ = False

    test_case: TestCase

    def __init__(self, *, test_case: TestCase, **kwargs):
        super().__init__(**kwargs)
        self.test_case = test_case

    async def _to_sse_stream(self, chunks: List[dict]) -> AsyncIterator[bytes]:
        for chunk in chunks:
            print(f"CHUNK: {chunk}")
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    async def send(self, request, **kwargs):
        request_dict = json.loads(request.content.decode())

        if not self.test_case.request_top_level_extra.is_valid(request_dict):
            return httpx.Response(
                request=request,
                status_code=400,
                content="Unexpected result for the top-level extra field",
            )

        for idx, value in self.test_case.request_message_extra.items():
            if not value.is_valid(request_dict["messages"][idx]):
                return httpx.Response(
                    request=request,
                    status_code=400,
                    content="Unexpected result for the per-message extra field",
                )

        message = {
            "role": "assistant",
            "content": "answer",
            **self.test_case.response_message_extra.value,
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
            **self.test_case.response_top_level_extra.value,
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
