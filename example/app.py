import json

import httpx
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr


class MockClient(httpx.Client):
    def send(self, request, **kwargs):
        request_dict = json.loads(request.content.decode())

        # 1. per-message request extra
        if request_dict["messages"][0].get("custom_content") != {
            "state": "foobar"
        }:
            print("(1) Missing per-message request extra")

        # 2. top-level request extra
        if request_dict.get("custom_fields") != {"configuration": {"a": "b"}}:
            print("(2) Missing top-level request extra")

        message = {
            "role": "assistant",
            "content": "answer",
            "custom_content": {
                "attachments": []
            },  # 3. per-message response extra
        }

        return httpx.Response(
            request=request,
            status_code=200,
            headers={"Content-Type": "application/json"},
            json={
                "choices": [{"index": 0, "message": message}],
                "statistics": {"a": "b"},  # 4. top-level response extra
            },
        )


chat_client = AzureChatOpenAI(
    api_key=SecretStr("dummy-key"),
    api_version="dummy-version",
    azure_endpoint="dummy-url",
    http_client=MockClient(),
    max_retries=0,
)

request_message = HumanMessage(
    content="question",
    additional_kwargs={
        "custom_content": {"state": "foobar"}
    },  # 1. per-message request extra
)

output = chat_client.generate(
    messages=[[request_message]],
    extra_body={
        "custom_fields": {"configuration": {"a": "b"}}
    },  # 2. top-level request extra
)

generation: ChatGeneration = output.generations[0][0]  # type: ignore
response: BaseMessage = generation.message

# 3. per-message response extra
if response.additional_kwargs.get("custom_content") != {"attachments": []}:
    print("(3) Missing per-message response extra")

# 4. top-level response extra
if response.response_metadata.get("statistics") != {"a": "b"}:
    print("(4) Missing top-level response extra")
