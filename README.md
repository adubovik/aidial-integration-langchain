# Overview

Monkey patching `langchain_openai` to proxy extra parameters to the upstream model.

A hacky workaround for https://github.com/langchain-ai/langchain/issues/26617.

## Usage

Copy the patch module to your project, then import before Langchain is imported:

```python
import dial_langchain_patch # isort:skip  # noqa: F401
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
)
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_openai import ChatOpenAI

chat_client = ChatOpenAI()

request_message = HumanMessage(
    content="question",
    additional_kwargs={"custom_content": {"state": "foobar"}}, # per-message request extra
)

output = await chat_client.agenerate(
    messages=[[request_message]],
    extra_body={"custom_fields": {"configuration": {"a": "b"}}}, # top-level request extra
)

generation: ChatGeneration = output.generations[0][0]  # type: ignore
response: BaseMessage = generation.message

# top-level response extra
assert response.response_metadata.get("statistics") == {"a": "b"}

# per-message response extra
assert response.additional_kwargs.get("custom_content") == {
    "attachments": []
}
```

## Dependencies

Tested for the following dependencies:

```txt
langchain==0.3.0
langchain-core==0.3.1
langchain-openai==0.2.0
```
