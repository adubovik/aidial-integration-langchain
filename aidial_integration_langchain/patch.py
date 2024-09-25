"""
Monkey patching langchain_openai to proxy extra parameters to the upstream model.

Workaround for https://github.com/langchain-ai/langchain/issues/26617
"""

import sys

if "langchain_openai" in sys.modules.keys():
    raise RuntimeError(
        "Import patch module before any langchain_openai imports"
    )

import logging
from typing import Any, Dict, Mapping, Optional, Type, Union

import langchain_openai.chat_models.base
import openai
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.chat_models.base import (
    _convert_chunk_to_generation_chunk as _original_convert_chunk_to_generation_chunk,
)
from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk as _original_convert_delta_to_message_chunk,
)
from langchain_openai.chat_models.base import (
    _convert_dict_to_message as _original_convert_dict_to_message,
)
from langchain_openai.chat_models.base import (
    _convert_message_to_dict as _original_convert_message_to_dict,
)

logging.getLogger(__name__).info("Patching langchain_open library...")


def _mask_by_keys(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys if k in d}


EXTRA_REQUEST_MESSAGE_FIELDS = ["custom_content"]
# NOTE: not really needed, since they are propagated automatically via extra_body
# EXTRA_REQUEST_FIELDS = ["addons", "max_prompt_tokens", "custom_fields"]
EXTRA_RESPONSE_MESSAGE_FIELDS = ["custom_content"]
EXTRA_RESPONSE_FIELDS = ["statistics"]


##############################


def _patched_convert_message_to_dict(message: BaseMessage) -> dict:
    result = _original_convert_message_to_dict(message)
    result.update(
        _mask_by_keys(message.additional_kwargs, EXTRA_REQUEST_MESSAGE_FIELDS)
    )

    return result


langchain_openai.chat_models.base._convert_message_to_dict = (
    _patched_convert_message_to_dict
)

##############################

_original_create_chat_result = BaseChatOpenAI._create_chat_result


def _patched_create_chat_result(
    self,
    response: Union[dict, openai.BaseModel],
    generation_info: Optional[Dict] = None,
):
    result = _original_create_chat_result(self, response, generation_info)

    _dict = response if isinstance(response, dict) else response.model_dump()

    if extra := _mask_by_keys(_dict, EXTRA_RESPONSE_FIELDS):
        result.llm_output = result.llm_output or {}
        result.llm_output.update(extra)

    return result


BaseChatOpenAI._create_chat_result = _patched_create_chat_result

##############################


def _patched_convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    result = _original_convert_dict_to_message(_dict)
    result.additional_kwargs.update(_mask_by_keys(_dict, EXTRA_RESPONSE_MESSAGE_FIELDS))  # type: ignore

    return result


langchain_openai.chat_models.base._convert_dict_to_message = (
    _patched_convert_dict_to_message
)

##############################


def _patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    result = _original_convert_delta_to_message_chunk(_dict, default_class)
    result.additional_kwargs.update(_mask_by_keys(_dict, EXTRA_RESPONSE_MESSAGE_FIELDS))  # type: ignore
    return result


langchain_openai.chat_models.base._convert_delta_to_message_chunk = (
    _patched_convert_delta_to_message_chunk
)


##############################


def _patched_convert_chunk_to_generation_chunk(
    chunk: dict, default_chunk_class: Type, base_generation_info: Optional[Dict]
) -> Optional[ChatGenerationChunk]:
    result = _original_convert_chunk_to_generation_chunk(
        chunk, default_chunk_class, base_generation_info
    )
    if result:
        result.message.response_metadata.update(
            _mask_by_keys(chunk, EXTRA_RESPONSE_FIELDS)
        )
    return result


langchain_openai.chat_models.base._convert_chunk_to_generation_chunk = (
    _patched_convert_chunk_to_generation_chunk
)
