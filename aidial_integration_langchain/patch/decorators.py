from typing import Any, Dict, Mapping, Optional, Type, Union

import openai
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk


def _mask_by_keys(d: dict, keys: list[str]) -> dict:
    return {k: d[k] for k in keys if k in d}


EXTRA_REQUEST_MESSAGE_FIELDS = ["custom_content"]
# NOTE: not really needed, since they are propagated automatically via extra_body
# EXTRA_REQUEST_FIELDS = ["addons", "max_prompt_tokens", "custom_fields"]
EXTRA_RESPONSE_MESSAGE_FIELDS = ["custom_content"]
EXTRA_RESPONSE_FIELDS = ["statistics"]


def patch_convert_message_to_dict(func):
    def _func(message: BaseMessage) -> dict:
        result = func(message)
        result.update(
            _mask_by_keys(
                message.additional_kwargs, EXTRA_REQUEST_MESSAGE_FIELDS
            )
        )

        return result

    return _func


def patch_create_chat_result(func):
    def _func(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ):
        result = func(self, response, generation_info)

        _dict = (
            response if isinstance(response, dict) else response.model_dump()
        )

        if extra := _mask_by_keys(_dict, EXTRA_RESPONSE_FIELDS):
            result.llm_output = result.llm_output or {}
            result.llm_output.update(extra)

        return result

    return _func


def patch_convert_dict_to_message(func):
    def _func(_dict: Mapping[str, Any]) -> BaseMessage:
        result = func(_dict)
        result.additional_kwargs.update(_mask_by_keys(_dict, EXTRA_RESPONSE_MESSAGE_FIELDS))  # type: ignore

        return result

    return _func


def patch_convert_delta_to_message_chunk(func):
    def _func(
        _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
    ) -> BaseMessageChunk:
        result = func(_dict, default_class)
        result.additional_kwargs.update(_mask_by_keys(_dict, EXTRA_RESPONSE_MESSAGE_FIELDS))  # type: ignore
        return result

    return _func


def patch_convert_chunk_to_generation_chunk(func):
    def _func(
        chunk: dict,
        default_chunk_class: Type,
        base_generation_info: Optional[Dict],
    ) -> Optional[ChatGenerationChunk]:
        result = func(chunk, default_chunk_class, base_generation_info)
        if result:
            result.message.response_metadata.update(
                _mask_by_keys(chunk, EXTRA_RESPONSE_FIELDS)
            )
        return result

    return _func
