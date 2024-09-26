"""
Monkey patching langchain_openai to proxy extra parameters to the upstream model.

Workaround for https://github.com/langchain-ai/langchain/issues/26617
"""

import logging
import sys

from aidial_integration_langchain.patch.decorators import (
    patch_convert_chunk_to_generation_chunk,
    patch_convert_delta_to_message_chunk,
    patch_convert_dict_to_message,
    patch_convert_message_to_dict,
    patch_create_chat_result,
)

if "langchain_openai" in sys.modules.keys():
    raise RuntimeError(
        "Import patch module before any langchain_openai imports"
    )

import langchain_openai.chat_models.base

logging.getLogger(__name__).info("Patching langchain_open library...")

langchain_openai.chat_models.base._convert_message_to_dict = (
    patch_convert_message_to_dict(
        langchain_openai.chat_models.base._convert_message_to_dict
    )
)

if hasattr(langchain_openai.chat_models.base, "BaseChatOpenAI"):
    # Since langchain_openai>=0.1.5
    langchain_openai.chat_models.base.BaseChatOpenAI._create_chat_result = (
        patch_create_chat_result(
            langchain_openai.chat_models.base.BaseChatOpenAI._create_chat_result
        )
    )
elif hasattr(langchain_openai.chat_models.base, "ChatOpenAI"):
    langchain_openai.chat_models.base.ChatOpenAI._create_chat_result = (
        patch_create_chat_result(
            langchain_openai.chat_models.base.ChatOpenAI._create_chat_result
        )
    )

langchain_openai.chat_models.base._convert_dict_to_message = (
    patch_convert_dict_to_message(
        langchain_openai.chat_models.base._convert_dict_to_message
    )
)

langchain_openai.chat_models.base._convert_delta_to_message_chunk = (
    patch_convert_delta_to_message_chunk(
        langchain_openai.chat_models.base._convert_delta_to_message_chunk
    )
)

if hasattr(
    langchain_openai.chat_models.base, "_convert_chunk_to_generation_chunk"
):
    langchain_openai.chat_models.base._convert_chunk_to_generation_chunk = (  # type: ignore
        patch_convert_chunk_to_generation_chunk(
            langchain_openai.chat_models.base._convert_chunk_to_generation_chunk  # type: ignore
        )
    )
