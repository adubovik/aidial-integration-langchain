import importlib
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Optional, Tuple

from tests.client import TestHTTPClient
from tests.test_case import IncludeTest, TestCase

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
def with_monkey_patch():
    module = "aidial_integration_langchain.patch"
    importlib.import_module(module)
    try:
        yield
    finally:
        unload_module(module)


@contextmanager
def with_langchain(is_azure: bool, is_custom_class: bool):
    langchain_core = importlib.import_module("langchain_core")
    langchain_openai = importlib.import_module(
        "aidial_integration_langchain.langchain_openai"
        if is_custom_class
        else "langchain_openai"
    )

    def get_langchain_chat_client(test_case: TestCase):
        cls, extra_kwargs = (
            (
                langchain_openai.AzureChatOpenAI,
                {"api_version": "dummy-version", "azure_endpoint": "dummy-url"},
            )
            if is_azure
            else (langchain_openai.ChatOpenAI, {})
        )
        return cls(
            api_key="dummy-key",
            http_async_client=TestHTTPClient(test_case=test_case),
            max_retries=0,
            **extra_kwargs,
        )

    try:
        yield langchain_core.messages.HumanMessage, get_langchain_chat_client
    finally:
        unload_langchain()


class PatchType(int, Enum):
    MONKEY_PATCH = 0
    CUSTOM_CLASS = 1


@contextmanager
def get_langchain_manager(patch_mode: Optional[PatchType], is_azure: bool):
    if patch_mode is None:
        with with_langchain(is_azure, False) as lc:
            yield lc
    elif patch_mode == PatchType.MONKEY_PATCH:
        with with_monkey_patch():
            with with_langchain(is_azure, False) as lc:
                yield lc
    elif patch_mode == PatchType.CUSTOM_CLASS:
        with with_langchain(is_azure, True) as lc:
            yield lc
    else:
        raise RuntimeError(f"Unexpected patch mode: {patch_mode}")


def get_langchain_test_case(patch_mode: Optional[PatchType]) -> TestCase:
    if patch_mode is None:
        return create_test_case((True, False, False, False))
    else:
        return create_test_case((True, True, True, True))
