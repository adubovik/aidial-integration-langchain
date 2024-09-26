import pytest

from tests.test_cases import (
    run_test_langchain_block,
    run_test_langchain_streaming,
)
from tests.utils import PatchType


@pytest.mark.asyncio
async def test_langchain_block():
    await run_test_langchain_block(PatchType.CUSTOM_CLASS, True)


@pytest.mark.asyncio
async def test_langchain_streaming():
    await run_test_langchain_streaming(PatchType.CUSTOM_CLASS, True)
