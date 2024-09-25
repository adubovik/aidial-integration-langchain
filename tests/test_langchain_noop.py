import logging

import pytest

from tests.test_cases import (
    run_test_langchain_block,
    run_test_langchain_streaming,
)

logging.getLogger().setLevel(logging.DEBUG)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_block(is_azure):
    await run_test_langchain_block(None, is_azure)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_azure", [True, False])
async def test_langchain_streaming(is_azure):
    await run_test_langchain_streaming(None, is_azure)
