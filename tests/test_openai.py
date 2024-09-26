import pytest

from tests.test_cases import run_test_openai_stream
from tests.utils import get_openai_test_case


@pytest.mark.asyncio
async def test_openai_stream():
    await run_test_openai_stream(get_openai_test_case())
