import pytest

from tests.test_case import TestCase
from tests.test_cases import run_test_openai_stream
from tests.utils import test_case_openai


@pytest.mark.asyncio
@pytest.mark.parametrize("test_case", [test_case_openai])
async def test_openai_stream(test_case: TestCase):
    await run_test_openai_stream(test_case)
