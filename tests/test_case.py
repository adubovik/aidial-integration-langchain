from typing import Dict

from openai import BaseModel


class TestCase(BaseModel):
    __test__ = False

    request_top_level_extra: dict
    request_message_extra: Dict[int, dict]
    response_top_level_extra: dict
    response_message_extra: dict
