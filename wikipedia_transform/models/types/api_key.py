from typing import Annotated

from pydantic import Field


ApiKey = Annotated[
    str, Field(min_length=1, json_schema_extra={"strip_whitespace": True})
]
