from typing import Annotated

from pydantic import Field

# Tiny type for a large language model's question.
ModelQuestion = Annotated[
    str, Field(min_length=1, json_schema_extra={"strip_whitespace": "True"})
]