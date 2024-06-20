from typing import Annotated

from pydantic import Field

# Tiny type for a data file's name.
DataFileName = Annotated[
    str, Field(min_length=1, json_schema_extra={"strip_whitespace": "True"})
]
