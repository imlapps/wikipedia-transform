from typing import Annotated

from pydantic import Field

# A tiny type for Article's summary field.
Summary = Annotated[
    str,
    Field(
        min_length=1,
        json_schema_extra={"strip_whitespace": True},
    ),
]
