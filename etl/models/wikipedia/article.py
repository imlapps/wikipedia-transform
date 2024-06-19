from typing import Annotated

from pydantic import Field

from etl.models.record import Record
from etl.models.types import RecordKey

# A tiny type for Article's summary field
Summary = Annotated[
    str,
    Field(
        min_length=1,
        json_schema_extra={"strip_whitespace": "True"},
    ),
]


class Article(Record):
    """Pydantic Model to hold the contents of a Wikipedia Article."""

    key: RecordKey = Field(..., alias="title")
    summary: Summary | None = None