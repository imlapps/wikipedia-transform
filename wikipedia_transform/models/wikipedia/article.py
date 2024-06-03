from typing import Annotated

from pydantic import Field

from wikipedia_transform.models.record import Record
from wikipedia_transform.models.types import RecordKey


class Article(Record):
    """Pydantic Model to hold the contents of a Wikipedia Article."""

    key: RecordKey = Field(..., alias="title")
