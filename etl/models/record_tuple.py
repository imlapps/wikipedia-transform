from dataclasses import dataclass

from etl.models import Record


@dataclass(frozen=True)
class RecordTuple:
    """A dataclass that holds a tuple of Records."""

    records: tuple[Record, ...]
