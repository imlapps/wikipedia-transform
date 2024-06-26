from abc import ABC, abstractmethod
from collections.abc import Iterable

from etl.models import Record


class Reader(ABC):
    """An interface to read and parse Records from storage."""

    def read() -> Iterable[Record]:
        """Read in data from storage and yield them as Records."""
        pass
