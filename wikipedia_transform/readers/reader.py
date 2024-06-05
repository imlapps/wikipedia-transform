from abc import ABC, abstractmethod
from collections.abc import Iterable

from wikipedia_transform.models.record import Record


class Reader(ABC):
    """An interface to read and parse Records from storage."""

    @abstractmethod
    def read(self) -> Iterable[Record]:
        """Read in output data and yield Records."""
        pass
