from abc import ABC, abstractmethod


class RecordTransform(ABC):
    """An interface to transform Records into embeddings."""

    @abstractmethod
    def transform(self) -> None:
        """Transform Records into embeddings."""
        pass
