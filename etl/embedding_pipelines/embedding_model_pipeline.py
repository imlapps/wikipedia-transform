from abc import ABC, abstractmethod

from etl.models import Record
from etl.models.types import RecordType


class EmbeddingModelPipeline(ABC):
    """An interface to build embedding model pipelines that convert Records into embeddings."""

    @abstractmethod
    def create_embedding_store(
        self, *, records: tuple[Record, ...], record_type: RecordType
    ) -> None:
        """Create an embedding store."""
        pass
