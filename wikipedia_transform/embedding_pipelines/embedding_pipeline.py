from abc import ABC, abstractmethod

from wikipedia_transform.models import Record
from wikipedia_transform.models.types import RecordType


class EmbeddingPipeline(ABC):

    @abstractmethod
    def create_embedding_store(
        self, *, records: tuple[Record], record_type: RecordType
    ) -> None:
        """Create an embedding store."""
        pass
