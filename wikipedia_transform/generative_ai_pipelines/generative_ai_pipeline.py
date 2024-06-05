from abc import ABC, abstractmethod
from collections.abc import Iterable

from wikipedia_transform.models import Record
from wikipedia_transform.models.types import EnhancementType, RecordType


class GenerativeAiPipeline(ABC):
    """An interface to build generative AI pipelines that enhance Records."""

    @abstractmethod
    def enhance_record(
        self,
        *,
        record: Record,
        record_type: RecordType,
        enhancement_type: EnhancementType
    ) -> Iterable[Record]:
        """Enhance Records using generative AI models."""
        pass
