from abc import ABC, abstractmethod
from collections.abc import Iterable

from etl.models import Record


class GenerativeModelPipeline(ABC):
    """An interface to build generative AI model pipelines that enrich Records."""

    @abstractmethod
    def enrich_record(
        self,
        *,
        record: Record,
    ) -> Iterable[Record]:
        """Enrich Records using generative AI models."""
        pass
