from abc import ABC, abstractmethod

from etl.models import Record


class GenerativeModelPipeline(ABC):
    """An interface to build generative AI model pipelines that enrich Records."""

    @abstractmethod
    def enrich_record(
        self,
        record: Record,
    ) -> Record:
        """Enrich Records using generative AI models."""
