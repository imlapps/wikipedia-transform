from abc import ABC, abstractmethod

from etl.models import Record


class RecordEnrichmentPipeline(ABC):
    """An interface to build pipelines that enrich Records."""

    @abstractmethod
    def enrich_record(
        self,
        record: Record,
    ) -> Record:
        """Enrich and return a Record"""
