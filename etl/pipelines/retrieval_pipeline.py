from abc import ABC, abstractmethod

from langchain.docstore.document import Document

from etl.models.types import K, RecordKey


class RetrievalPipeline(ABC):
    """An interface to build pipelines that retrieve Documents from a VectorStore."""

    @abstractmethod
    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        k: K,
    ) -> tuple[tuple[Document, float], ...]:
        """
        Return anti-recommendations of a record_key.
        k is the number of Documents to retrieve.
        """
