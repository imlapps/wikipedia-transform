from abc import ABC, abstractmethod

from langchain.docstore.document import Document

from etl.models.types import DocumentsLimit, RecordKey


class RetrievalPipeline(ABC):
    """An interface to build pipelines that retrieve Documents from a VectorStore."""

    @abstractmethod
    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        k: DocumentsLimit,
    ) -> tuple[tuple[Document, float], ...]:
        """
        Return a tuple of Document-float tuple pairs, where Document is an anti-recommendation of record_key,
        and float is the similarity score of the Document.

        k is the number of Documents to retrieve.
        """
