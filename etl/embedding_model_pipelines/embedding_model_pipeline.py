from abc import ABC, abstractmethod

from langchain.docstore.document import Document
from langchain_core.vectorstores import VectorStore


class EmbeddingModelPipeline(ABC):
    """An interface to build embedding model pipelines that transform Records into embeddings."""

    @abstractmethod
    def create_embedding_store(self, documents: tuple[Document, ...]) -> VectorStore:
        """Create an embedding store."""
