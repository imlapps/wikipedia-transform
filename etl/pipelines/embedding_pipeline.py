from abc import ABC, abstractmethod
from typing import Annotated, final

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from pydantic import Field


class EmbeddingPipeline(ABC):
    """An interface to build embedding pipelines that transform Records into embeddings."""

    @abstractmethod
    def _create_embedding_model(self) -> Embeddings:
        """Return an embedding model that will be used to create an embedding store."""

    @final
    def create_embedding_store(
        self,
        *,
        documents: tuple[Document, ...],
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        score_threshold: Annotated[float, Field(json_schema_extra={"min": 0, "max": 1})]
    ) -> VectorStore:
        """
        Return an embedding store that contains embeddings of Documents.

        Vector embeddings will be retrieved from the returned VectorStore using the selected distance strategy.
        EUCLIDEAN_DISTANCE is the default distance_stratgy for the FAISS VectorStore.

        All vector embeddings retrieved from the returned VectorStore must have a similarity score greater than or equal to the score_threshold.

        """

        return FAISS.from_documents(
            documents=list(documents),
            embedding=self._create_embedding_model(),
            distance_strategy=distance_strategy,
            score_threshold=score_threshold,
        )
