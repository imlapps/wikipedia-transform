from typing import Annotated

from langchain.docstore.document import Document
from langchain_community.vectorstores import VectorStore
from pydantic import Field

from etl.models.types import ModelQuestion, RecordKey
from etl.pipelines import RetrievalPipeline


class AntiRecommendationRetrievalPipeline(RetrievalPipeline):
    """
    A concrete implementation of RetrievalPipeline.

    Retrieves anti-recommendations of a Record key using Documents stored in a VectorStore.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.__vector_store = vector_store

    def __create_query(
        self,
        *,
        record_key: RecordKey,
        k: Annotated[int, Field(default=1, json_schema_extra={"min": 1})],
    ) -> ModelQuestion:
        """Return a query for the retrieval algorithm."""

        return f"What are {k} Wikipedia articles that are dissimilar but surprisingly similar to the Wikipedia article {record_key}"

    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        k: Annotated[int, Field(default=1, json_schema_extra={"min": 1})],
    ) -> tuple[tuple[Document, float], ...]:
        """
        Return a tuple of Document-float tuple pairs, where the Document is an anti-recommendation of record_key,
        and the float is the similarity score of the Document.

        k is the number of Documents to retrieve.
        """
        return tuple(
            self.__vector_store.similarity_search_with_score(
                query=self.__create_query(
                    record_key=record_key,
                    k=k,
                ),
                k=k,
                score_threshold=0.5,
            )
        )
