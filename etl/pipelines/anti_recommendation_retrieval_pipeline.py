from etl.pipelines import RetrievalPipeline
from typing import Annotated
from pydantic import Field
from langchain.docstore.document import Document
from langchain_community.vectorstores import VectorStore
from etl.models.types import RecordKey, ModelQuestion


class AntiRecommendationRetrievalPipeline(RetrievalPipeline):
    def __init__(self, vector_store: VectorStore) -> None:
        self.__vector_store = vector_store

    def __create_question(
        self,
        *,
        record_key: RecordKey,
        k: Annotated[int, Field(default=1, json_schema_extra={"min": 1})],
    ) -> ModelQuestion:
        return f"What are {k} Wikipedia articles that are dissimilar but surprisingly similar to the Wikipedia article {record_key}"

    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        k: Annotated[int, Field(default=1, json_schema_extra={"min": 1})],
    ) -> tuple[tuple[Document, float], ...]:
        return tuple(
            self.__vector_store.similarity_search_with_score(
                query=self.__create_question(
                    record_key=record_key,
                    k=k,
                ),
                k=k,
            )
        )
