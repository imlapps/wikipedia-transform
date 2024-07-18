from etl.pipelines import RetrievalPipeline
from typing import Annotated
from pydantic import Field
from langchain.docstore.document import Document
from langchain_community.vectorstores import VectorStore
from etl.models.types import RecordKey


class OpenAiRetrievalPipeline(RetrievalPipeline):
    def __init__(self, vector_store: VectorStore) -> None:
        self.__vector_store = vector_store

    def __create_question(
        self,
        *,
        record_key: RecordKey,
        number_of_documents_to_retrieve: Annotated[
            int, Field(default=1, json_schema_extra={"min": 1})
        ],
    ):
        return f"What are {number_of_documents_to_retrieve} Wikipedia articles that are dissimilar but surprisingly similar to the Wikipedia article {record_key}"

    def __create_vector_store_retriever(self, number_of_documents_to_retrieve):
        return self.__vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": number_of_documents_to_retrieve,
                "score_threshold": 0.5,
            },
        )

    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        number_of_documents_to_retrieve: Annotated[
            int, Field(default=1, json_schema_extra={"min": 1})
        ],
    ) -> tuple[Document, ...]:
        return self.__create_vector_store_retriever(
            number_of_documents_to_retrieve=number_of_documents_to_retrieve
        ).invoke(
            self.__create_question(
                record_key=record_key,
                number_of_documents_to_retrieve=number_of_documents_to_retrieve,
            )
        )
