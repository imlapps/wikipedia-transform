from abc import ABC, abstractmethod
from typing import Annotated
from langchain.docstore.document import Document
from pydantic import Field
from etl.models.types import RecordKey


class RetrievalPipeline(ABC):

    @abstractmethod
    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        number_of_retrieved_documents: Annotated[
            int, Field(default=1, json_schema_extra={"min": 1})
        ],
    ) -> tuple[Document, ...]:
        pass
