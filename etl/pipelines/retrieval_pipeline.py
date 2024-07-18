from abc import ABC, abstractmethod
from typing import Annotated
from langchain.docstore.document import Document

from etl.models.types import RecordKey, K


class RetrievalPipeline(ABC):

    @abstractmethod
    def retrieve_documents(
        self,
        *,
        record_key: RecordKey,
        k: K,
    ) -> tuple[tuple[Document, float], ...]:
        pass
