from pathlib import Path

from dagster import ConfigurableResource
from langchain.docstore.document import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from etl.models import OpenAiResourceParams, Record
from etl.models.types import EnrichmentType, RecordType


class OpenAiEmbeddingModelResource(ConfigurableResource):  # type: ignore
    """A ConfigurableResource to create an embedding store with OpenAI's embedding models."""

    openai_resource_params: OpenAiResourceParams

    __output_path = Path(__file__).parent.parent.absolute() / "data" / "output"

    def __create_documents(self, records: tuple[Record, ...]) -> tuple[Document, ...]:
        """Create and return Documents from Records."""

        if self.openai_resource_params.record_type == RecordType.WIKIPEDIA:
            match self.openai_resource_params.enrichment_type:
                case EnrichmentType.SUMMARY:

                    return tuple(
                        Document(
                            page_content=str(record.model_dump().get("summary")),
                            metadata={
                                "source": "https://en.wikipedia.org/wiki/{record.key}"
                            },
                        )
                        for record in records
                    )

    def __create_embedding_model(self) -> Embeddings:
        """Create and return an OpenAI embedding model."""

        store = LocalFileStore(self.__output_path)  # cache directory

        openai_embeddings_model = OpenAIEmbeddings(
            model=str(self.openai_resource_params.openai_settings.embedding_model_name)
        )

        return CacheBackedEmbeddings.from_bytes_store(
            openai_embeddings_model, store, namespace=openai_embeddings_model.model
        )

    def create_embedding_store(self, records: tuple[Record, ...]) -> FAISS:
        """Return an embedding store that was created with an OpenAI embedding model."""

        return FAISS.from_documents(
            documents=list(self.__create_documents(records=records)),
            embedding=self.__create_embedding_model(),
        )
