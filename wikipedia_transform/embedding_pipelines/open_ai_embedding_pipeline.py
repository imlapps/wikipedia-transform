from collections.abc import Callable

from langchain.docstore.document import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from mypy_extensions import NamedArg

from wikipedia_transform.embedding_pipelines import EmbeddingPipeline
from wikipedia_transform.models import OpenAiEmbeddingModel, Record, settings
from wikipedia_transform.models.types import OpenAiEmbeddingModelName, RecordType


class OpenAiEmbeddingPipeline(EmbeddingPipeline):
    """
    A concrete implementation of EmbeddingPipeline that uses OpenAI's embedding models to convert
    Records into embeddings.

    Embeddings are stored at settings.output_path.
    """

    def _create_documents(
        self, *, records: tuple[Record, ...], record_type: RecordType
    ) -> tuple[Document, ...]:
        """Create and return Documents from Records."""

        return tuple(
            [
                Document(
                    page_content=record.key,
                    metadata={"source": record_type.value}
                    | record.model_dump(by_alias=True, exclude={"key", "url"}),
                )
                for record in records
            ]
        )

    def _create_embedding_model(
        self, open_ai_embedding_model: OpenAiEmbeddingModel
    ) -> Embeddings:
        """Create and return an OpenAI embedding model."""

        store = LocalFileStore(settings.output_path)  # cache directory

        openai_embeddings_model = OpenAIEmbeddings(
            model=open_ai_embedding_model.name)

        return CacheBackedEmbeddings.from_bytes_store(
            openai_embeddings_model, store, namespace=openai_embeddings_model.model
        )

    def _create_embedding_store(
        self,
        *,
        records: tuple[Record, ...],
        record_type: RecordType,
        create_documents: Callable[
            [
                NamedArg(tuple[Record, ...], "records"),
                NamedArg(RecordType, "record_type"),
            ],
            tuple[Document, ...],
        ],
        create_embedding_model: Callable[[OpenAiEmbeddingModel], Embeddings]
    ) -> None:
        """Create a generalized workflow to create an embedding store."""

        FAISS.from_documents(
            documents=list(create_documents(
                records=records, record_type=record_type)),
            embedding=create_embedding_model(
                OpenAiEmbeddingModel(
                    name=OpenAiEmbeddingModelName.TEXT_EMBEDDING_MODEL_SMALL
                )
            ),
        )

    def create_embedding_store(
        self, *, records: tuple[Record, ...], record_type: RecordType
    ) -> None:
        """Create an embedding store with OpenAI's embedding model."""

        self._create_embedding_store(
            records=records,
            record_type=record_type,
            create_documents=self._create_documents,
            create_embedding_model=self._create_embedding_model,
        )
