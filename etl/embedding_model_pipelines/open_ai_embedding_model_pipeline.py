from langchain.docstore.document import Document
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from etl.embedding_model_pipelines import EmbeddingModelPipeline
from etl.models import OpenAiSettings, OutputConfig


class OpenAiEmbeddingModelPipeline(EmbeddingModelPipeline):  # type: ignore

    def __init__(
        self, *, openai_settings: OpenAiSettings, output_config: OutputConfig
    ) -> None:
        self.__openai_settings: OpenAiSettings = openai_settings
        self.__parsed_output_config: OutputConfig.Parsed = output_config.parse()

    def __create_embedding_model(self) -> Embeddings:
        """Create and return an OpenAI embedding model."""

        store = LocalFileStore(
            self.__parsed_output_config.directory_path
        )  # cache directory

        openai_embeddings_model = OpenAIEmbeddings(
            model=str(self.__openai_settings.embedding_model_name)
        )

        return CacheBackedEmbeddings.from_bytes_store(
            openai_embeddings_model, store, namespace=openai_embeddings_model.model
        )

    def create_embedding_store(self, documents: tuple[Document, ...]) -> FAISS:
        """Return an embedding store that was created with an OpenAI embedding model."""

        return FAISS.from_documents(
            documents=list(documents),
            embedding=self.__create_embedding_model(),
        )