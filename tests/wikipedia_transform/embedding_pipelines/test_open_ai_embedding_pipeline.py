from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from wikipedia_transform.embedding_pipelines import OpenAiEmbeddingPipeline
from wikipedia_transform.models import wikipedia
from wikipedia_transform.models.types import RecordType


def test_create_embedding_store(
    session_mocker: MockFixture,
    tuple_of_article: tuple[wikipedia.Article],
    record_type: RecordType,
    open_ai_embedding_pipeline: OpenAiEmbeddingPipeline,
    skip_if_ci: None,
) -> None:
    """Test that OpenAiEmbeddingPipeline.create_embedding_store calls methods that a necessary to create an embedding store."""

    mock_cache_backed_embeddings__from_bytes_store = session_mocker.patch.object(
        CacheBackedEmbeddings, "from_bytes_store", return_value=None
    )
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    open_ai_embedding_pipeline.create_embedding_store(
        records=tuple_of_article, record_type=record_type
    )

    mock_cache_backed_embeddings__from_bytes_store.assert_called_once()
    mock_faiss__from_documents.assert_called_once()
