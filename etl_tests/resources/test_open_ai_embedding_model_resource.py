from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.models import wikipedia


def test_create_embedding_store(
    session_mocker: MockFixture,
    tuple_of_article_with_summary: tuple[wikipedia.Article, ...],
    openai_embedding_model_resource,
) -> None:
    """Test that OpenAiEmbeddingPipeline.create_embedding_store calls methods that a necessary to create an embedding store."""

    # Mock CacheBackedEmbeddings.from_bytes_store
    mock_cache_backed_embeddings__from_bytes_store = session_mocker.patch.object(
        CacheBackedEmbeddings, "from_bytes_store", return_value=None
    )

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    openai_embedding_model_resource.create_embedding_store(
        records=tuple_of_article_with_summary
    )

    mock_cache_backed_embeddings__from_bytes_store.assert_called_once()

    mock_faiss__from_documents.assert_called_once()
