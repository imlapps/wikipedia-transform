from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.pipelines.open_ai_embedding_pipeline import (
    OpenAiEmbeddingPipeline,
)
from etl.resources import OpenAiSettings


def test_create_embedding_store(
    session_mocker: MockFixture,
    document_of_article_with_summary: Document,
    openai_embedding_model_pipeline: OpenAiEmbeddingPipeline,
) -> None:
    """Test that OpenAiEmbeddingPipeline.create_embedding_store invokes a method that is required to create an embedding store."""

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    openai_embedding_model_pipeline.create_embedding_store(
        documents=(document_of_article_with_summary,)
    )

    mock_faiss__from_documents.assert_called_once()


def test_create_embedding_model(
    openai_embedding_model_pipeline: OpenAiEmbeddingPipeline,
    openai_settings: OpenAiSettings,
) -> None:
    """Test that OpenAiEmbeddingPipeline._create_embedding_model returns an Embedding model that matches the parameters of OpenAiSettings."""

    assert (
        openai_embedding_model_pipeline._create_embedding_model().namespace
        == openai_settings.embedding_model_name
    )
