from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.embedding_model_pipelines.open_ai_embedding_model_pipeline import (
    OpenAiEmbeddingModelPipeline,
)


def test_create_embedding_store(
    session_mocker: MockFixture,
    document_of_article_with_summary: Document,
    openai_embedding_model_pipeline: OpenAiEmbeddingModelPipeline,
) -> None:
    """Test that OpenAiEmbeddingPipeline.create_embedding_store calls methods that are needed to create an embedding store."""

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    openai_embedding_model_pipeline.create_embedding_store(
        documents=(document_of_article_with_summary,)
    )

    mock_faiss__from_documents.assert_called_once()
