from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from wikipedia_transform import WikipediaTransform
from wikipedia_transform.embedding_pipelines import OpenAiEmbeddingPipeline


def test_transform(
    session_mocker: MockFixture,
    wikipedia_transform: WikipediaTransform,
    skip_if_ci: None,
) -> None:
    """Test that WikipediaTransform.transform performs the necessary operations."""

    session_mocker.patch.object(FAISS, "from_documents", return_value=None)

    mock_open_ai_embedding_pipeline__create_embedding_store = (
        session_mocker.patch.object(
            OpenAiEmbeddingPipeline, "create_embedding_store", return_value=None
        )
    )

    wikipedia_transform.transform()

    mock_open_ai_embedding_pipeline__create_embedding_store.assert_called_once()
