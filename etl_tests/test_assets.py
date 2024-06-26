from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS

from pytest_mock import MockFixture

from etl.assets import (
    wikipedia_articles_embeddings,
    wikipedia_articles_from_storage,
    wikipedia_articles_with_summaries,
)

from etl.models import wikipedia, OpenAiPipelineConfig, OpenAiSettings
from etl.models.types import ModelResponse


def test_wikipedia_articles_from_storage() -> None:
    """Test that wikipedia_articles_from_storage successfully materializes a tuple of Wikipedia articles."""

    assert isinstance(wikipedia_articles_from_storage()[0], wikipedia.Article)


def test_wikipedia_articles_with_summaries(
    session_mocker: MockFixture,
    openai_pipeline_config: OpenAiPipelineConfig,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    article_with_summary: wikipedia.Article,
    openai_model_response: ModelResponse,
):
    """Test that wikipedia_articles_with_summaries succesfully materializes a tuple of Wikipedia articles with summaries."""

    # Mock RunnableSequence.invoke and return a ModelResponse
    session_mocker.patch.object(
        RunnableSequence, "invoke", return_value=openai_model_response
    )

    assert (
        wikipedia_articles_with_summaries(
            openai_pipeline_config, tuple_of_articles_with_summaries
        )[0].summary
        == article_with_summary.summary
    )


def test_wikipedia_articles_embeddings(
    session_mocker: MockFixture,
    openai_settings: OpenAiSettings,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
):
    """Test that wikipedia_articles_embeddings calls the methods that are needed to materialize an embeddings store."""

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    wikipedia_articles_embeddings(tuple_of_articles_with_summaries, openai_settings)

    mock_faiss__from_documents.assert_called_once()
