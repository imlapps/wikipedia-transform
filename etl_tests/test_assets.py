from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.assets import (wikipedia_articles_embeddings,
                        wikipedia_articles_from_storage,
                        wikipedia_articles_with_summaries)
from etl.models import wikipedia
from etl.models.types import ModelResponse
from etl.resources import (OpenAiEmbeddingModelResource,
                           OpenAiGenerativeModelResource,
                           WikipediaReaderResource)


def test_wikipedia_articles_from_storage(
    wikipedia_reader_resource: WikipediaReaderResource,
) -> None:
    """Test that wikipedia_articles_from_storage successfully materializes a tuple of Wikipedia articles."""

    assert isinstance(
        wikipedia_articles_from_storage(wikipedia_reader_resource)[0], wikipedia.Article
    )


def test_wikipedia_articles_with_summaries(
    session_mocker: MockFixture,
    tuple_of_article_with_summary: tuple[wikipedia.Article, ...],
    openai_generative_model_resource: OpenAiGenerativeModelResource,
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
            tuple_of_article_with_summary, openai_generative_model_resource
        )[0].summary
        == article_with_summary.summary
    )


def test_wikipedia_articles_embeddings(
    session_mocker: MockFixture,
    tuple_of_article_with_summary: tuple[wikipedia.Article, ...],
    openai_embedding_model_resource: OpenAiEmbeddingModelResource,
):
    """Test that wikipedia_articles_embeddings calls the methods needed to materialize an embeddings store."""

    # Mock CacheBackedEmbeddings.from_bytes_store
    mock_cache_backed_embeddings__from_bytes_store = session_mocker.patch.object(
        CacheBackedEmbeddings, "from_bytes_store", return_value=None
    )

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    wikipedia_articles_embeddings(
        tuple_of_article_with_summary, openai_embedding_model_resource
    )

    mock_cache_backed_embeddings__from_bytes_store.assert_called_once()

    mock_faiss__from_documents.assert_called_once()
