import json

from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.assets import (
    documents_of_wikipedia_articles_with_summaries,
    wikipedia_articles_embeddings,
    wikipedia_articles_from_storage,
    wikipedia_articles_with_summaries,
    wikipedia_articles_with_summaries_json_file,
)
from etl.models import (
    OpenAiPipelineConfig,
    OpenAiSettings,
    output_config_from_env_vars,
    wikipedia,
)
from etl.models.types import ModelResponse


def test_wikipedia_articles_from_storage() -> None:
    """Test that wikipedia_articles_from_storage successfully materializes a tuple of Wikipedia articles."""

    assert isinstance(
        wikipedia_articles_from_storage()[0], wikipedia.Article  # type: ignore[index]
    )


def test_wikipedia_articles_with_summaries(
    session_mocker: MockFixture,
    openai_pipeline_config: OpenAiPipelineConfig,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    article_with_summary: wikipedia.Article,
    openai_model_response: ModelResponse,
) -> None:
    """Test that wikipedia_articles_with_summaries succesfully materializes a tuple of Wikipedia articles with summaries."""

    # Mock RunnableSequence.invoke and return a ModelResponse
    session_mocker.patch.object(
        RunnableSequence, "invoke", return_value=openai_model_response
    )

    assert (
        wikipedia_articles_with_summaries(  # type: ignore[index]
            openai_pipeline_config, tuple_of_articles_with_summaries
        )[0].model_dump(by_alias=True)["summary"]
        == article_with_summary.summary
    )


def test_wikipedia_articles_with_summaries_to_json(
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
) -> None:
    """Test that wikipedia_articles_with_summaries_to_json writes articles to a JSON file"""

    wikipedia_articles_with_summaries_json_file(tuple_of_articles_with_summaries)

    wikipedia_json_file_path = (
        output_config_from_env_vars.parse().directory_path
        / "wikipedia_articles_with_summaries.txt"
    )
    with wikipedia_json_file_path.open() as wikipedia_json_file:

        iter_tuples_of_articles_with_summaries = iter(tuple_of_articles_with_summaries)

        for wikipedia_json_line in wikipedia_json_file:
            wikipedia_json = json.loads(wikipedia_json_line)
            assert wikipedia.Article(**(wikipedia_json)) == next(
                iter_tuples_of_articles_with_summaries
            )


def test_documents_of_wikipedia_articles_with_summaries(
    openai_settings: OpenAiPipelineConfig,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    document_of_article_with_summary: Document,
) -> None:
    """Test that documents_of_wikipedia_articles_with_summaries successfully materializes a tuple of Wikipedia documents."""

    assert (
        documents_of_wikipedia_articles_with_summaries(  # type: ignore[index]
            tuple_of_articles_with_summaries, openai_settings
        )[0]
        == document_of_article_with_summary
    )


def test_wikipedia_articles_embeddings(
    session_mocker: MockFixture,
    openai_settings: OpenAiSettings,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
) -> None:
    """Test that wikipedia_articles_embeddings calls the methods that are needed to materialize an embeddings store."""

    # Mock FAISS.from_documents
    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=None
    )

    wikipedia_articles_embeddings(tuple_of_articles_with_summaries, openai_settings)

    mock_faiss__from_documents.assert_called_once()
