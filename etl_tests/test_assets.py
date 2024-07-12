import json

from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.assets import (
    documents_of_wikipedia_articles_with_summaries,
    wikipedia_articles_embedding_store,
    wikipedia_articles_from_storage,
    wikipedia_articles_with_summaries,
    wikipedia_articles_with_summaries_json_file,
)
from etl.models import DocumentTuple, RecordTuple, wikipedia
from etl.models.types import ModelResponse
from etl.resources import (
    InputDataFilesConfig,
    OpenAiPipelineConfig,
    OpenAiSettings,
    OutputConfig,
)


def test_wikipedia_articles_from_storage(
    input_data_files_config: InputDataFilesConfig,
) -> None:
    """Test that wikipedia_articles_from_storage successfully materializes a tuple of Wikipedia articles."""

    assert isinstance(
        wikipedia_articles_from_storage(input_data_files_config).records[0], wikipedia.Article  # type: ignore[attr-defined]
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
        wikipedia_articles_with_summaries(  # type: ignore[attr-defined]
            RecordTuple(records=tuple_of_articles_with_summaries),
            openai_pipeline_config,
        )
        .records[0]
        .model_dump(by_alias=True)["summary"]
        == article_with_summary.summary
    )


def test_wikipedia_articles_with_summaries_to_json(
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    output_config: OutputConfig,
    openai_settings: OpenAiSettings,  # noqa: ARG001
) -> None:
    """Test that wikipedia_articles_with_summaries_to_json writes articles to a JSON file."""

    wikipedia_articles_with_summaries_json_file(
        RecordTuple(records=tuple_of_articles_with_summaries), output_config
    )

    with output_config.parse().wikipedia_articles_with_summaries_file_path.open() as wikipedia_json_file:

        iter_tuples_of_articles_with_summaries = iter(tuple_of_articles_with_summaries)

        for wikipedia_json_line in wikipedia_json_file:
            wikipedia_json = json.loads(wikipedia_json_line)
            assert wikipedia.Article(**(wikipedia_json)) == next(
                iter_tuples_of_articles_with_summaries
            )


def test_documents_of_wikipedia_articles_with_summaries(
    openai_pipeline_config: OpenAiPipelineConfig,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    document_of_article_with_summary: Document,
) -> None:
    """Test that documents_of_wikipedia_articles_with_summaries successfully materializes a tuple of Documents."""

    assert (
        documents_of_wikipedia_articles_with_summaries(  # type: ignore[attr-defined]
            RecordTuple(records=tuple_of_articles_with_summaries),
            openai_pipeline_config,
        ).documents[0]
        == document_of_article_with_summary
    )


def test_wikipedia_articles_embeddings(
    session_mocker: MockFixture,
    openai_settings: OpenAiSettings,
    output_config: OutputConfig,
    faiss: FAISS,
    document_of_article_with_summary: Document,
) -> None:
    """Test that wikipedia_articles_embedding_store calls a method that is required to create an embedding store."""

    mock_faiss__from_documents = session_mocker.patch.object(
        FAISS, "from_documents", return_value=faiss
    )

    wikipedia_articles_embedding_store(
        DocumentTuple(documents=(document_of_article_with_summary,)),
        openai_settings,
        output_config,
    )

    mock_faiss__from_documents.assert_called_once()
