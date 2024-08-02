import json

from langchain.docstore.document import Document
from langchain.schema.runnable import RunnableSequence
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.assets import (
    documents_of_wikipedia_articles_with_summaries,
    wikipedia_anti_recommendations,
    wikipedia_anti_recommendations_json_file,
    wikipedia_arkg,
    wikipedia_articles_embedding_store,
    wikipedia_articles_from_storage,
    wikipedia_articles_with_summaries,
    wikipedia_articles_with_summaries_json_file,
)
from etl.models import (
    AntiRecommendationGraphTuple,
    DocumentTuple,
    RecordTuple,
    wikipedia,
)
from etl.models.types import AntiRecommendationKey, ModelResponse, RecordKey
from etl.pipelines import ArkgBuilderPipeline
from etl.resources import (
    InputConfig,
    OpenaiPipelineConfig,
    OpenaiSettings,
    OutputConfig,
)


def test_wikipedia_articles_from_storage(
    input_config: InputConfig,
) -> None:
    """Test that wikipedia_articles_from_storage successfully materializes a tuple of Wikipedia articles."""

    assert isinstance(
        wikipedia_articles_from_storage(input_config).records[0], wikipedia.Article  # type: ignore[attr-defined]
    )


def test_wikipedia_articles_with_summaries(
    session_mocker: MockFixture,
    openai_pipeline_config: OpenaiPipelineConfig,
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


def test_wikipedia_articles_with_summaries_json_file(
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    output_config: OutputConfig,
    openai_settings: OpenaiSettings,  # noqa: ARG001
) -> None:
    """Test that wikipedia_articles_with_summaries_json_file writes articles to a JSON file."""

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
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    document_of_article_with_summary: Document,
) -> None:
    """Test that documents_of_wikipedia_articles_with_summaries successfully materializes a tuple of Documents."""

    assert (
        documents_of_wikipedia_articles_with_summaries(  # type: ignore[attr-defined]
            RecordTuple(records=tuple_of_articles_with_summaries),
        ).documents[0]
        == document_of_article_with_summary
    )


def test_wikipedia_articles_embeddings(  # noqa: PLR0913
    session_mocker: MockFixture,
    openai_settings: OpenaiSettings,
    input_config: InputConfig,
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
        input_config,
        output_config,
    )

    mock_faiss__from_documents.assert_called_once()


def test_wikipedia_anti_recommendations(  # noqa: PLR0913
    openai_settings: OpenaiSettings,
    input_config: InputConfig,
    output_config: OutputConfig,
    document_of_article_with_summary: Document,
    article: wikipedia.Article,
    anti_recommendation_graph: tuple[
        tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...
    ],
) -> None:
    """Test that wikipedia_anti_recommendations successfully returns anti_recommendation_graphs."""

    assert (
        wikipedia_anti_recommendations(  # type: ignore[attr-defined]
            RecordTuple(records=(article,)),
            DocumentTuple(documents=(document_of_article_with_summary,)),
            openai_settings,
            input_config,
            output_config,
        ).anti_recommendation_graphs[0]
        == anti_recommendation_graph[0]
    )


def test_wikipedia_anti_recommendations_json_file(
    output_config: OutputConfig,
    anti_recommendation_graph: tuple[
        tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...
    ],
) -> None:
    """Test that wikipedia_anti_recommendations_json_file successfully writes an anti_recommendation_graph to a JSON file."""

    wikipedia_anti_recommendations_json_file(
        AntiRecommendationGraphTuple(
            anti_recommendation_graphs=anti_recommendation_graph
        ),
        output_config,
    )

    with output_config.parse().wikipedia_anti_recommendations_file_path.open() as wikipedia_anti_recommendations_file:

        for wikipedia_json_line in wikipedia_anti_recommendations_file:

            assert (
                json.loads(wikipedia_json_line)[0][-1]
                == anti_recommendation_graph[0][0][-1]
            )


def test_wikipedia_arkg(
    session_mocker: MockFixture,
    anti_recommendation_graph: tuple[
        tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...
    ],
    input_config: InputConfig,
) -> None:
    """Test that wikipedia_arkg calls a method required to build an Anti-Recommendation Knowledge Graph."""

    mock_arkgbuilderpipeline__construct_graph = session_mocker.patch.object(
        ArkgBuilderPipeline, "construct_graph", return_value=None
    )

    wikipedia_arkg(
        AntiRecommendationGraphTuple(
            anti_recommendation_graphs=anti_recommendation_graph
        ),
        input_config,
    )

    mock_arkgbuilderpipeline__construct_graph.assert_called_once()
