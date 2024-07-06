from langchain.schema.runnable import RunnableSequence
from pytest_mock import MockFixture

from etl.pipelines import OpenAiRecordEnrichmentPipeline
from etl.models import wikipedia
from etl.models.types import ModelResponse


def test_enrich_records(
    session_mocker: MockFixture,
    openai_generative_model_pipeline: OpenAiRecordEnrichmentPipeline,
    article: wikipedia.Article,
    openai_model_response: ModelResponse,
    article_with_summary: wikipedia.Article,
) -> None:
    """Test that OpenAiGenerativeModelPipeline.enrich_records returns enriched Records."""

    # Mock RunnableSequence.invoke and return a ModelResponse
    session_mocker.patch.object(
        RunnableSequence, "invoke", return_value=openai_model_response
    )

    assert (
        openai_generative_model_pipeline.enrich_record(record=article).model_dump(
            by_alias=True
        )["summary"]
        == article_with_summary.summary
    )
