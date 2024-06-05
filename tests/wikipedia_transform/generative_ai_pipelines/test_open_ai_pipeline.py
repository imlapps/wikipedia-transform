from langchain.schema.runnable import RunnableSequence
from pytest_mock import MockFixture

from wikipedia_transform.generative_ai_pipelines import OpenAiPipeline
from wikipedia_transform.models import wikipedia
from wikipedia_transform.models.types import EnhancementType, ModelResponse, RecordType


def test_enhance_records(
    session_mocker: MockFixture,
    open_ai_pipeline: OpenAiPipeline,
    article: wikipedia.Article,
    record_type: RecordType,
    enhancement_type: EnhancementType,
    open_ai_model_response: ModelResponse,
    article_with_summary: wikipedia.ArticleSummary,
) -> None:
    """Test that OpenAiPipeline.enhance_records yields enhanced Records."""

    # Mock RunnableSequence.invoke and return a ModelResponse
    session_mocker.patch.object(
        RunnableSequence, "invoke", return_value=open_ai_model_response
    )

    assert (
        next(
            iter(
                open_ai_pipeline.enhance_record(
                    record=article,
                    record_type=record_type,
                    enhancement_type=enhancement_type,
                )
            )
        )
        == article_with_summary
    )
