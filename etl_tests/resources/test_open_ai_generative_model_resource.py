from langchain.schema.runnable import RunnableSequence
from pytest_mock import MockFixture

from etl.resources import OpenAiGenerativeModelResource
from etl.models import wikipedia
from etl.models.types import ModelResponse

def test_enrich_records(
    session_mocker: MockFixture,
    open_ai_generative_model_resource: OpenAiGenerativeModelResource,
    article: wikipedia.Article,
    open_ai_model_response: ModelResponse,

    article_with_summary: wikipedia.Article,
) -> None:
    """Test that OpenAiPipeline.enhance_records yields enhanced Records."""

    # Mock RunnableSequence.invoke and return a ModelResponse
    session_mocker.patch.object(
        RunnableSequence, "invoke", return_value=open_ai_model_response
    )

    assert (
        open_ai_generative_model_resource.enrich_record(
                    record=article
                ).summary
        == article_with_summary.summary
    )