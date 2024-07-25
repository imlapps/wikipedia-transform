from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.models.anti_recommendation import AntiRecommendation
from etl.models.types import RecordKey
from etl.pipelines import AntiRecommendationRetrievalPipeline


def test_retrieve_documents(
    session_mocker: MockFixture,
    anti_recommendation_retrieval_pipeline: AntiRecommendationRetrievalPipeline,
    anti_recommendation: AntiRecommendation,
    anti_recommendation_record_key: RecordKey,
    record_key: RecordKey,
) -> None:
    """Test that AntiRecommendationRetrievalPipeline.retrieve_documents returns a tuple of AntiRecommendations when given a Record key."""

    session_mocker.patch.object(
        FAISS,
        "similarity_search_with_score",
        return_value=(
            (
                anti_recommendation.document,
                anti_recommendation.similarity_score,
            ),
        ),
    )

    assert (
        anti_recommendation_retrieval_pipeline.retrieve_documents(
            record_key=record_key, k=1
        )[0].key
        == anti_recommendation_record_key
    )
