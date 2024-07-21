from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from pytest_mock import MockFixture

from etl.models.types import RecordKey
from etl.pipelines import AntiRecommendationRetrievalPipeline


def test_retrieve_documents(
    session_mocker: MockFixture,
    anti_recommendation_retrieval_pipeline: AntiRecommendationRetrievalPipeline,
    document_of_anti_recommendation_article: Document,
    record_key: RecordKey,
) -> None:
    """Test that AntiRecommendationRetrievalPipeline.retrieve_documents returns a tuple of Document-float tuple pairs when given a Record key."""

    session_mocker.patch.object(
        FAISS,
        "similarity_search_with_score",
        return_value=((document_of_anti_recommendation_article, 0.82),),
    )

    assert (
        anti_recommendation_retrieval_pipeline.retrieve_documents(
            record_key=record_key, k=1
        )[0][0]
        == document_of_anti_recommendation_article
    )
