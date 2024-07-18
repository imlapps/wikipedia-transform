from pytest_mock import MockFixture
from langchain_community.vectorstores import FAISS
from etl.pipelines import AntiRecommendationRetrievalPipeline
from langchain.docstore.document import Document


def test_retrieve_documents(
    session_mocker: MockFixture,
    anti_recommendation_retrieval_pipeline: AntiRecommendationRetrievalPipeline,
    document_of_article_with_summary: Document,
) -> None:
    """Test that AntiRecommendationRetrievalPipeline.retrieve_documents successfully returns a tuple of Documents when given a Record key."""

    session_mocker.patch.object(
        FAISS,
        "similarity_search_with_score",
        return_value=((document_of_article_with_summary, 0.42),),
    )

    assert (
        anti_recommendation_retrieval_pipeline.retrieve_documents(
            record_key="Imlapps", k=1
        )[0][0]
        == document_of_article_with_summary
    )
