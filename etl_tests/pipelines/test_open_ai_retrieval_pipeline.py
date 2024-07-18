from pytest_mock import MockFixture
from langchain_core.retrievers import BaseRetriever
from etl.pipelines import OpenAiRetrievalPipeline
from langchain.docstore.document import Document


def test_retrieve_documents(
    session_mocker: MockFixture,
    open_ai_retrieval_pipeline: OpenAiRetrievalPipeline,
    document_of_article_with_summary: Document,
) -> None:
    """Test that OpenAiRetrievalPipeline.retrieve_documents successfully returns a tuple of Documents when given a Record key."""

    session_mocker.patch.object(
        BaseRetriever, "invoke", return_value=(document_of_article_with_summary,)
    )

    assert (
        open_ai_retrieval_pipeline.retrieve_documents(
            record_key="Imlapps", number_of_documents_to_retrieve=1
        )[0]
        == document_of_article_with_summary
    )
