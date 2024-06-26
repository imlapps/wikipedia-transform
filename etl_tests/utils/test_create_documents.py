from langchain.docstore.document import Document
from etl.models import wikipedia
from etl.models.types import RecordType, EnrichmentType
from etl.utils import create_documents


def test_create_documents(
    document_of_article_with_summary: Document,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    record_type: RecordType,
    enrichment_type: EnrichmentType,
) -> None:
    """Test that create_documents successfully creates and returns a Document."""

    assert (
        create_documents(
            records=tuple_of_articles_with_summaries,
            record_type=record_type,
            enrichment_type=enrichment_type,
        )[0]
        == document_of_article_with_summary
    )
