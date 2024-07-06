from langchain.docstore.document import Document

from etl.models import DocumentTuple, wikipedia
from etl.models.types import EnrichmentType, RecordType


def test_document_tuple_from_records(
    document_of_article_with_summary: Document,
    tuple_of_articles_with_summaries: tuple[wikipedia.Article, ...],
    record_type: RecordType,
    enrichment_type: EnrichmentType,
) -> None:
    # """Test that DocumentTuple.from_records successfully creates and returns an object containing a tuple of Documents."""

    assert (
        DocumentTuple.from_records(
            records=tuple_of_articles_with_summaries,
            record_type=record_type,
            enrichment_type=enrichment_type,
        ).documents[0]
        == document_of_article_with_summary
    )
