from langchain.docstore.document import Document

from etl.models import Record
from etl.models.types import EnrichmentType, RecordType


def create_documents(
    *,
    records: tuple[Record, ...],
    record_type: RecordType,
    enrichment_type: EnrichmentType,
) -> tuple[Document, ...]:
    """Convert Records into Documents and return them."""

    if record_type == RecordType.WIKIPEDIA:
        match enrichment_type:
            case EnrichmentType.SUMMARY:
                return tuple(
                    Document(
                        page_content=str(record.model_dump().get("summary")),
                        metadata={
                            "source": "https://en.wikipedia.org/wiki/{record.key}"
                        },
                    )
                    for record in records
                )
