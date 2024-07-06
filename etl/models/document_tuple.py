from dataclasses import dataclass
from typing import Self

from langchain.docstore.document import Document

from etl.models.record import Record
from etl.models.types import EnrichmentType, RecordType, DocumentTupleExceptionMsg


@dataclass(frozen=True)
class DocumentTuple:
    """A dataclass that holds a tuple of Documents."""

    documents: tuple[Document, ...]

    @classmethod
    def from_records(
        cls,
        *,
        records: tuple[Record, ...],
        record_type: RecordType,
        enrichment_type: EnrichmentType,
    ) -> Self:
        """
        Convert Records into Documents and return an instance of DocumentTuple.

        Use record_type and enrichment_type to determine the content of a Document.
        """
        match record_type:
            case RecordType.WIKIPEDIA:
                match enrichment_type:
                    case EnrichmentType.SUMMARY:
                        return cls(
                            documents=tuple(
                                Document(
                                    page_content=str(
                                        record.model_dump().get("summary")
                                    ),
                                    metadata={
                                        "source": "https://en.wikipedia.org/wiki/{record.key}"
                                    },
                                )
                                for record in records
                            )
                        )
                    case _:

                        raise ValueError(
                            DocumentTupleExceptionMsg.INVALID_ENRICHMENT_TYPE_MSG
                        )
            case _:
                raise ValueError(DocumentTupleExceptionMsg.INVALID_RECORD_TYPE_MSG)
