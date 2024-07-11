from dataclasses import dataclass
from typing import Self

from langchain.docstore.document import Document

from etl.models.record import Record
from etl.models.types import EnrichmentType


@dataclass(frozen=True)
class DocumentTuple:
    """A dataclass that holds a tuple of Documents."""

    documents: tuple[Document, ...]

    @classmethod
    def from_records(
        cls,
        *,
        records: tuple[Record, ...],
        enrichment_type: EnrichmentType,
    ) -> Self:
        """
        Convert Records into Documents and return an instance of DocumentTuple.

        Use enrichment_type to determine the content of a Document.
        """

        match enrichment_type:
            case EnrichmentType.SUMMARY:
                return cls(
                    documents=tuple(
                        Document(
                            page_content=str(record.model_dump().get("summary")),
                            metadata={
                                "source": "https://en.wikipedia.org/wiki/{record.key}"
                            },
                        )
                        for record in records
                    )
                )
            case _:
                raise ValueError(
                    f" {enrichment_type} is an invalid WikipediaTransform enrichment type."
                )
