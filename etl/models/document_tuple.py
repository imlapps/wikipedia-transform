from dataclasses import dataclass

from langchain.docstore.document import Document


@dataclass(frozen=True)
class DocumentTuple:
    """A dataclass that holds a tuple of Documents."""

    documents: tuple[Document, ...]
