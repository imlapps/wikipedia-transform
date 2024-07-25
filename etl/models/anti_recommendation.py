from typing import NamedTuple

from langchain.docstore.document import Document

from etl.models.types import RecordKey


class AntiRecommendation(NamedTuple):
    key: RecordKey
    document: Document
    similarity_score: float
