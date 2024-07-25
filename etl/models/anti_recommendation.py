from typing import NamedTuple

from langchain.docstore.document import Document

from etl.models.types import RecordKey


class AntiRecommendation(NamedTuple):
    """
    A NamedTuple that contains data on an anti-recommendation.

    `key` is the name of a Record that was converted into a Document. 
    `document` is the Document representation of a Record that was retrieved from a VectorStore.
    `similarity_score` is the similarity score of the Document that was retrieved from a VectorStore.
    """
    key: RecordKey
    document: Document
    similarity_score: float
