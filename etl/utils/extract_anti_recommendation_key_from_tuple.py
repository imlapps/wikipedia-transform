from etl.models.types.record_key import RecordKey
from langchain.docstore.document import Document


def extract_anti_recommendation_key_from_tuple(
    *,
    anti_recommendation_and_similarity_score_tuple: tuple[Document, float],
    record_key: RecordKey,
) -> RecordKey:
    """
    Extract and return the key of an anti-recommendation from a Document.

    An anti-recommendation key is returned if only it is different from record_key.
    """

    anti_recommendation_record_key = anti_recommendation_and_similarity_score_tuple[
        0
    ].metadata["source"][len("https://en.wikipedia.org/wiki/") :]

    if anti_recommendation_record_key != record_key:
        return anti_recommendation_record_key
