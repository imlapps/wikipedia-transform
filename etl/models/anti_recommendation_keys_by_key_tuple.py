from dataclasses import dataclass

from etl.models.types import RecordKey


@dataclass(frozen=True)
class AntiRecommendationKeysByKeyTuple:
    """
    A dataclass that holds a tuple of dictionaries.

    The key of a dictionary is a Record key, and its values are the keys of the Record's anti-recommendations.
    """

    anti_recommendation_keys_by_key: tuple[
        dict[
            RecordKey,
            tuple[RecordKey, ...],
        ],
        ...,
    ]
