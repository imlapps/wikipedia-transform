from dataclasses import dataclass

from etl.models.types import AntiRecommendationKey, RecordKey


@dataclass(frozen=True)
class AntiRecommendationGraphTuple:
    """
    A dataclass that holds a tuple of anti-recommendation graphs.

    An anti-recommendation graph is defined by a (subject, objects) tuple structure.
    Subject is a Record key, and objects are keys of subject's anti-recommendations.
    """

    anti_recommendation_graphs: tuple[
        tuple[
            RecordKey,
            tuple[AntiRecommendationKey, ...],
        ],
        ...,
    ]
