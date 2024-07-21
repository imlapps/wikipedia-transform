from dataclasses import dataclass
from typing import Annotated

from pydantic import Field

# String type for AntiRecommendationsByKeyTuple.
ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR = Annotated[
    str,
    Field(json_schema_extra={"min_length": 1, "strip_whitespace": True}),
]


@dataclass(frozen=True)
class AntiRecommendationsByKeyTuple:
    """
    A dataclass that holds a tuple of dictionaries containing anti-recommendations of a key.
    """

    anti_recommendations_by_key: tuple[
        dict[
            ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR,
            tuple[ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR, ...],
        ],
        ...,
    ]
