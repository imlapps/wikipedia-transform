from dataclasses import dataclass
from typing import Annotated, Iterable
from pydantic import Field


# String type for AntiRecommendationsByKeyTuple.
ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR = Annotated[
    str,
    Field(json_schema_extra={"min_length": 1, "strip_whitespace": True}),
]


@dataclass(frozen=True)
class AntiRecommendationsByKeyTuple:
    anti_recommendations_by_key: tuple[
        dict[
            ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR,
            Iterable[ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR],
        ],
        ...,
    ]
