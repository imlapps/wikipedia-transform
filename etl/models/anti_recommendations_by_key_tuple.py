from dataclasses import dataclass
from typing import Annotated
from pydantic import Field
from langchain_community.docstore.document import Document

ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR = Annotated[
    str,
    Field(json_schema_extra={"min_length": 1, "strip_whitespace": True}),
]


@dataclass(frozen=True)
class AntiRecommendationsByKeyTuple:
    anti_recommendations_by_key: tuple[
        dict[
            ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR,
            tuple[
                ANTI_RECOMMENDATIONS_BY_KEY_TUPLE_STR,
                ...,
            ],
        ],
        ...,
    ]
