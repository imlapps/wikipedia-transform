from enum import Enum


class Predicate(str, Enum):
    """An enum of predicates for RDF Triples."""

    HAS_TITLE = "hasTitle"
    HAS_URL = "hasURL"
    HAS_ANTI_RECOMMENDATION = "hasAntiRecommendation"
