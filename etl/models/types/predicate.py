from enum import Enum


class Predicate(str, Enum):
    HAS_TITLE = "hasTitle"
    HAS_URL = "hasURL"
    HAS_ANTI_RECOMMENDATION = "hasAntiRecommendation"
