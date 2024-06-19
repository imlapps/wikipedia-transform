from enum import Enum


class EnhancementType(str, Enum):
    """An enum of enhancement types for Wikipedia articles."""

    SUMMARY = "summary"