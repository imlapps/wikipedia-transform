from enum import Enum


class EnrichmentType(str, Enum):
    """An enum of enrichment types for Wikipedia articles."""

    SUMMARY = "summary"