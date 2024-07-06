from enum import Enum


class DocumentTupleExceptionMsg(str, Enum):
    """An enum of Exception messages for a DocumentTuple."""

    INVALID_RECORD_TYPE_MSG = "Invalid Record type."
    INVALID_ENRICHMENT_TYPE_MSG = "Invalid Enrichment type."
