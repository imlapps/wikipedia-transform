from enum import Enum


class OpenAiRecordEnrichmentPipelineExceptionMsg(str, Enum):
    """An enum of Exception messages for an OpenAiRecordEnrichmentPipeline."""

    INVALID_RECORD_TYPE_MSG = "Invalid Record type."
    INVALID_ENRICHMENT_TYPE_MSG = "Invalid Enrichment type."
