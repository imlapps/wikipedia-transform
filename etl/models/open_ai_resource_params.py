from dagster import Config
from etl.models import OpenAiSettings
from etl.models.types import RecordType, EnrichmentType

class OpenAiResourceParams(Config):
      openai_settings : OpenAiSettings
      record_type: RecordType
      enrichment_type: EnrichmentType