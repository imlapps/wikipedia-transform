import json

from dagster import Definitions, EnvVar, FilesystemIOManager, load_assets_from_modules

from etl.models import OpenAiResourceParams, OpenAiSettings
from etl.models.types import EnrichmentType, RecordType
from etl.resources import (
    OpenAiEmbeddingModelResource,
    OpenAiGenerativeModelResource,
    WikipediaReaderResource,
)

from . import assets
from .assets import embedding_job

all_assets = load_assets_from_modules([assets])
all_jobs = [embedding_job]

openai_resource_params_dict = {
    "openai_resource_params": OpenAiResourceParams(
        openai_settings=OpenAiSettings(
            openai_api_key=EnvVar("OPENAI_API_KEY").get_value("")
        ),
        record_type=EnvVar("RECORD_TYPE").get_value(default=RecordType.WIKIPEDIA),
        enrichment_type=EnvVar("ENRICHMENT_TYPE").get_value(
            default=EnrichmentType.SUMMARY
        ),
    )
}

defs = Definitions(
    assets=all_assets,
    jobs=all_jobs,
    resources={
        "wikipedia_reader_resource": WikipediaReaderResource(
            data_file_names=json.loads(EnvVar("DATA_FILE_NAMES").get_value(default="[]"))  # type: ignore
        ),
        "openai_generative_model_resource": OpenAiGenerativeModelResource(
            **openai_resource_params_dict
        ),
        "openai_embedding_model_resource": OpenAiEmbeddingModelResource(
            **openai_resource_params_dict
        ),
        "io_manager": FilesystemIOManager(),
    },
)
