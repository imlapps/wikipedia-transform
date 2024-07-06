from pathlib import Path
from dagster import Definitions, load_assets_from_modules, EnvVar

from etl.models.types.enrichment_type import EnrichmentType
from etl.models.types.record_type import RecordType

from . import assets
from .jobs import embedding_job
from .resources import (
    InputDataFilesConfig,
    OpenAiPipelineConfig,
    OpenAiSettings,
    OutputConfig,
)


openai_settings = OpenAiSettings(openai_api_key=EnvVar("OPENAI_API_KEY").get_value(""))

definitions = Definitions(
    assets=load_assets_from_modules([assets]),
    jobs=[embedding_job],
    resources={
        "input_data_files_config": InputDataFilesConfig.from_env_vars(
            data_files_directory_path_default=Path(__file__).parent.absolute()
            / "data"
            / "input"
            / "data_files",
            data_file_names_default=("mini-wikipedia.output.txt",),
        ),
        "open_ai_settings": openai_settings,
        "openai_pipeline_config": OpenAiPipelineConfig(
            openai_settings=openai_settings,
            record_type=EnvVar("RECORD_TYPE").get_value(default=RecordType.WIKIPEDIA),
            enrichment_type=EnvVar("ENRICHMENT_TYPE").get_value(
                default=EnrichmentType.SUMMARY
            ),
        ),
        "output_config": OutputConfig.from_env_vars(
            output_directory_path_default=Path(__file__).parent.absolute()
            / "data"
            / "output"
        ),
    },
)
