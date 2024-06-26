import json
from pathlib import Path

from etl.models import OpenAiResourceParams, OpenAiSettings
from etl.models.types import EnrichmentType, RecordType

from . import assets
from .jobs import embedding_job

from dagster import Definitions, EnvVar, FilesystemIOManager, load_assets_from_modules

all_assets = load_assets_from_modules([assets])
all_jobs = [embedding_job]


definitions = Definitions(assets=all_assets, jobs=all_jobs)
