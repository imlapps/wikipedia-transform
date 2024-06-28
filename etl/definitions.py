from dagster import Definitions, load_assets_from_modules

from . import assets
from .jobs import embedding_job

all_assets = load_assets_from_modules([assets])
all_jobs = [embedding_job]


definitions = Definitions(assets=all_assets, jobs=all_jobs)
