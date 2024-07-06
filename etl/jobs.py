from dagster import define_asset_job

from .assets import (
    wikipedia_articles_embeddings,
)


embedding_job = define_asset_job(
    "embedding_job", selection=["*" + wikipedia_articles_embeddings.key.path[0]]
)
