from dagster import define_asset_job

from .assets import wikipedia_articles_embedding_store

embedding_job = define_asset_job(
    "embedding_job", selection=["*" + wikipedia_articles_embedding_store.key.path[0]]
)
