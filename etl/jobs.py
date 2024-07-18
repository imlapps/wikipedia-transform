from dagster import define_asset_job

from .assets import (
    retrievals_of_wikipedia_anti_recommendations,
    wikipedia_articles_embedding_store,
)

embedding_job = define_asset_job(
    "embedding_job", selection=["*" + wikipedia_articles_embedding_store.key.path[0]]
)
retrieval_job = define_asset_job(
    "retrieval_job",
    selection=["*" + retrievals_of_wikipedia_anti_recommendations.key.path[0]],
)
