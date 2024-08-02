from pyoxigraph import Store
from etl.models.types import AntiRecommendationKey, RecordKey
from etl.models.types.iri import Iri
from etl.pipelines import ArkgBuilderPipeline


def test_construct_graph(
    anti_recommendation_graph: tuple[
        tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...
    ],
    record_key: RecordKey,
    base_iri: Iri,
    anti_recommendation_key: AntiRecommendationKey,
) -> None:
    """Test that ArKgBuilderPipeline.construct_graph returns a RDF Store."""

    store: Store = ArkgBuilderPipeline(base_iri=base_iri).construct_graph(
        anti_recommendation_graph
    )
    for binding in store.query(
        query=f"SELECT ?title WHERE {{ <{record_key}> <hasAntiRecommendation> ?anti_recommendation {{?anti_recommendation <hasTitle> ?title}}}}",
        base_iri=base_iri,
    ):
        assert binding["title"].value == anti_recommendation_key
