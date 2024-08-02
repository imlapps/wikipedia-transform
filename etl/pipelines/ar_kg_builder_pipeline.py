from pyoxigraph import Literal, NamedNode, Quad, Store

from etl.models import BASE_WIKIPEDIA_URL
from etl.models.types import AntiRecommendationKey, Iri, Predicate, RecordKey
from etl.pipelines.kg_builder_pipeline import KgBuilderPipeline


class ArKgBuilderPipeline(KgBuilderPipeline):
    """
    A concrete implementation of KgBuilderPipeline.

    Constructs a RDF store from a tuple of RDF graphs.
    """

    def __init__(self, base_iri: Iri) -> None:
        self.__base_iri = base_iri
        self.__store = Store()

    def __add_title_quad_to_store(self, record_key: RecordKey) -> None:
        self.__store.add(
            Quad(
                NamedNode(self.__base_iri + record_key.replace(" ", "_")),
                NamedNode(self.__base_iri + Predicate.HAS_TITLE.value),
                Literal(record_key),
            )
        )

    def __add_url_quad_to_store(self, record_key: RecordKey) -> None:
        self.__store.add(
            Quad(
                NamedNode(self.__base_iri + record_key.replace(" ", "_")),
                NamedNode(self.__base_iri + Predicate.HAS_URL.value),
                Literal(BASE_WIKIPEDIA_URL + record_key),
            )
        )

    def __add_anti_recommendation_quad_to_store(
        self,
        anti_recommendation_graph: tuple[RecordKey, tuple[AntiRecommendationKey, ...]],
    ) -> None:
        for anti_recommendation_key in anti_recommendation_graph[1]:
            self.__store.add(
                Quad(
                    NamedNode(
                        self.__base_iri + anti_recommendation_graph[0].replace(" ", "_")
                    ),
                    NamedNode(
                        self.__base_iri + Predicate.HAS_ANTI_RECOMMENDATION.value
                    ),
                    NamedNode(
                        self.__base_iri + anti_recommendation_key.replace(" ", "_")
                    ),
                )
            )

    def construct_graph(
        self,
        graphs: tuple[tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...],
    ) -> Store:

        for graph in graphs:
            self.__add_title_quad_to_store(graph[0])
            self.__add_url_quad_to_store(graph[0])
            self.__add_anti_recommendation_quad_to_store(graph)

        return self.__store
