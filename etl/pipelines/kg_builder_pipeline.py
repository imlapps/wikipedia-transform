from abc import ABC, abstractmethod

from pyoxigraph import Store

from etl.models.types import AntiRecommendationKey, RecordKey


class KgBuilderPipeline(ABC):
    """An interface to build pipelines that construct knowledge graphs."""

    @abstractmethod
    def construct_graph(
        self,
        anti_recommendation_graphs: tuple[
            tuple[RecordKey, tuple[AntiRecommendationKey, ...]], ...
        ],
    ) -> Store:
        """Return a RDF Store populated with anti_recommendation_graphs."""
