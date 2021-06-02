from .base import Aggregator


class ClientChooseAggregator(Aggregator):
    """The base class for federated learning servers."""

    def aggregate(self):
        # 客户端选择
        pass

    def exit_check(self):
        """check if should exit federated learning job"""
