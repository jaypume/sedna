# 参考自Plato.server
# 需要考虑是否把aggregator和server的定位区分开？


from abc import ABC, abstractmethod


class Aggregator(ABC):
    """The base class for federated learning servers."""

    def register_client(self, client_id, websocket):
        """Adding a newly arrived client to the list of clients."""

    def unregister_client(self, websocket):
        """Removing an existing client from the list of clients."""

    @abstractmethod
    def aggregate(self):
        """customized aggregate algorithm."""

    @abstractmethod
    def exit_check(self):
        """check if should exit federated learning job"""



class HitAggregator(ABC):
    """The base class for federated learning servers."""


    @abstractmethod
    def aggregate(self):
        # 客户端选择
        pass

    @abstractmethod
    def exit_check(self):
        """check if should exit federated learning job"""