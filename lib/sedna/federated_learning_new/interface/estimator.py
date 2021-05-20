# 当前的接口定义就是来自于Plato的，我们认可并且把这个接口定义当作一个标准
# 这个Trainer是继承自某个基类，还是说和增量学习/协同推理一样的？
# 每个抽象函数需要和其他特性一起讨论：1. 如果删掉Plato是否能使用； 2. 如果保留是否有必要？其他特性不用的话会有什么影响？
# Trainer是否有test，eval函数


from abc import ABC, abstractmethod


class Estimator(ABC):
    """Base class for all the trainers."""

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass




class HitEstimator(Estimator):
    """Base class for all the trainers."""

    @abstractmethod
    def train(self):
        # 数据蒸馏
        pass

    @abstractmethod
    def test(self):
        pass