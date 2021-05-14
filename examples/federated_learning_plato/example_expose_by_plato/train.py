import asyncio
import os

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

os.environ['config_file'] = 'examples/demo/client.yml'

from plato.clients import simple
from plato.datasources import base
from plato.config import Config

from .network import model, Trainer


class DataSource(base.DataSource):
    """A custom datasource with custom training and validation
       datasets.
    """

    def __init__(self):  #   plato,sedna 都是来自于pytorch/tensorflow原生的dataset对象，不冲突。
        # 用户使用自己
        # 如果dataset有通用的方法，比如image2numpy，sedna.dataset是否要规范输出为trainset=x,y?
        super().__init__()

        self.trainset = MNIST("./data",
                              train=True,
                              download=True,
                              transform=ToTensor())
        self.testset = MNIST("./data",
                             train=False,
                             download=True,
                             transform=ToTensor())


class Myclient(simple.Client):
    def __init__(self, model=None, datasource=None, trainer=None):
        super().__init__(model, datasource, trainer)


def main():
    Config().args.id = int(Config().args.id)
    Config().args.port = int(Config().args.port)

    loop = asyncio.get_event_loop()
    coroutines = []

    datasource = DataSource()
    trainer = Trainer(model=model)
    client = Myclient(model=model, datasource=datasource, trainer=trainer, ) # 用户看不到，client这边暴露上传下载权重的接口，可能是websocket的，sedna来实现。
    client.configure()
    client.run()
    # coroutines.append(client.start_client())
    # loop.run_until_complete(asyncio.gather(*coroutines))


if __name__ == "__main__":
    main()
