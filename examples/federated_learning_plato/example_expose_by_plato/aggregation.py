import os

os.environ['config_file'] = 'examples/demo/server.yml'

from plato.servers import fedavg
from .network import model, Trainer


class Myserver(fedavg.Server):
    def __init__(self, model=None, trainer=None):
        super().__init__(model, trainer)


def main():
    trainer = Trainer(model=model)
    server = Myserver(model=model, trainer=trainer)
    server.run()


if __name__ == "__main__":
    main()
