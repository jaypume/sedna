import os
import numpy as np
import keras.preprocessing.image as img_preprocessing
from interface import NetWork

from sedna.common.config import Context
from sedna.algorithms.data_process import TxtDataParse

from sedna.core.federated_learning import FederatedLearning
from .estimator import MyEstimator


def image_process(line):
    file_path, label = line.split(',')
    original_dataset_url = Context.get_parameters('original_dataset_url')
    root_path = os.path.dirname(original_dataset_url)
    file_path = os.path.join(root_path, file_path)
    img = img_preprocessing.load_img(file_path).resize((128, 128))
    data = img_preprocessing.img_to_array(img) / 255.0
    label = [0, 1] if int(label) == 0 else [1, 0]
    data = np.array(data)
    label = np.array(label)
    return [data, label]


def main():
    fl_instance = FederatedLearning(estimator=MyEstimator)

    # load dataset.
    train_data = TxtDataParse(func=image_process)(
        fl_instance.config.train_dataset_url
    )
    x = np.array([tup[0] for tup in train_data])
    y = np.array([tup[1] for tup in train_data])

    fl_instance.initial()
    fl_instance.train(x, y, fed_alg='')


if __name__ == '__main__':
    main()

