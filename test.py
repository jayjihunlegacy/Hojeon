from src.dataset import DataGenerator
from src.classifier.network import BinaryModel, ClassifyModel
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


patch_shape = (128, 128)


def dataset_binary(batch_size, cloth_type=None, defect_type=None, label=None):
    data_gen = DataGenerator()
    while True:
        images, cloth_types0, defect_types0, defected = data_gen.generates(n=batch_size, patch_shape=patch_shape,
                                                                   defected=label,
                                                                   cloth_type=cloth_type,
                                                                   defect_type=defect_type)
        x = images.astype(np.float32)
        x /= 255.

        y = defected.astype(np.float32)
        yield x, y


def dataset_classify(batch_size, defect_type=None):
    data_gen = DataGenerator()
    while True:
        images, cloth_types0, defect_types0, defected = data_gen.generates(n=batch_size,
                                                                           patch_shape=patch_shape,
                                                                           defected=None,
                                                                           cloth_type=None,
                                                                           defect_type=defect_type)
        x = images.astype(np.float32)
        x /= 255.

        y = tf.keras.utils.to_categorical(
            cloth_types0.astype(np.int32), num_classes=4
        )
        yield x, y


def main():
    model = BinaryModel()
    model.main_train(dataset_binary(64))


def train_classify():
    model = ClassifyModel()
    model.main_train(dataset_classify(64))


def show():
    for x, y in dataset_classify(1):
        print(y)
        plt.imshow((x[0] * 255).astype('uint8'))
        plt.show()


if __name__ == '__main__':
    show()
