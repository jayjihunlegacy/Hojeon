from src.dataset import DataGenerator

import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.show()


def main():
    patch_shape = (128, 128)
    data_gen = DataGenerator()
    data = data_gen.generates(n=32, patch_shape=patch_shape)
    print(data.shape)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(data[0])
    axes[0][1].imshow(data[1])
    axes[1][0].imshow(data[2])
    axes[1][1].imshow(data[3])
    plt.show()


if __name__ == '__main__':
    main()
