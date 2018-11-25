from src.defect_generator import DefectGenerator
from src.cloth_generator import ClothGenerator
from src.blender import PoissonBlender

import numpy as np
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.show()


def main():
    cloth_gen    = ClothGenerator()
    def_gen      = DefectGenerator(debug=False)
    blender      = PoissonBlender()

    patch_shape  = (128, 128)
    cloth        = cloth_gen.generate(shape=patch_shape)
    defect, mask = def_gen.generate(shape=patch_shape)
    result       = blender.blend(cloth, defect, mask)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    axes[0][0].imshow(cloth)
    axes[0][0].set_title('cloth')

    axes[0][1].imshow(defect)
    axes[0][1].set_title('defect')

    axes[1][0].imshow(result)
    axes[1][0].set_title('result')

    axes[1][1].imshow(mask, cmap='gray')
    axes[1][1].set_title('mask')
    fig.show()


if __name__ == '__main__':
    main()
