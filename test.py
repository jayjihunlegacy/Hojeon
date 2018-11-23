from src.defect_generator import DefectGenerator
from src.cloth_generator import ClothGenerator
from src.composer import PoissonComposer
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.show()


def main():
    cloth_gen    = ClothGenerator()
    def_gen      = DefectGenerator(debug=False)
    composer     = PoissonComposer()

    patch_shape  = (128, 128)
    cloth        = cloth_gen.generate(shape=patch_shape)
    defect, mask = def_gen.generate(shape=patch_shape)
    result       = composer.compose(cloth, defect, mask)

    fig, axes = plt.subplots(nrows=3, ncols=2)
    axes[0][0].imshow(cloth)
    axes[1][0].imshow(defect)
    axes[2][0].imshow(mask)
    fig.show()


if __name__ == '__main__':
    main()
