from src.defect_generator import DefectGenerator
from src.composer import PoissonComposer
import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.show()


def main():
    cloth_gen    = ClothGenerator()
    def_gen      = DefectGenerator(debug=True)
    composer     = PoissonComposer()

    patch_shape  = (128, 128)
    cloth        = cloth_gen.generate(shape=patch_shape)
    defect, mask = def_gen.generate(shape=patch_shape)
    result       = composer.compose(cloth, defect, mask)


if __name__ == '__main__':
    main()
