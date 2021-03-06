import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse import linalg
from .cloth_generator import ClothGenerator
from .defect_generator import DefectGenerator


class SeamlessEditingTool:
    def __init__(self, ref, target, mask):
        self.target = target
        self.mask = mask
        self.ref = np.array(Image.fromarray(ref).convert('RGBA'))
        self.target_size_y = self.target.shape[0] - 5
        self.target_size_x = self.target.shape[1] - 5

        self.height, self.width, blank = self.ref.shape
        self.newImage = Image.new('RGB', (self.width, self.height))
        self.maskidx2Corrd = []
        self.Coord2indx = -1 * np.ones([self.height, self.width])
        self.if_strict_interior = []  # left, right, top, botton
        N = 0

        for i, j in zip(*np.nonzero(mask)):
            self.maskidx2Corrd.append([i, j])
            self.if_strict_interior.append([
                i > 0 and self.mask[i - 1, j] == 255,
                i < self.height - 1 and self.mask[i + 1, j] == 255,
                j > 0 and self.mask[i, j - 1] == 255,
                j < self.width - 1 and self.mask[i, j + 1] == 255
            ])
            self.Coord2indx[i][j] = N
            N += 1

        if N == 0:
            raise ValueError('no mask found')

        self.b = np.zeros([N, 3])
        self.A = np.zeros([N, N])

    def create_possion_equation(self):
        # Using the finite difference method
        N = self.b.shape[0]
        for i in range(N):
            # for every pixel in interior and boundary
            self.A[i, i] = 4
            x, y = self.maskidx2Corrd[i]
            if self.if_strict_interior[i][0]:
                self.A[i, int(self.Coord2indx[x - 1, y])] = -1
            if self.if_strict_interior[i][1]:
                self.A[i, int(self.Coord2indx[x + 1, y])] = -1
            if self.if_strict_interior[i][2]:
                self.A[i, int(self.Coord2indx[x, y - 1])] = -1
            if self.if_strict_interior[i][3]:
                self.A[i, int(self.Coord2indx[x, y + 1])] = -1

        # Row-based linked list sparse matrix
        # This is an efficient structure for
        # constructing sparse matrices incrementally.
        self.A = sparse.lil_matrix(self.A, dtype=int)

        for i in range(N):
            flag = np.mod(
                np.array(self.if_strict_interior[i], dtype=int) + 1, 2)
            x, y = self.maskidx2Corrd[i]
            for j in range(3):

                self.b[i, j] = 4 * self.ref[x, y, j] - self.ref[x - 1, y, j] - \
                    self.ref[x + 1, y, j] - self.ref[x, y - 1, j] - self.ref[x, y + 1, j]
                self.b[i, j] += flag[0] * self.target[x - 1, y, j] + \
                    flag[1] * self.target[x + 1, y, j] + flag[2] * \
                    self.target[x, y - 1, j] + \
                    flag[3] * self.target[x, y + 1, j]

    def possion_solver(self):
        self.create_possion_equation()

        # Use Conjugate Gradient iteration to solve A x = b
        x_r = linalg.cg(self.A, self.b[:, 0])[0]
        x_g = linalg.cg(self.A, self.b[:, 1])[0]
        x_b = linalg.cg(self.A, self.b[:, 2])[0]

        self.newImage = self.target

        for i in range(self.b.shape[0]):
            x, y = self.maskidx2Corrd[i]
            self.newImage[x, y, 0] = np.clip(x_r[i], 0, 255)
            self.newImage[x, y, 1] = np.clip(x_g[i], 0, 255)
            self.newImage[x, y, 2] = np.clip(x_b[i], 0, 255)

        self.newImage = Image.fromarray(self.newImage)
        return self.newImage


class PoissonBlender:
    def blend(self, cloth, defect, mask):
        t = SeamlessEditingTool(defect, cloth.copy(), mask)
        return t.possion_solver()


class DataGenerator:
    def __init__(self):
        self.cloth_gen = ClothGenerator()
        self.defec_gen = DefectGenerator(debug=False)
        self.blender = PoissonBlender()

    def generate_failesafe(self, *args, **kwargs):
        while True:
            try:
                return self.generate(*args, **kwargs)
            except ValueError:
                pass

    def generate(self, patch_shape=(128, 128), defected=None, cloth_type=None, defect_type=None):
        if defected is None:
            defected = np.random.randint(2)
        if cloth_type is None:
            cloth_type = np.random.randint(4)
        if defect_type is None:
            defect_type = np.random.randint(4)

        cloth = self.cloth_gen.generate(shape=patch_shape, cloth_type=cloth_type)

        if not defected:
            return cloth, cloth_type, -1, defected

        else:
            defect, mask = self.defec_gen.generate(shape=patch_shape, defect_type=defect_type)
            return self.blender.blend(cloth, defect, mask), cloth_type, defect_type, defected

    def generates(self, n=1, cloth_type=None, defect_type=None, patch_shape=(128, 128), defected=None):
        images = list()
        cloth_types = list()
        defect_types = list()
        defecteds = list()
        for i in range(n):
            image0, cloth_type0, defect_type0, defected0 = self.generate_failesafe(patch_shape,
                                                                     defected=defected,
                                                                     cloth_type=cloth_type,
                                                                     defect_type=defect_type)
            images.append(image0)
            cloth_types.append(cloth_type0)
            defect_types.append(defect_type0)
            defecteds.append(defected0)

        if len(images) != 0:
            return np.stack(images), np.stack(cloth_types), np.stack(defect_types), np.stack(defecteds)

        else:
            return np.zeros((0,) + patch_shape + (3,), dtype=np.float32), np.zeros((0,)), np.zeros((0,)), np.zeros((0,))
