import os
import numpy as np
import cv2
from scipy.misc import imread
import matplotlib.pyplot as plt


def most_frequent(arr):
    """
    Get most frequent integer of array

    :param arr: 1-D np.ndarray
    :return:
    """
    v, c = np.unique(arr, return_counts=True)
    return int(v[np.argmax(c)])


def infer_border(image):
    """
    Infer border of an input image.

    :param image: np.ndarray

    :return:
    """
    borders = np.concatenate([image[0], image[:, 0], image[-1], image[:, -1]])

    if len(image.shape) == 3:
        border_values = np.reshape(borders, [-1, 3])
        #return np.mean(border_values, axis=0)
        return most_frequent(border_values[0]), most_frequent(border_values[1]), most_frequent(border_values[2])

    else:
        border_values = np.reshape(borders, [-1])
        return most_frequent(border_values)


def rotate_deprecated(image, angle):
    """
    Rotate image by angle degree.

    :param image:
    :param angle:
    :return:
    """
    x, y = image.shape[:2]
    M = cv2.getRotationMatrix2D((x // 2, y // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (x, y), flags=cv2.INTER_CUBIC)


def rotate(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def rescale(image, scale):
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def locate_patch(patch, rel_x, rel_y, shape):
    if len(shape) == 2 and len(patch.shape) == 3:
        shape = np.append(shape, patch.shape[-1])
    result = np.zeros(shape, dtype='uint8')
    result[rel_x: rel_x + patch.shape[0], rel_y: rel_y + patch.shape[1]] = patch
    return result


def hard_threshold(img):
    img[img >= 128] = 255
    return img


class DefectGenerator:
    DEFECT_MAX_PORTION = 1 / 8
    DEFECT_MIN_PORTION = 1 / 10
    DEFECT_TYPES = ['line', 'dot', 'ink', 'curve']

    def __init__(self, debug=False, defect_folder='./defects'):
        self.defect_folder = defect_folder
        self.debug = debug

        if self.debug:
            self.fig, self.axes = plt.subplots(3, 2)
        else:
            self.fig, self.axes = None, None

    def generate(self, shape, defect_type=None):
        """

        :param shape:
        :param defect_type: ['line', 'dot', 'ink', 'curve']
        :return: defect, mask
        """
        if defect_type is None:
            defect_type = np.random.choice(self.DEFECT_TYPES)

        defect, mask = self._load_random_defect(defect_type)
        # mask = hard_threshold(mask)
        if defect_type == 'dot':
            if np.random.binomial(1, 0.8):
                defect = self._random_ellipse()
                mask = 255 - defect  # invert

        if self.debug:
            self.axes[0][0].imshow(defect)
            self.axes[0][1].imshow(mask)

        defect, mask = self._random_rotate(defect, mask)

        if self.debug:
            self.axes[1][0].imshow(defect)
            self.axes[1][1].imshow(mask)

        defect, mask = self._random_scale(defect, mask, shape)
        defect, mask = self._random_position(defect, mask, shape)

        # mask = hard_threshold(mask)

        if self.debug:
            self.axes[2][0].imshow(defect)
            self.axes[2][1].imshow(mask)
            plt.show()

        # binarize
        mask_bool = mask.sum(axis=-1) >= 125 * 3
        mask[mask_bool] = 255
        mask[~mask_bool] = 0
        if len(mask.shape) == 3:
            mask = mask.any(axis=-1) * 255

        return defect, mask

    def _load_random_defect(self, defect_type):
        folder = os.path.join(self.defect_folder, defect_type)
        filenames = os.listdir(folder)
        inds = [filename.split('.')[0].split('_')[1] for filename in filenames if filename.startswith('defect')]
        if len(inds) == 0:
            raise ValueError('no inds')
        else:
            ind = np.random.choice(inds)

        defect_name = 'defect_%s' % ind
        mask_name = 'mask_%s' % ind

        if self.debug:
            self.fig.suptitle(defect_name)

        image = imread(os.path.join(folder, '%s.png' % defect_name))
        mask = imread(os.path.join(folder, '%s.png' % mask_name))

        if image is None:
            raise ValueError('Image not loaded')

        return image, mask

    ##########

    def _random_scale(self, image, mask, shape):
        shape = np.array(shape[:2])
        image_shape = np.array(image.shape[:2])
        min_scale = (shape * DefectGenerator.DEFECT_MIN_PORTION / image_shape).max()
        max_scale = (shape * DefectGenerator.DEFECT_MAX_PORTION / image_shape).min()

        if min_scale < max_scale:
            scale = np.random.uniform(min_scale, max_scale)
        else:
            scale = max_scale

        if self.debug:
            print('scale: %.3f' % scale)

        return rescale(image, scale), rescale(mask, scale)

    def _random_position(self, image, mask, shape):
        shape = np.array(shape)
        assert (image.shape[:2] <= shape[:2] * DefectGenerator.DEFECT_MAX_PORTION).all(), 'defect image too big. shape: %s, image shape: %s' % (shape, image.shape)
        rel_x_max, rel_y_max = shape[:2] - image.shape[:2]
        rel_x = np.random.randint(rel_x_max)
        rel_y = np.random.randint(rel_y_max)

        if self.debug:
            print('(rel_x, rel_y): (%d, %d)' % (rel_x, rel_y))

        image = locate_patch(image, rel_x, rel_y, shape)
        mask = locate_patch(mask, rel_x, rel_y, shape)
        return image, mask

    def _random_rotate(self, image, mask):
        angle = np.random.randint(0, 360)
        if self.debug:
            print('angle: %d' % angle)
        return rotate(image, angle), rotate(mask, angle)

    def _random_ellipse(self):
        r1 = np.random.randint(5, 20)
        r2 = np.random.randint(5, 20)
        background = np.zeros((50, 50, 3), dtype='uint8')
        center = (25, 25)
        return 255 - cv2.ellipse(background, center, (r1, r2),
                                 0, 0, 360, (255, 255, 255), -1)

