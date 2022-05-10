from time import perf_counter
import cv2
from skimage.feature import local_binary_pattern
from config import *
import pdb

def computeLBP(images_names):
    """ Compute LBP descriptor for gray image.

    Parameters
    ----------
    images_names : dict
        Images groupped by class

    Returns
    -------
    feature_matrix_base : np.ndarray
        Matrix of size N x n where:
        - N is the number of patterns, 1 row for each image.
        - n is the number of atributes. 
    """

    # ESTADISTICAS ###
    init_time = perf_counter()
    no_images = 0
    _aux_ = list(images_names.keys())[0]
    n_totales = len(images_names.keys()) * len(images_names[_aux_])
    print(f'Computing lbp... 0/{n_totales}')

    matrix_features = []

    for class_name in images_names.keys(): 
        for img_name in images_names[class_name]:
            img = cv2.imread(img_name, 0)
            lbp = local_binary_pattern(img, N_POINTS, RADIUS, METHOD)
            # Histogram
            hist, _ = np.histogram(lbp, bins=BINS)
            matrix_features.append(hist)

            # ESTADISTICAS ###
            no_images += 1
            if (no_images % 50) == 0:
                actual_time = perf_counter() - init_time
                print('Computing LBP... %d/%d ( eta: %.1f s )' % (no_images, n_totales, (n_totales - no_images)  * actual_time / no_images))
    matrix_features = np.array(matrix_features)
    return matrix_features