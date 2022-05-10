import numpy as np
import os
# DATA 
DATA_PATH = 'data'
RESULT_PATH = 'results'
BOW_PATH = os.path.join(RESULT_PATH, 'BOW.pkl')
MODEL_PATH = os.path.join(RESULT_PATH, 'model.pkl')
TRAIN_IMAGES_FEATURES_PATH = os.path.join(RESULT_PATH, 'train_images_features.txt')
VAL_IMAGES_FEATURES_PATH = os.path.join(RESULT_PATH, 'val_images_features.txt')
TEST_IMAGES_FEATURES_PATH = os.path.join(RESULT_PATH, 'test_images_features.txt')

MAX_IMAGES_PER_CLASS = 150
TRAIN_N_IMAGES = 80
VAL_N_IMAGES = 20
TEST_N_IMAGES = 50


LABEL_MAPPER = {
    'bakery': 0,
    'bathroom': 1,
    'bookstore': 2,
    'casino': 3,
    'corridor': 4,
    'gym': 5,
    'kitchen': 6,
    'locker_room': 7,
    'subway': 8,
    'winecellar': 9
}


# SIFT DENSE APLICATION
GRIDSPACING = 8
PATCHSIZE = 16

NANGLES = 8
NBINS = 4
NSAMPLES = NBINS**2
ALPHA = 9.0
ANGLES = np.array(range(NANGLES))*2.0*np.pi/NANGLES

# BOW
N_CLUSTERS = 100
RESIZE_SIZE = (250, 250)

# LBP
RADIUS = 3
N_POINTS = 8 * RADIUS
METHOD = 'uniform'
BINS = 25