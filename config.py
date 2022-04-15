import numpy as np

# DATA 
DATA_PATH = 'data'
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
NANGLES = 8
NBINS = 4
NSAMPLES = NBINS**2
ALPHA = 9.0
ANGLES = np.array(range(NANGLES))*2.0*np.pi/NANGLES

# BOW
N_CLUSTERS = 40
RESIZE_SIZE = 300