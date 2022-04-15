import argparse
from modules.data import *
from modules.bow import *
from config import *
import matplotlib.pyplot as plt
import pdb

def execute(data_path):
    
    # STEP 1: data acquisition. 150 images per class
    print("Taking and splitting images...")
    images_names = find_images(data_path, LABEL_MAPPER)
    train_names, val_names, test_names = train_val_test_split(images_names, TRAIN_N_IMAGES, VAL_N_IMAGES, TEST_N_IMAGES)

    # STEP 2: BOW construction
    print("Building BOW")
    sift_des = DsiftExtractor(8,16,1)
    bow_ = build_bow(train_names, N_CLUSTERS, sift_des)


if __name__ == "__main__":
    # Read user arguments: input path to the images folder.
    parser = argparse.ArgumentParser(description='Scene Classification Project')
    parser.add_argument('-i', '--input', help='<Required> Path to the images'
                        ' directory. Must be a folder', default=None)
    args = parser.parse_args()
    images_folder = args.input

    if not images_folder:
        images_folder = DATA_PATH

    execute(DATA_PATH)