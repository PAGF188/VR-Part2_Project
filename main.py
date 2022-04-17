import argparse
from modules.data import *
from modules.bow import *
from modules.classifier import *
from config import *
import matplotlib.pyplot as plt
import pdb
import pickle
def execute(data_path):
    
    # STEP 1: data acquisition. 150 images per class
    print("Taking and splitting images...")
    images_names = find_images(data_path, LABEL_MAPPER)
    train_names, val_names, test_names = train_val_test_split(images_names, TRAIN_N_IMAGES, VAL_N_IMAGES, TEST_N_IMAGES)

    # STEP 2: BOW construction + STEP 3: Describe each image by its histogram of visual features ocurrences.
    if os.path.exists(IMAGES_FEATURES_PATH):
        im_features = np.loadtxt(IMAGES_FEATURES_PATH)
    else:
        # STEP 2: BOW construction
        sift_des = DsiftExtractor(GRIDSPACING, PATCHSIZE, 1)
        print("Building BOW...")
        kmeans_bow, descriptor_list, labels = build_bow(train_names, N_CLUSTERS, sift_des)
        pickle.dump(kmeans_bow, open(BOW_PATH, "wb")) # save bow

        # STEP 3: Describe each image by its histogram of visual features ocurrences.
        print("Extracting images features...")
        im_features = extractFeatures(kmeans_bow, descriptor_list, labels, N_CLUSTERS)
        # Save features
        np.savetxt(IMAGES_FEATURES_PATH, im_features)

    # STEP4: Train 
    model = train(im_features)
    

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