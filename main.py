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
    if os.path.exists(TRAIN_IMAGES_FEATURES_PATH) and os.path.exists(VAL_IMAGES_FEATURES_PATH) and os.path.exists(TEST_IMAGES_FEATURES_PATH):
        print("Loading final imgs. features...")
        train_im_features = np.loadtxt(TRAIN_IMAGES_FEATURES_PATH)
        val_im_features = np.loadtxt(VAL_IMAGES_FEATURES_PATH)
        test_im_features = np.loadtxt(TEST_IMAGES_FEATURES_PATH)
    else:
        sift_des = DsiftExtractor(GRIDSPACING, PATCHSIZE, 1)
        # STEP 2: BOW construction
        print("Building BOW...")
        # Get dense response for each image
        train_descriptor_list, train_labels = obtain_dense_features(train_names, sift_des)
        # Build BOW
        kmeans_bow = build_bow(train_descriptor_list, N_CLUSTERS)

        pickle.dump(kmeans_bow, open(BOW_PATH, "wb")) # save bow

        # STEP 3: Describe each image by its histogram of visual features ocurrences.
        print("TRAIN IMAGES...")
        train_im_features = extractFeatures(kmeans_bow, train_descriptor_list, train_labels, N_CLUSTERS)
        # Save features
        np.savetxt(TRAIN_IMAGES_FEATURES_PATH, train_im_features)

        # Same with val and test
        print("TEST IMAGES...")
        val_descriptor_list, val_labels = obtain_dense_features(val_names, sift_des)
        val_im_features = extractFeatures(kmeans_bow, val_descriptor_list, val_labels, N_CLUSTERS)
        np.savetxt(VAL_IMAGES_FEATURES_PATH, val_im_features)

        print("VAL IMAGES...")
        test_descriptor_list, test_labels = obtain_dense_features(test_names, sift_des)
        test_im_features = extractFeatures(kmeans_bow, test_descriptor_list, test_labels, N_CLUSTERS)
        np.savetxt(TEST_IMAGES_FEATURES_PATH, test_im_features)

    # STEP 4: Train and TEST
    print("Training and testing SVC model...")
    model = train_test(train_im_features, val_im_features, test_im_features)
    pickle.dump(model, open(MODEL_PATH, "wb")) # save model
    

if __name__ == "__main__":
    # Read user arguments: input path to the images folder.
    parser = argparse.ArgumentParser(description='Scene Classification Project')
    parser.add_argument('-i', '--input', help='<Required> Path to the images'
                        ' directory. Must be a folder', default=None)
    args = parser.parse_args()
    images_folder = args.input

    if not images_folder:
        images_folder = DATA_PATH

    np.random.seed(88)
    execute(DATA_PATH)