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
    sift_des = DsiftExtractor(GRIDSPACING, PATCHSIZE, 1)
    if os.path.exists(IMAGES_FEATURES_PATH) and os.path.exists(BOW_PATH):
        print("Loading BOW and imgs. features...")
        im_features = np.loadtxt(IMAGES_FEATURES_PATH)
        kmeans_bow = pickle.load(open(BOW_PATH, "rb"))
    else:
        # STEP 2: BOW construction
        print("Building BOW...")
        # Get dense response for each image
        descriptor_list, labels = obtain_dense_features(train_names, sift_des)
        # Build BOW
        kmeans_bow = build_bow(descriptor_list, N_CLUSTERS)

        pickle.dump(kmeans_bow, open(BOW_PATH, "wb")) # save bow

        # STEP 3: Describe each image by its histogram of visual features ocurrences.
        print("Histogram features...")
        im_features = extractFeatures(kmeans_bow, descriptor_list, labels, N_CLUSTERS)
        # Save features
        np.savetxt(IMAGES_FEATURES_PATH, im_features)

    # STEP 4: Train 
    if os.path.exists(MODEL_PATH):
        print("Loading SVC model...")
        model = pickle.load(open(MODEL_PATH, "rb"))
    else:
        print("Training SVC model...")
        model = train(im_features)
        pickle.dump(model, open(MODEL_PATH, "wb")) # save model
    
    # STEP5 5: TEST
    descriptor_list_test, labels_test = obtain_dense_features(test_names, sift_des)
    test_im_features = extractFeatures(kmeans_bow, descriptor_list_test, labels_test, N_CLUSTERS)
    test(test_im_features, model)
    

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