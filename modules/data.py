import os
from config import *
import pdb

def find_images(dirpath, label_mapper):
    """Detect image files contained in a folder.

    Parameters
    ----------
    dirpath : string
        Path name of the folder that contains the images.
    label_mapper : dict
        It associates a class, identified by its name before '-' character 
        with a particular integer label.

    Returns
    -------
    imgfiles : dict
        Full path names of all the image files in `dirpath` (and its
        subfolders) grouped by class name.
    """
    images_dictionary = {}
    for class_name in label_mapper.keys():
        images_aux = []
        root_path = os.path.join(dirpath, class_name)
        for img_name in os.listdir(root_path):
            images_aux.append(os.path.join(root_path, img_name))
            # Stop at MAX_IMAGES_PER_CLASS images
            if len(images_aux) == MAX_IMAGES_PER_CLASS:
                break
        images_dictionary[class_name] = images_aux
    return images_dictionary

def train_val_test_split(imgfiles, train_n, val_n, test_n):
    """Train-val-test split.

    Parameters
    ----------
    imgfiles : dict
        Full path names of all the image files in `dirpath` (and its
        subfolders) grouped by class name.

    Returns
    -------
    {train}{va}{test}_names : dict
    """
    train_names = {}
    val_names = {}
    test_names = {}

    for class_name in imgfiles: 
        train_names[class_name] = imgfiles[class_name][:train_n]
        val_names[class_name] = imgfiles[class_name][train_n : train_n + val_n]
        test_names[class_name] = imgfiles[class_name][train_n + val_n : train_n + val_n + test_n]
    return train_names, val_names, test_names

def get_class(filename, label_mapper):
    """Extract the class integer label from the path of the image.
    
    Parameters
    ----------
    filename : string
        Filename (including path) of a shape sample.
    label_mapper : dict

    Returns
    -------
    class_name : integer
        Class integer to which the shape sample belongs.
    
    """
    label = os.path.split(os.path.split(filename)[0])[1]
    return label_mapper[label]

