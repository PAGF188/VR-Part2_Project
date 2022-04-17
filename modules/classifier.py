from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from config import *

def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()

def train(feature_matrix):
    """Obtain SVC model.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Matrix of size N x n+1 where:
        - N is the number of patterns, 1 row for each image.
        - n is the number of atributes (computed shape measures).
          The last column codify the class.r

    Returns
    -------
    """

    # Label
    y_data = feature_matrix[:, -1]
    # Features
    x_data = feature_matrix[:, 0:-1]


    # Normalize features
    scale = StandardScaler().fit(x_data)        
    x_data = scale.transform(x_data)

    plotHistogram(x_data, N_CLUSTERS)


