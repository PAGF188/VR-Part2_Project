from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from config import *
from sklearn.svm import LinearSVC
from sklearn.metrics import *
import seaborn as sns


def plotHistogram(im_features, no_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.savefig(os.path.join(RESULT_PATH, "histogram.png")); plt.clf()

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

    plotHistogram(x_data, N_CLUSTERS) # Guardar en vez de plot

    model = LinearSVC(random_state=0, tol=1e-5)
    model.fit(x_data, y_data)
    return model


def test(feature_matrix, model):
    # Label
    y_data = feature_matrix[:, -1]
    # Features
    x_data = feature_matrix[:, 0:-1]

    z = model.predict(x_data)

    acc = 100 * accuracy_score(y_data, z)
    print(acc)

    cf = confusion_matrix(y_data, z)
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig(os.path.join(RESULT_PATH, "cf.png")); plt.clf()






