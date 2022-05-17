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

def train_test(train_im_features, val_im_features, test_im_features):
    """Obtain SVC model. Evaluate it

    Parameters
    ----------
    train_im_features : np.ndarray
        Matrix of size N x n+1 where:
        - N is the number of patterns, 1 row for each image.
        - n is the number of atributes (computed shape measures).
          The last column codify the class.r
    val_im_features : same for val images
    test_im_features :  same for test images

    Returns
    -------
    """

    y_train = train_im_features[:, -1]
    x_train = train_im_features[:, 0:-1]

    y_val = val_im_features[:, -1]
    x_val = val_im_features[:, 0:-1]

    y_test = test_im_features[:, -1]
    x_test = test_im_features[:, 0:-1]

    # Normalize features
    scale = StandardScaler().fit(x_train)        
    x_train = scale.transform(x_train)
    x_val = scale.transform(x_val)
    x_test = scale.transform(x_test)

    #plotHistogram(x_train, N_CLUSTERS) # Guardar en vez de plot

    # TRAIN
    # Sintonizacion lambda y sigma
    print("Sintonizacion:")
    vL=2.**np.arange(-5, 16, 2)
    vG=2.**np.arange(-7, 8, 2)

    kappa_sintonizacion=np.zeros((len(vL), len(vG))); 
    kappa_mellor=-np.Inf; L_mellor=vL[0]; G_mellor=vG[0]
    print('%10s %15s %10s %10s'%('Lambda','Gamma', 'Kappa (%)', 'Mejor'))

    for i,L in enumerate(vL):
        for j,G in enumerate(vG):
            modelo=SVC(C=L, kernel ='rbf', gamma=G, verbose=False).fit(x_train, y_train)
            z = modelo.predict(x_val)
            kappa = cohen_kappa_score(y_val, z) * 100
            kappa_sintonizacion[i,j] = kappa
            if kappa>kappa_mellor:
                kappa_mellor=kappa; L_mellor=L; G_mellor = G
            print('%.2f %15g %10.1f %10.1f'%(L,G, kappa, kappa_mellor))
    print('L_mejor=%g, G_mejor=%g, kappa=%.2f%%\n'%(L_mellor, G_mellor, kappa_mellor))

    # MODELO CON MEJOR PARAMS
    X = np.vstack((x_train, x_val))
    Y = np.concatenate((y_train, y_val))
    model = SVC(C=L_mellor, kernel ='rbf', gamma=G_mellor, verbose=False).fit(X,Y)

    z = model.predict(x_test)
    acc = 100 * accuracy_score(y_test, z)
    print(acc)
    cf = confusion_matrix(y_test, z)
    cf_image = sns.heatmap(cf, cmap='Blues', annot=True, fmt='g')
    figure = cf_image.get_figure()    
    figure.savefig(os.path.join(RESULT_PATH, "cf.png")); plt.clf()
    
    return model





