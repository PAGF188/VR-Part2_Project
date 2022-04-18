import numpy as np
from scipy import signal
from config import *
import pdb
import cv2
from time import perf_counter
from sklearn.cluster import KMeans

def gen_dgauss(sigma):
    '''
    generating a derivative of Gauss filter on both the X and Y direction.
    '''
    fwid = np.int(2*np.ceil(sigma))
    G = np.array(range(-fwid,fwid+1))**2
    G = G.reshape((G.size,1)) + G
    G = np.exp(- G / 2.0 / sigma / sigma)
    G /= np.sum(G)
    GH,GW = np.gradient(G)
    GH *= 2.0/np.sum(np.abs(GH))
    GW *= 2.0/np.sum(np.abs(GW))
    return GH,GW

class DsiftExtractor:
    '''
    The class that does dense sift feature extractor.
    Sample Usage:
        extractor = DsiftExtractor(gridSpacing,patchSize,[optional params])
        feaArr,positions = extractor.process_image(Image)
    '''
    def __init__(self, gridSpacing, patchSize,
                 nrml_thres = 1.0,\
                 sigma_edge = 0.8,\
                 sift_thres = 0.2):
        '''
        gridSpacing: the spacing for sampling dense descriptors
        patchSize: the size for each sift patch
        nrml_thres: low contrast normalization threshold
        sigma_edge: the standard deviation for the gaussian smoothing before computing the gradient
        sift_thres: sift thresholding (0.2 works well based on Lowe's SIFT paper)
        '''
        self.gS = gridSpacing
        self.pS = patchSize
        self.nrml_thres = nrml_thres
        self.sigma = sigma_edge
        self.sift_thres = sift_thres
        # compute the weight contribution map
        sample_res = self.pS / np.double(NBINS)
        sample_p = np.array(range(self.pS))
        sample_ph, sample_pw = np.meshgrid(sample_p,sample_p)
        sample_ph.resize(sample_ph.size)
        sample_pw.resize(sample_pw.size)
        bincenter = np.array(range(1,NBINS*2,2)) / 2.0 / NBINS * self.pS - 0.5 
        bincenter_h, bincenter_w = np.meshgrid(bincenter,bincenter)
        bincenter_h.resize((bincenter_h.size,1))
        bincenter_w.resize((bincenter_w.size,1))
        dist_ph = abs(sample_ph - bincenter_h)
        dist_pw = abs(sample_pw - bincenter_w)
        weights_h = dist_ph / sample_res
        weights_w = dist_pw / sample_res
        weights_h = (1-weights_h) * (weights_h <= 1)
        weights_w = (1-weights_w) * (weights_w <= 1)
        # weights is the contribution of each pixel to the corresponding bin center
        self.weights = weights_h * weights_w
        #pyplot.imshow(self.weights)
        #pyplot.show()
        
    def process_image(self, image, positionNormalize = True, verbose = False):
        '''
        processes a single image, return the locations and the values of detected SIFT features.
        image: a M*N image which is a numpy 2D array. Color images will automatically be converted to grayscale.
        positionNormalize: whether to normalize the positions to [0,1]. If False, the pixel-based positions of the
            top-right position of the patches is returned.
        
        Return values:
        feaArr: the feature array, each row is a feature positions: the positions of the features
        '''

        image = image.astype(np.double)
        if image.ndim == 3:
            # we do not deal with color images.
            image = np.mean(image,axis=2)
        # compute the grids
        H,W = image.shape
        gS = self.gS
        pS = self.pS
        remH = np.mod(H-pS, gS)
        remW = np.mod(W-pS, gS)
        offsetH = int(remH/2)
        offsetW = int(remW/2)
        gridH,gridW = np.meshgrid(range(offsetH,H-pS+1,gS), range(offsetW,W-pS+1,gS))
        gridH = gridH.flatten()
        gridW = gridW.flatten()
        if verbose:
            print('Image: w {}, h {}, gs {}, ps {}, nFea {}'.format(W,H,gS,pS,gridH.size))
        feaArr = self.calculate_sift_grid(image,gridH,gridW)
        feaArr = self.normalize_sift(feaArr)
        if positionNormalize:
            positions = np.vstack((gridH / np.double(H), gridW / np.double(W)))
        else:
            positions = np.vstack((gridH, gridW))
        return feaArr, positions

    def calculate_sift_grid(self,image,gridH,gridW):
        '''
        This function calculates the unnormalized sift features
        It is called by process_image().
        '''
        H,W = image.shape
        Npatches = gridH.size
        feaArr = np.zeros((Npatches,NSAMPLES*NANGLES))

        # calculate gradient
        GH,GW = gen_dgauss(self.sigma)
        IH = signal.convolve2d(image,GH,mode='same')
        IW = signal.convolve2d(image,GW,mode='same')
        Imag = np.sqrt(IH**2+IW**2)
        Itheta = np.arctan2(IH,IW)
        Iorient = np.zeros((NANGLES,H,W))
        for i in range(NANGLES):
            Iorient[i] = Imag * np.maximum(np.cos(Itheta - ANGLES[i])**ALPHA,0)
            #pyplot.imshow(Iorient[i])
            #pyplot.show()
        for i in range(Npatches):
            currFeature = np.zeros((NANGLES,NSAMPLES))
            for j in range(NANGLES):
                currFeature[j] = np.dot(self.weights,\
                        Iorient[j,gridH[i]:gridH[i]+self.pS, gridW[i]:gridW[i]+self.pS].flatten())
            feaArr[i] = currFeature.flatten()
        return feaArr

    def normalize_sift(self,feaArr):
        '''
        This function does sift feature normalization following David Lowe's definition 
         (normalize length -> thresholding at 0.2 -> renormalize length)
        '''
        siftlen = np.sqrt(np.sum(feaArr**2,axis=1))
        hcontrast = (siftlen >= self.nrml_thres)
        siftlen[siftlen < self.nrml_thres] = self.nrml_thres
        # normalize with contrast thresholding
        feaArr /= siftlen.reshape((siftlen.size,1))
        # suppress large gradients
        feaArr[feaArr>self.sift_thres] = self.sift_thres
        # renormalize high-contrast ones
        feaArr[hcontrast] /= np.sqrt(np.sum(feaArr[hcontrast]**2,axis=1)).\
                reshape((feaArr[hcontrast].shape[0],1))
        return feaArr

def obtain_dense_features(images_names, des):
    """Apply descriptor to each image.

    Parameters
    ----------
    images_names : dict
        Images groupped by class
    des : feature dense descriptor

    Returns
    -------
    descriptor_list: list (len = no. images)
        Raw features of each image.
    labels : nd.array
        True output of each image. Same order than descriptor_list
    """

    # ESTADISTICAS ###
    init_time = perf_counter()
    no_images = 0
    n_totales = len(images_names.keys()) * TRAIN_N_IMAGES
    print(f'Computing dense response... 0/{n_totales}')

    descriptor_list = []  # len = no. images
    labels = np.array([]) # labels. Same order than descriptor_list

    for class_name in images_names.keys(): 
        labels = np.concatenate([labels, np.repeat(int(LABEL_MAPPER[class_name]), len(images_names[class_name]))])
        for img_name in images_names[class_name]:
            img = cv2.imread(img_name, 0)
            img = cv2.resize(img, RESIZE_SIZE)   # TO RESIZE. CONSIDERAR ELIMINARLO
            feaArr, _ = des.process_image(img)
            descriptor_list.append(feaArr)

            # ESTADISTICAS ###
            no_images += 1
            if (no_images % 50) == 0:
                actual_time = perf_counter() - init_time
                print('Computing dense response... %d/%d ( eta: %.1f s )' % (no_images, n_totales, (n_totales - no_images)  * actual_time / no_images))    
    
    return descriptor_list, labels

def _vstackDenseFeatures(descriptor_list):
    """ Vstack the list of features by image

    Parameters
    ----------
    descriptor_list : list (len = no. images)
        Raw features of each image.

    Returns
    -------
    descriptors : nd.array
        1 row per feature
    """
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors

def build_bow(descriptor_list, n_clusters):
    """ Build bow clustering features

    Parameters
    ----------
    descriptor_list : list (len = no. images)
        Raw features of each image.
    n_clusters : int

    Returns
    -------
    kmeans: <class 'sklearn.cluster._kmeans.KMeans'>
        Bag of visual words
    """
    # Vstack descriptor_list
    print("Vstacking results....")
    descriptors = _vstackDenseFeatures(descriptor_list)

    # Clustering to obtain bow
    print("Clustering....")
    kmeans_bow = KMeans(n_clusters=n_clusters).fit(descriptors)
    return kmeans_bow
    

def extractFeatures(kmeans, descriptor_list, labels, no_clusters):
    """ Describe each image by its histogram of visual features ocurrences.

    Parameters
    ----------
    kmeans: <class 'sklearn.cluster._kmeans.KMeans'>
        Bag of visual words
    descriptor_list : list (len = no. images)
        Raw features of each image. 
    labels : nd.array
        True output of each image. Same order than descriptor_list
    n_clusters : int

    Returns
    -------
    im_features: np.array
        Matrix of size N x n+1 where:
        - N is the number of patterns, 1 row for each image.
        - n is the number of atributes (computed shape measures). The last column codify the class.
    """
    image_count = len(descriptor_list)
    im_features = np.array([np.zeros(no_clusters+1) for i in range(image_count)])
    no_images = 0
    init_time = perf_counter()
    for i in range(image_count):
        predictions = kmeans.predict(descriptor_list[i])
        unique, counts = np.unique(predictions, return_counts=True)
        for idx_, sum_ in zip(unique, counts):
            im_features[i][idx_] += sum_
        # append label
        im_features[i][-1] = labels[i]

        no_images += 1
        if (no_images % 50) == 0:
                actual_time = perf_counter() - init_time
                print('Encoding images... %d/%d ( eta: %.1f s )' % (no_images, image_count, (image_count - no_images)  * actual_time / no_images))   

    return im_features