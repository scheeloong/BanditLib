from collections import Counter
from math import log
import numpy as np
from random import *
from custom_errors import FileExists

def getPoolArticleArr(pool_articles):
    '''
    Return all articles in pool_articles to an array
    The array contains the feature vectors of each article directly
    '''
    article_arr = []
    for x in pool_articles:
        article_arr.append(np.array(x.featureVector))
    return np.array(article_arr)


def gaussianFeature(dimension, argv):
    '''
    '''
    mean = argv['mean'] if 'mean' in argv else 0
    std = argv['std'] if 'std' in argv else 1
    mean_vector = np.ones(dimension) * mean
    stdev = np.identity(dimension) * std
    vector = np.random.multivariate_normal(np.zeros(dimension), stdev)
    l2_norm = np.linalg.norm(vector, ord=2)
    if 'l2_limit' in argv and l2_norm > argv['l2_limit']:
        "This makes it uniform over the circular range"
        vector = (vector / l2_norm)
        vector = vector * (random())
        vector = vector * argv['l2_limit']
    if mean is not 0:
        vector = vector + mean_vector
    vectorNormalized = []
    for i in range(len(vector)):
        vectorNormalized.append(vector[i] / sum(vector))
    return vectorNormalized

def featureUniform(dimension, argv=None):
    '''
    '''
    vector = np.array([random() for _ in range(dimension)])
    l2_norm = np.linalg.norm(vector, ord=2)
    vector = vector / l2_norm
    return vector


def getBatchStats(arr):
    '''
    '''
    return np.concatenate((np.array([arr[0]]), np.diff(arr)))


def checkFileExists(filename):
    '''
    Returns 1 if a file exist
    '''
    try:
        with open(filename, 'r'):
            return 1
    except IOError:
        return 0


def fileOverWriteWarning(filename, force):
    '''
    Print a warning if going to overwrite a file
    otherwise, if not supposed to overwrite, printError if file already exist
    '''
    if checkFileExists(filename):
        if force == True:
            print "Warning : fileOverWriteWarning %s" % (filename)
        else:
            raise FileExists(filename)

def vectorize(M):
    '''
    Vectorize a 2D matrix to 1D vector
    Change from (N, M) to (N*M)
    '''
    return np.reshape(M.T, M.shape[0] * M.shape[1])


def matrixize(V, C_dimension):
    '''
    Reshape a 1D vector V to a 2D Matrix of size (V.dim/newHiddenDim, newHiddenDim)
    '''
    return np.transpose(np.reshape(V, (int(len(V) / C_dimension), C_dimension)))
