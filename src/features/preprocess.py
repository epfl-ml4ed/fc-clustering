
import numpy as np
import pandas as pd

from tslearn.metrics import cdist_dtw
from scipy.spatial import distance




def get_distance_matrix(X, metric='euclidean', window=2, feature_name = ''):
    """
    calculates distance matrix given a metric
    :param X: np.array with students' time-series
    :param metric: str distance metric to compute
    :param window: int for DTW
    :return: np.array with distance matrix
    """

    if metric == 'dtw':
        distance_matrix = cdist_dtw(X,global_constraint='sakoe_chiba',
                                    sakoe_chiba_radius=window)
    else:
        X = X.reshape(-1,1)
        distance_vector = distance.pdist(X, metric)
        distance_matrix = distance.squareform(distance_vector)
    return distance_matrix


def get_affinity_matrix(D, gamma=1):
    """
    calculates affinity matrix from distance matrix
    :param D: np.array distance matrix
    :param gamma: float coefficient for Gaussian Kernel
    :return:
    """
    # fix infinity
    #flat_nan = D[~np.isinf(D)]
    #upper = np.percentile(flat_nan, 100)
    D[np.isinf(D)] = 0 #upper

    S = np.exp(-gamma * D ** 2)

    S[np.isinf(S)] = 0
    S[np.isnan(S)] = 0
    return S



def get_group_feature_kernels_np(data, metric = 'euclidean',gamma = 1,
                        window = 2, group = False,
                        gamma_list = None, window_list = None):

    len_features = data.shape[0]
    students = data.shape[1]
    kernel_distances = np.zeros((len_features,students,students))
    kernel_affinities = np.zeros((len_features,students,students))

    for i in range(len_features):
        if group:
            gamma = gamma_list[i]
            window = window_list[i]

        if len(data.shape)== 3:
            X_formatted = data[i,:,:]
        elif len(data.shape)== 2:
            X_formatted = data[i,:]
            X_formatted.squeeze()
            X_formatted = X_formatted.reshape(-1,1)

        D = get_distance_matrix(X_formatted, metric, window)
        # to prevent overflow
        Dn = normalize(D) #We normalize for the affinity matrix to work
        kernel_distances[i] = Dn
        kernel_affinities[i] = get_affinity_matrix(Dn, gamma)

    combined_affinity = np.sum(kernel_affinities, axis = 0)
    combined_distances = np.sum(kernel_distances, axis = 0)

    #combined_affinity = get_affinity_matrix(normalize(combined_distances),
    #                        np.mean(gamma_list))
    return combined_affinity, combined_distances



def get_feature_kernels_np(X, metric = 'euclidean',gamma = 1, window= 3):
    D = get_distance_matrix(X, metric, window)
    # to prevent overflow
    Dn = normalize(D) #We normalize for the affinity matrix 
    A = get_affinity_matrix(Dn, gamma)

    return A, D



def normalize(distance_matrix):
    # fix infinity
    #flat_nan = distance_matrix[~np.isinf(distance_matrix)]
    #upper = np.percentile(flat_nan, 100)
    distance_matrix[np.isinf(distance_matrix)] = 0

    # robust standarizatoin
    flat_nan = distance_matrix[~np.isnan(distance_matrix)]
    median = np.median(flat_nan)
    q1 = np.percentile(flat_nan, 10)
    q3 = np.percentile(flat_nan, 90)
    IQR = q3 - q1
    if IQR==0:
        IQR = 1
    S =  (distance_matrix - median) / IQR

    # then normalization
    range_matrix = np.max(S) - np.min(S)
    normalized = (S - np.min(S)) / range_matrix
    return normalized


