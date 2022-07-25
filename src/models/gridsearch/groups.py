import itertools as it
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append('./../../src/')


from features.preprocess import  get_distance_matrix, \
get_affinity_matrix, normalize

from project_settings import PREFIX, FEATURE_GROUPS, TEST_ID, DATASETS
from etl.postgres_utils import get_select
from features.preprocess import get_feature_kernels_np,format_feature,get_group_feature_kernels_np
from features.load_data import get_data, load_files, get_features
from models.clustering import spectral_clustering
from models.cluster_analysis import get_scores, get_group_composition_np
from models.results import save_metadata, save_labels
from models.gridsearch.utils import get_optimal_params

from visualization.evaluation import plot_mean, plot_heatmap

def create_feature_groups(search = True, image = False, fast = False,
                          save = False, gridsearch = False,
                          feature_groups = FEATURE_GROUPS, datasets = DATASETS):

    for dataset in datasets:
        for group in feature_groups.keys():
            metric = feature_groups[group]['metric']
            features = feature_groups[group]['features']

            feat,  feature_names, feat_labels = load_files(dataset, group, metric)
            data = get_features(feat, feature_names, features, group, metric)

            try:
                optimal_pipeline_groups(dataset, features, data, group,
                feat_labels, metric,search, image, fast, save,  gridsearch)
            except  Exception as e:
                print(e)



def optimal_pipeline_groups(dataset, features, data, group, \
                        feat_labels, metric,search, image, fast, save, gridsearch):

    experiment_name = '{0}_{1}_{2}'.format(PREFIX, dataset,
                                          group).replace(" ", "_").lower()

    if search:
        if gridsearch: #Look for gamma and window
            gammas, windows = get_combinations_features(features,fast)
        else:
            gammas, windows = get_best_gammas_window(features, dataset, group)
            print("Best:", gammas, windows)


        clusters_list = [2,3,4,5,6,7,8,9,10]


        for gamma_list, window_list in it.product(gammas, windows):
            kernel_matrix, distances = get_group_feature_kernels_np(data, metric,
                                        group = True, gamma_list = gamma_list,
                                        window_list = window_list)

            for clusters in clusters_list:
                metadata = {'experiment_name':experiment_name,
                'features':features, 'metric':metric, 'dataset':dataset,
                'gamma':gamma_list, 'window': window_list, 'clusters':clusters
                }
                spectral_feature_groups(kernel_matrix, distances, clusters,
                                        metadata, feat_labels, image)


    if save:
        optimal_clusters, exp_id, gamma_list,\
        window_list = get_optimal_params_group(experiment_name)

        kernel_matrix, _ = get_group_feature_kernels_np(data, metric,
                                    group = True, gamma_list = gamma_list,
                                    window_list = window_list)

        save_group_labels(kernel_matrix,feat_labels, experiment_name, optimal_clusters)

        metadata = {'experiment_name': experiment_name, 'exp_id': exp_id,
                    'features': features, 'metric': metric,
                    'optimal_clusters': optimal_clusters, 'gamma': gamma_list,
                    'window': window_list, 'dataset': dataset, 'group':group}

        save_metadata(metadata, table_name = 'results.group_labels_metadata')




def save_group_labels(kernel_matrix, feat_labels, experiment_name, optimal_clusters):


    kmeans, _, _ = spectral_clustering(kernel_matrix,
                                      optimal_clusters)
    cluster_labels =   kmeans.labels_
    feat_labels['label'] = cluster_labels
    feat_labels['student'] = feat_labels['user_index']

    students_group = list(pd.Series(cluster_labels).value_counts())
    print("Students: {} \n ".format(students_group))

    save_labels(feat_labels, experiment_name)



def spectral_feature_groups(kernel_matrix, distances, clusters, metadata, feat_labels,
                        image, table_name = 'results.gridsearch_fg'):

        kmeans, proj_X, eigenvals_sorted = spectral_clustering(kernel_matrix,clusters)

        cluster_labels =   kmeans.labels_
        feat_labels['label'] = cluster_labels

        metadata = get_scores(kmeans, kernel_matrix, proj_X, distances, metadata)
        metadata['s_eigen'] = eigenvals_sorted[:10].tolist()


        students_group = list(pd.Series(cluster_labels).value_counts())
        metadata['students_group'] = students_group
        metadata['median_groups'] = np.median(students_group)

        save_metadata(metadata,  table_name)



def get_optimal_params_group(experiment_name):
    optimal_clusters, gamma, window, exp_id = get_optimal_params(experiment_name,
                                            table_name = 'results.gridsearch_fg')

    gamma_list =  list(map(float, gamma))
    window_list = list(map(float,window))
    return optimal_clusters, exp_id, gamma_list, window_list



def get_combinations_features(features, fast = True):
    r = len(features)

    gammas_search = [0.05, 0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10] 
    window_search = [1, 2, 3, 4, 5, 6, 7, ,8, 9, 10] 

    if r == 2:
        gammas = [list(x) for x in list(it.product(gammas_search, gammas_search ))]
        windows = [list(x) for x in list(it.product(window_search, window_search ))]
    elif r == 3:
        gammas = [list(x) for x in list(it.product(gammas_search, gammas_search ,gammas_search))]
        windows = [list(x) for x in list(it.product(window_search, window_search ,window_search))]
    return gammas, windows




def get_best_gammas_window(features, dataset, group):
        gamma_list = []
        window_list = []

        for feature in features:
            gamma, window, _ = get_gamma_window(feature, dataset, group)
            gamma_list.append(gamma)
            window_list.append(window)

        return  [gamma_list], [window_list]


def get_gamma_window(feature_name, dataset, group):

    experiment_name = '{0}_{1}_{2}_{3}'.format(TEST_ID,
    dataset, group, feature_name).lower().replace('-','_').replace('.','_')
    query = """
            select * from results.labels_metadata
            where experiment_name = '{experiment_name}'
            order by experiment_date desc
            """.format(experiment_name = experiment_name)
    params = get_select(query)
    if len(params)==0:
        print("Error: This should not happen. No data in metadata for this experiment: ", experiment_name)
        gamma = 1
        window = 2
        clusters = 3
    else:
        gamma = params.gamma[0]
        window = params.window_size[0]
        clusters = params.clusters[0]
        print(experiment_name,gamma, window, clusters )
    return gamma, window, clusters



def main():
    create_feature_groups(search = True, image = False, fast = False,
                          save = True, gridsearch = True,
                          feature_groups = FEATURE_GROUPS, datasets = DATASETS)



if __name__ == "__main__":
    main()