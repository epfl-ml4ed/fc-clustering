from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from datetime import datetime
import pickle
from pathlib import Path

import random
import numpy as np
import seaborn as sns
import pandas as pd
import itertools as it

import sys
sys.path.append('./../../src/')

from project_settings import PREFIX, FEATURE_GROUPS, TEST_ID, SEED, DATASETS
from etl.postgres_utils import get_select, execute_query
from features.load_data import get_data
from models.cluster_analysis import get_scores,  get_pairwise_wilcoxon, get_group_composition_np
from models.results import save_metadata, save_labels
from models.gridsearch.utils import  get_optimal_params
from visualization.evaluation import plot_boxplot, plot_barplot
pd.options.mode.chained_assignment = None  # default='warn'


def run_all_profiles(search = True, datasets = DATASETS,
                    feature_groups=FEATURE_GROUPS,
                    features_names = None):

    if not features_names:
        features_names = list(feature_groups.keys())

    for dataset in datasets:
        optimal_pipeline_profiles(dataset,  feature_groups = feature_groups,
                                    search = search, features_names = features_names)




def optimal_pipeline_profiles(dataset, feature_groups=FEATURE_GROUPS,
search = True, features_names = list(FEATURE_GROUPS.keys())):

    experiment_name = '{0}_{1}_{2}'.format(PREFIX,
                                          dataset,
                                          'profile').replace(" ", "_").lower()

    df = get_data(dataset = dataset, metric = 'labels',
                  feature_groups =features_names)

    df_feat = df[features_names]

    if search:
        gridsearch_kmodes(df, df_feat, experiment_name, dataset)

    optimal_clusters, _, _, _ = get_optimal_params(experiment_name,
                                    table_name = 'results.gridsearch_kmodes')

    # Runs spectral clustering with optimal parameters 
    save_profile_labels(df, df_feat, experiment_name, optimal_clusters, dataset)



def gridsearch_kmodes(df, df_feat, experiment_name,  dataset,
                        cluster_list = [3,4,5]):
                        #cluster_list = [2,3,4,5,6]):
    for clusters in cluster_list:
        km = KModes(n_clusters=clusters, init='Huang',
                    n_init=150,  max_iter = 100, random_state = SEED)

        cluster_labels= km.fit_predict(df_feat)
        df['label'] = cluster_labels

        metadata = {'experiment_name':experiment_name,
         'dataset':dataset, 'clusters':clusters,
         }

        metadata['s_sil_km']= silhouette_score(df_feat,  cluster_labels)

        metadata = get_group_composition_np(df, clusters, metadata)

        students_group = list(pd.Series(cluster_labels).value_counts())
        metadata['students_group'] = students_group
        metadata['median_groups'] = np.median(students_group)

        save_metadata(metadata,  table_name ='results.gridsearch_kmodes')



def save_profile_labels(df, df_feat, experiment_name, optimal_clusters, dataset):

    km = KModes(n_clusters=optimal_clusters, init='Huang', n_init=150,
                max_iter = 500, random_state = 123)

    cluster_labels= km.fit_predict(df_feat)
    df['label'] = cluster_labels

    plot_boxplot(df,  feature = 'label_grade', dataset = dataset,
                    group = 'profiles', bygroup = True)

    df['label_pass_fail'] = 0
    df['label_pass_fail'][df['label_grade'] < 4] = 1

    plot_barplot(df,  feature = 'label_pass_fail', dataset = dataset,
                    group = 'profiles', bygroup = True)

    plot_barplot(df,  feature = 'label_dropout', dataset = dataset,
                    group = 'profiles', bygroup = True)

    plot_boxplot(df,  feature = 'label_stopout', dataset = dataset,
                    group = 'profiles', bygroup = True)

    save_data(experiment_name, cluster_labels,km.cluster_centroids_)
    save_labels(df, experiment_name)

    metadata = {'experiment_name': experiment_name, 'exp_id': datetime.now(),
                'features': list(df_feat.columns), 'metric': 'profiles',
                'optimal_clusters': optimal_clusters, 'gamma': None,
                'window': None,  'group':'profiles'}
    save_metadata(metadata, table_name = 'results.group_labels_metadata')




def load_data(experiment_name):
    folder_dir = """./../results/{prefix}/temp_data/{experiment_name}
    """.format(prefix = TEST_ID,experiment_name=experiment_name )

    data_dir = "{0}/cluster_labels.txt".format(folder_dir)
    with open(data_dir, "rb") as fp:   #Pickling
        cluster_labels = pickle.load(fp)

    data_dir = "{0}/centroids.npy".format(folder_dir)
    centroids =np.load(data_dir)
    return cluster_labels,centroids



def save_data(experiment_name, cluster_labels,centroids):
    folder_dir = """./../results/{prefix}/temp_data/{experiment_name}
    """.format(prefix = TEST_ID,experiment_name=experiment_name )
    Path(folder_dir).mkdir(parents=True, exist_ok=True)

    data_dir = "{0}/cluster_labels.txt".format(folder_dir)

    with open(data_dir, "wb") as fp:   #Pickling
        pickle.dump(cluster_labels, fp)

    data_dir = "{0}/centroids".format(folder_dir)
    np.save(data_dir, centroids)





def main():
    run_all_profiles(search = True, datasets = DATASETS,
                    feature_groups=FEATURE_GROUPS,
                    features_names = None)



if __name__ == "__main__":
    main()